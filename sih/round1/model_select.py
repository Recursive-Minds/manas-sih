"""
Speed-model selection for the production profile, plus a simple mean ensemble.

`velocity_checkpoint` in the active Round1Config (config/round1/production.json once
promoted) chooses the speed model used when a caller does not pass model_path:
  ""                       -> canonical models/checkpoints/best_moe_velocity_model.pt (old behaviour)
  "path/a.pt"              -> that checkpoint
  "path/a.pt,path/b.pt"    -> mean ensemble of the listed checkpoints
A configured file that does not exist raises (no silent fallback to the canonical model).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np


def _abs(path: str, root_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root_dir, path)


def resolve_velocity_checkpoint(root_dir: str) -> Optional[str]:
    from sih.round1.config import get_active_config
    spec = (get_active_config().velocity_checkpoint or "").strip()
    if not spec:
        return None
    paths = [_abs(p.strip(), root_dir) for p in spec.split(",") if p.strip()]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"[round1] configured velocity_checkpoint not found: {missing}")
    return ",".join(paths)


def load_mean_ensemble(spec: str, device: Any, root_dir: str) -> Tuple[Any, np.ndarray, np.ndarray, str]:
    import torch.nn as nn
    from sih.models.inference import load_ai_model

    paths = [_abs(p.strip(), root_dir) for p in spec.split(",") if p.strip()]
    members, mean, std = [], None, None
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[round1] ensemble member not found: {p}")
        m, nm, ns, mtype = load_ai_model(device, model_path=p, root_dir=root_dir)
        if mtype != "moe":
            raise ValueError(f"[round1] ensemble supports MoE checkpoints only, got {mtype} for {p}")
        if mean is None:
            mean, std = nm, ns
        elif not (np.allclose(mean, nm, atol=1e-6) and np.allclose(std, ns, atol=1e-6)):
            raise ValueError(f"[round1] ensemble members use different normalisation: {p}")
        members.append(m)
    print(f"[AI Model] [ROUND1] Mean ensemble of {len(members)} MoE checkpoints")
    return MeanMoEEnsemble(members).to(device).eval(), mean, std, "moe"


def _make_module():
    import torch
    import torch.nn as nn

    class _MeanMoEEnsemble(nn.Module):
        """Same call signature as BayesianMoEFusion: forward(x_short, x_long) -> (v, var, diag)."""

        def __init__(self, members: List[nn.Module]):
            super().__init__()
            self.members = nn.ModuleList(members)

        def forward(self, x_short, x_long):
            outs = [m(x_short, x_long) for m in self.members]
            vs = torch.stack([o[0] for o in outs], dim=0)
            vars_ = torch.stack([o[1] for o in outs], dim=0)
            v = vs.mean(dim=0)
            var = vars_.mean(dim=0) + vs.var(dim=0, unbiased=False)   # total variance
            diag = dict(outs[0][2])
            diag["v_members"] = vs
            return v, var, diag

    return _MeanMoEEnsemble


def MeanMoEEnsemble(members):  # noqa: N802 - factory keeps torch import lazy
    return _make_module()(members)
