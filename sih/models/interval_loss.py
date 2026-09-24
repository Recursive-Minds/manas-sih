"""
T6 - Interval (distance-over-blackout) loss with pre-blackout scale emulation.

The deployed pipeline never uses a single speed sample: it (1) learns a scale
alpha from ~20 s of GNSS before the blackout, then (2) integrates alpha * v_hat
for 30-75 s. A model can have a low per-sample error and still integrate badly
(correlated bias), or have a bias that alpha removes anyway. This loss trains
for what the pipeline actually does.

For one training sequence  [ calibration (cal_s) | blackout (H) ]:

    alpha = clip( mean(v_gt[cal] where v_gt > 2) / max(0.5, mean(v_hat[cal] same mask)), lo, hi )
            (alpha = 1 when fewer than min_count valid calibration samples)
    D_hat(h) = sum_{t < h} alpha * v_hat(t) * dt
    D_gt(h)  = sum_{t < h} v_gt(t) * dt
    L_int    = mean over sequences and checkpoints h of  |D_hat(h) - D_gt(h)| / max(D_gt(h), d_min)

alpha is NOT detached: the gradient sees exactly the pipeline's correction.
Total training loss:  L = L_existing(per-sample, phase55) + lambda * L_int
"""

from __future__ import annotations

from typing import Sequence

import torch


def emulate_alpha(
    v_hat_cal: torch.Tensor,
    v_gt_cal: torch.Tensor,
    gnss_min_speed: float = 2.0,
    lo: float = 0.85,
    hi: float = 1.25,
    min_count: int = 30,
) -> torch.Tensor:
    """v_hat_cal, v_gt_cal: (B, Tc). Returns alpha: (B,)."""
    mask = (v_gt_cal > gnss_min_speed).to(v_hat_cal.dtype)
    n = mask.sum(dim=1)
    mg = (v_gt_cal * mask).sum(dim=1) / n.clamp(min=1.0)
    ma = (v_hat_cal * mask).sum(dim=1) / n.clamp(min=1.0)
    alpha = torch.clamp(mg / torch.clamp(ma, min=0.5), lo, hi)
    return torch.where(n >= min_count, alpha, torch.ones_like(alpha))


def interval_distance_loss(
    v_hat_bo: torch.Tensor,
    v_gt_bo: torch.Tensor,
    alpha: torch.Tensor,
    dt: float = 0.1,
    checkpoints_s: Sequence[float] = (10.0, 20.0, 30.0, 45.0, 60.0, 75.0),
    d_min_m: float = 20.0,
) -> torch.Tensor:
    """v_hat_bo, v_gt_bo: (B, Tb) blackout part. alpha: (B,). Returns scalar."""
    Tb = v_hat_bo.shape[1]
    idx = sorted({int(round(h / dt)) - 1 for h in checkpoints_s if int(round(h / dt)) <= Tb} | {Tb - 1})
    cum_hat = torch.cumsum(alpha.unsqueeze(1) * v_hat_bo, dim=1) * dt
    cum_gt = torch.cumsum(v_gt_bo, dim=1) * dt
    idx_t = torch.tensor(idx, device=v_hat_bo.device, dtype=torch.long)
    d_hat = cum_hat.index_select(1, idx_t)
    d_gt = cum_gt.index_select(1, idx_t)
    rel = torch.abs(d_hat - d_gt) / torch.clamp(d_gt, min=d_min_m)
    return rel.mean()


def interval_error_final(v_hat_bo, v_gt_bo, alpha, dt: float = 0.1, d_min_m: float = 20.0) -> torch.Tensor:
    """Per-sequence |D_hat - D_gt| / max(D_gt, d_min) at the end of the blackout: (B,)."""
    d_hat = (alpha.unsqueeze(1) * v_hat_bo).sum(dim=1) * dt
    d_gt = v_gt_bo.sum(dim=1) * dt
    return torch.abs(d_hat - d_gt) / torch.clamp(d_gt, min=d_min_m)
