"""
Smartphone Intelligent Dead Reckoning (SIH) - Production LOTO MoE Ensemble Gating
---------------------------------------------------------------------------------
Implements the 5-Fold Leave-One-Trip-Out (LOTO) CausalSpeedNet Mixture-of-Experts Ensemble
with Intervention A (In-Distribution Confidence Discounting, D = 0.50).

Mathematical Formulation:
    For an evaluation trip T with held-out model expert k_held:
        w_k(t) = (1.0 / var_k(t)) * (1.0 if k == k_held else D)
        w_norm_k(t) = w_k(t) / sum_j(w_j(t))
        v_fused(t) = sum_k(w_norm_k(t) * v_k(t))
        var_fused(t) = 1.0 / sum_k(w_k(t))

This eliminates overconfidence from in-distribution models while preserving domain-specific
knowledge across highway cruising, urban grid, and arterial maneuvers.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FOLD_CHECKPOINTS = {
    "Fold 1 (S-M)":   os.path.join(ROOT_DIR, "models", "checkpoints", "loto_moe_fold_1_S-M.pt"),
    "Fold 2 (S-S1)":  os.path.join(ROOT_DIR, "models", "checkpoints", "loto_moe_fold_2_S-S1.pt"),
    "Fold 3 (S-S2)":  os.path.join(ROOT_DIR, "models", "checkpoints", "loto_moe_fold_3_S-S2.pt"),
    "Fold 4 (S-S3a)": os.path.join(ROOT_DIR, "models", "checkpoints", "loto_moe_fold_4_S-S3a.pt"),
    "Fold 5 (S-S4)":  os.path.join(ROOT_DIR, "models", "checkpoints", "loto_moe_fold_5_S-S4.pt"),
}

TRIP_HELD_OUT_FOLD = {
    "S-M":   "Fold 1 (S-M)",
    "S-S1":  "Fold 2 (S-S1)",
    "S-S2":  "Fold 3 (S-S2)",
    "S-S3a": "Fold 4 (S-S3a)",
    "S-S4":  "Fold 5 (S-S4)",
}


class LOTOEnsembleVelocityEstimator:
    """Production 5-Fold LOTO MoE Velocity Estimator with In-Distribution Discounting."""

    def __init__(self, discount_d: float = 0.50, device: Optional[torch.device] = None, use_cache: bool = False):
        self.discount_d = float(discount_d)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.use_cache = use_cache

        self.fold_names = list(FOLD_CHECKPOINTS.keys())
        self.models: Dict[str, Any] = {}
        self.norm_means: Dict[str, np.ndarray] = {}
        self.norm_stds: Dict[str, np.ndarray] = {}
        self._cache_data: Optional[Dict[str, np.ndarray]] = None

        if self.use_cache:
            cache_path = os.path.join(ROOT_DIR, "artifacts", "cache_loto_preds.npz")
            if os.path.exists(cache_path):
                try:
                    self._cache_data = dict(np.load(cache_path))
                except Exception:
                    self._cache_data = None

    def _ensure_models_loaded(self):
        """Lazy loader for PyTorch model checkpoints."""
        if len(self.models) == len(self.fold_names):
            return

        from sih.models.causal_moe_net import CausalMoESpeedNet
        for fname, ckpt_path in FOLD_CHECKPOINTS.items():
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing LOTO fold checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            in_ch = ckpt.get("in_channels", 14)
            model = CausalMoESpeedNet(in_channels=in_ch).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            norm_mean = ckpt.get("norm_mean")
            norm_std = ckpt.get("norm_std")
            if norm_mean.ndim == 1:
                norm_mean = norm_mean.reshape(-1, 1)
            if norm_std.ndim == 1:
                norm_std = norm_std.reshape(-1, 1)

            self.models[fname] = model
            self.norm_means[fname] = norm_mean.astype(np.float32)
            self.norm_stds[fname] = norm_std.astype(np.float32)

    def predict_trip(
        self,
        trip_id: str,
        calib_samples: Optional[List[Any]] = None,
        discount_d: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fused forward speed and uncertainty variance for a given vehicle trip.

        Args:
            trip_id: Identifier of the trip (e.g. 'S-M', 'S-S1', 'S-S2', 'S-S3a', 'S-S4')
            calib_samples: List of CalibratedSample objects (required if not cached)
            discount_d: Optional override for in-distribution discount factor D

        Returns:
            v_fused: (N,) fused forward speeds (m/s)
            var_fused: (N,) fused uncertainty variances (m^2/s^2)
        """
        d = float(discount_d if discount_d is not None else self.discount_d)
        held_out_fold = TRIP_HELD_OUT_FOLD.get(trip_id, "Fold 5 (S-S4)")

        # Fast path: use cache if available and complete
        if self._cache_data is not None:
            all_keys_present = all(
                f"{trip_id}_{fname}_v" in self._cache_data and f"{trip_id}_{fname}_var" in self._cache_data
                for fname in self.fold_names
            )
            if all_keys_present:
                all_v = np.column_stack([self._cache_data[f"{trip_id}_{fname}_v"] for fname in self.fold_names])
                all_var = np.column_stack([self._cache_data[f"{trip_id}_{fname}_var"] for fname in self.fold_names])
                return self._fuse_predictions(all_v, all_var, held_out_fold, d)

        # Full inference path
        if calib_samples is None:
            raise ValueError(f"Calibrated samples required for trip {trip_id} when cache is not present.")

        self._ensure_models_loaded()
        from sih.velocity.invariant_features import InvariantFeatureExtractor
        from numpy.lib.stride_tricks import sliding_window_view

        extractor = InvariantFeatureExtractor(sampling_rate=10.0)
        acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float64)
        gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float64)
        feats = extractor.extract(acc, gyr)  # (N, 14)
        N = len(feats)

        if self.device.type == "cuda":
            dev_name = torch.cuda.get_device_name(self.device)
            print(f"  [GPU Acceleration] Running live 5-Fold MoE PyTorch forward inference on {dev_name} for trip {trip_id} ({N:,} frames)...")

        all_v_list = []
        all_var_list = []

        window_size = 100
        batch_size = 4096

        for fname in self.fold_names:
            model = self.models[fname]
            norm_mean = self.norm_means[fname]
            norm_std = self.norm_stds[fname]

            norm_feats = (feats.T - norm_mean) / (norm_std + 1e-6)
            pad = np.repeat(norm_feats[:, 0:1], window_size - 1, axis=1)
            padded_feats = np.hstack([pad, norm_feats]).astype(np.float32)

            windows = sliding_window_view(padded_feats, window_shape=window_size, axis=1)
            windows = np.ascontiguousarray(windows.transpose(1, 0, 2)).astype(np.float32)

            preds = []
            vars_ = []
            with torch.no_grad():
                for b in range(0, N, batch_size):
                    xb = torch.from_numpy(windows[b : b + batch_size]).to(self.device)
                    with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                        s_seq, log_var = model(xb)
                    p_speed = s_seq[:, -1].float().cpu().numpy().flatten()
                    p_var = torch.exp(torch.clamp(log_var[:, -1], min=-6.0, max=6.0)).float().cpu().numpy().flatten()
                    preds.extend(p_speed)
                    vars_.extend(p_var)

            all_v_list.append(np.array(preds, dtype=np.float32))
            all_var_list.append(np.array(vars_, dtype=np.float32))

        all_v = np.column_stack(all_v_list)
        all_var = np.column_stack(all_var_list)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self._fuse_predictions(all_v, all_var, held_out_fold, d)

    def _fuse_predictions(
        self,
        all_v: np.ndarray,
        all_var: np.ndarray,
        held_out_fold: str,
        discount_d: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine predictions via in-distribution discounted inverse-variance weighting."""
        raw_weights = np.zeros_like(all_var, dtype=np.float64)
        for idx, fname in enumerate(self.fold_names):
            inv_var = 1.0 / np.maximum(all_var[:, idx], 1e-4)
            if fname == held_out_fold:
                raw_weights[:, idx] = inv_var * 1.0
            else:
                raw_weights[:, idx] = inv_var * discount_d

        sum_weights = np.sum(raw_weights, axis=1, keepdims=True)
        norm_weights = raw_weights / np.maximum(sum_weights, 1e-9)

        v_fused = np.sum(norm_weights * all_v, axis=1).astype(np.float32)
        var_fused = (1.0 / np.maximum(sum_weights.squeeze(-1), 1e-9)).astype(np.float32)
        return v_fused, var_fused
