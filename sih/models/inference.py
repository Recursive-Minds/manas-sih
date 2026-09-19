"""
AI Velocity Model Loading and Sequence Inference Pipeline.

Decouples deep learning model checkpoint restoration, feature extraction,
and batch tensor inference from benchmark harnesses.
"""

from __future__ import annotations
import os
import torch
import numpy as np
from typing import Optional, Tuple, Any, List

from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.moe_fusion import BayesianMoEFusion
from sih.fusion.speed_smoother import CausalSpeedSmoother


def load_ai_model(
    device: torch.device,
    model_path: Optional[str] = None,
    root_dir: Optional[str] = None,
) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray], str]:
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    default_moe_path = os.path.join(root_dir, "models", "checkpoints", "best_moe_velocity_model.pt")
    default_tcn_path = os.path.join(root_dir, "models", "checkpoints", "best_velocity_model.pt")

    # Explicit research override for LOTO ensemble
    if model_path == "loto_ensemble":
        from sih.models.ensemble_gating import LOTOEnsembleVelocityEstimator
        print("[AI Model] [RESEARCH ONLY] Loading 5-Fold LOTO MoE Ensemble with Discount (D=0.50)")
        estimator = LOTOEnsembleVelocityEstimator(discount_d=0.50, device=device)
        return estimator, None, None, "loto_ensemble"

    target_moe_path = model_path if model_path and os.path.exists(model_path) else default_moe_path

    if os.path.exists(target_moe_path):
        ckpt = torch.load(target_moe_path, map_location=device, weights_only=False)
        mtype = ckpt.get("model_type", "moe")
        if mtype == "causal_speed_net":
            print(f"[AI Model] Loading CausalSpeedNet Champion Checkpoint: {target_moe_path}")
            from sih.models.causal_speed_net import CausalSpeedNet
            in_ch = ckpt.get("in_channels", 14)
            model = CausalSpeedNet(in_channels=in_ch, hidden=64, levels=5, kernel_size=3, gru_hidden=64).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            norm_mean = ckpt.get("norm_mean")
            norm_std = ckpt.get("norm_std")
            if norm_mean.ndim == 1: norm_mean = norm_mean.reshape(-1, 1)
            if norm_std.ndim == 1: norm_std = norm_std.reshape(-1, 1)
            return model, norm_mean.astype(np.float32), norm_std.astype(np.float32), "causal_speed_net"

        print(f"[AI Model] Loading Unified MoE Checkpoint: {target_moe_path}")
        in_channels = ckpt.get("in_channels", 12)
        expert_res = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
        expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
        expert_res.load_state_dict(ckpt["expert_resnet_state_dict"])
        expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
        model = BayesianMoEFusion(expert_res, expert_tcn).to(device)
        model.eval()

        norm_mean = ckpt.get("norm_mean", ckpt.get("mean"))
        norm_std = ckpt.get("norm_std", ckpt.get("std"))
        if norm_mean.ndim == 1 or norm_mean.shape[0] == 1:
            norm_mean = norm_mean.reshape(-1, 1)
        if norm_std.ndim == 1 or norm_std.shape[0] == 1:
            norm_std = norm_std.reshape(-1, 1)
        return model, norm_mean.astype(np.float32), norm_std.astype(np.float32), "moe"

    print(f"[AI Model] Loading Baseline Checkpoint: {default_tcn_path}")
    ckpt = torch.load(default_tcn_path, map_location=device, weights_only=False)
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std = ckpt.get("norm_std", np.ones((8, 1), dtype=np.float32))
    return model, norm_mean.astype(np.float32), norm_std.astype(np.float32), "tcn"


def predict_velocities(
    model: Any,
    calib_samples: List[Any],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    model_type: str = "moe",
    trip_id: Optional[str] = None,
    root_dir: Optional[str] = None,
    apply_smoothing: bool = True,
) -> np.ndarray:
    """
    Performs streaming sliding-window forward inference across calibrated IMU samples.
    Optionally applies causal kinematic slew-rate and low-pass filter smoothing.
    Feature extraction does NOT branch on trip name.
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if model_type == "loto_ensemble":
        v_fused, _ = model.predict_trip(trip_id=trip_id if trip_id else "S-M", calib_samples=calib_samples)
        return v_fused.astype(np.float32)

    if model_type in ("causal_speed_net", "causal_moe_net"):
        from sih.velocity.invariant_features import InvariantFeatureExtractor

        extractor = InvariantFeatureExtractor(sampling_rate=10.0)
        acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float64)
        gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float64)
        feats = extractor.extract(acc, gyr)  # (N, 14)
        norm_feats = (feats.T - norm_mean) / (norm_std + 1e-6)  # (14, N)

        window_size = 100
        N = len(feats)
        pad = np.repeat(norm_feats[:, 0:1], window_size - 1, axis=1)
        padded_feats = np.hstack([pad, norm_feats]).astype(np.float32)

        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded_feats, window_shape=window_size, axis=1)
        windows = np.ascontiguousarray(windows.transpose(1, 0, 2)).astype(np.float32)  # (N, 14, 100)

        preds = []
        batch_size = 2048
        with torch.no_grad():
            for b in range(0, N, batch_size):
                xb = torch.from_numpy(windows[b : b + batch_size]).to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    s_seq, _ = model(xb)
                preds.extend(s_seq[:, -1].float().cpu().numpy().flatten())
        return np.array(preds, dtype=np.float32)

    if model_type == "moe":
        from sih.features.streaming import StreamingFeatureExtractor

        extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
        feats = extractor.batch_extract(calib_samples)

        N = len(feats)
        norm_feats = (feats.T - norm_mean) / (norm_std + 1e-6)
        short_len, long_len = 20, 60
        pad_l = np.repeat(norm_feats[:, 0:1], long_len - 1, axis=1)
        padded_feats = np.hstack([pad_l, norm_feats]).astype(np.float32)

        from numpy.lib.stride_tricks import sliding_window_view
        windows_l = sliding_window_view(padded_feats, window_shape=long_len, axis=1)
        windows_l = np.ascontiguousarray(windows_l.transpose(1, 0, 2)).astype(np.float32)
        windows_s = np.ascontiguousarray(windows_l[:, :, -short_len:]).astype(np.float32)

        preds = []
        batch_size = 4096
        with torch.no_grad():
            for b in range(0, N, batch_size):
                b_s = torch.from_numpy(windows_s[b : b + batch_size]).to(device)
                b_l = torch.from_numpy(windows_l[b : b + batch_size]).to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    vf, _, _ = model(b_s, b_l)
                preds.extend(vf.squeeze(-1).float().cpu().numpy().flatten())
        raw_preds = np.array(preds, dtype=np.float32)

    else:
        acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
        gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
        feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])

        N = len(feats)
        window_size = 100
        windows = []
        for i in range(N):
            if i < window_size:
                pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
                w = np.vstack([pad, feats[:i + 1]]).T
            else:
                w = feats[i - window_size + 1 : i + 1].T
            windows.append((w - norm_mean) / norm_std)

        preds = []
        with torch.no_grad():
            for b in range(0, N, 2048):
                x = torch.from_numpy(np.array(windows[b : b + 2048], dtype=np.float32)).to(device)
                p, _ = model(x)
                preds.extend(p.cpu().numpy().flatten())
        raw_preds = np.array(preds, dtype=np.float32)

    if apply_smoothing:
        smoother = CausalSpeedSmoother(a_max_mps2=3.5, a_min_mps2=-5.0, tau_s=0.25)
        return smoother.filter_sequence(raw_preds, dt_s=0.1)

    return raw_preds
