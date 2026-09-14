"""
AI Velocity Estimator Stage implementing IVelocityEstimator.

Maintains a sliding window of calibrated IMU readings and performs online
neural network inference to estimate vehicle forward velocity and uncertainty.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import os
import collections
import numpy as np
import torch

from sih.core.contracts import CalibratedSample, VelocityEstimate
from sih.core.interfaces import IVelocityEstimator
from sih.core.pipeline import register_velocity_estimator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.moe_fusion import BayesianMoEFusion


class AIVelocityEstimator(IVelocityEstimator):
    """
    Online AI velocity estimator running dual-expert Bayesian MoE or TCN inference on rolling IMU buffer.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        window_size: int = 60,
        device: Optional[str] = None,
        stationary_accel_std_threshold: float = 0.18,
        **params,
    ) -> None:
        self.window_size = window_size
        self.stationary_thresh = stationary_accel_std_threshold
        
        # Determine device (CUDA RTX 4060 preferred, fallback to CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Priority: explicit path -> best_moe_velocity_model.pt -> best_velocity_model.pt
        if checkpoint_path is None:
            if os.path.exists("models/checkpoints/best_moe_velocity_model.pt"):
                checkpoint_path = "models/checkpoints/best_moe_velocity_model.pt"
            else:
                checkpoint_path = "models/checkpoints/best_velocity_model.pt"

        # Rolling buffer for calibrated samples
        self.buffer = collections.deque(maxlen=max(window_size, 60))

        # Default model setup
        self.in_channels = 12
        self.norm_mean = np.zeros((12, 1), dtype=np.float32)
        self.norm_std = np.ones((12, 1), dtype=np.float32)
        self.is_model_loaded = False
        self.is_moe = False

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            norm_mean = ckpt.get("norm_mean", ckpt.get("mean"))
            norm_std = ckpt.get("norm_std", ckpt.get("std"))
            if norm_mean is not None:
                self.norm_mean = norm_mean.reshape(-1, 1).astype(np.float32)
            if norm_std is not None:
                self.norm_std = norm_std.reshape(-1, 1).astype(np.float32)
            self.in_channels = ckpt.get("in_channels", self.norm_mean.shape[0])

            if "expert_resnet_state_dict" in ckpt and "expert_tcn_state_dict" in ckpt:
                # Bayesian Dual-Expert Mixture-of-Experts Champion
                expert_res = ResNet1DSpeedEstimator(in_channels=self.in_channels, base_channels=64)
                expert_tcn = TCNAttentionVelocityModel(in_channels=self.in_channels, base_channels=32, num_attention_heads=4)
                expert_res.load_state_dict(ckpt["expert_resnet_state_dict"])
                expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
                self.model = BayesianMoEFusion(expert_res, expert_tcn).to(self.device)
                self.is_moe = True
                self.is_model_loaded = True
            elif "model_state_dict" in ckpt:
                # Legacy TCN-Attention baseline
                self.model = TCNAttentionVelocityModel(in_channels=self.in_channels, base_channels=32, num_attention_heads=4)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.to(self.device)
                self.is_moe = False
                self.is_model_loaded = True
        else:
            self.model = TCNAttentionVelocityModel(in_channels=self.in_channels, base_channels=32, num_attention_heads=4).to(self.device)

        self.model.eval()

    def reset(self) -> None:
        self.buffer.clear()

    def estimate(self, sample: CalibratedSample) -> VelocityEstimate:
        """
        Ingest single calibrated IMU sample, update rolling window, and compute forward speed.
        """
        acc = sample.accel_vehicle
        gyro = sample.gyro_vehicle
        norm_a = float(np.linalg.norm(acc))
        norm_w = float(np.linalg.norm(gyro))

        # Build feature vector matching channel configuration
        if self.in_channels == 12:
            # 12-channel: accel (3), gyro (3), norm_a (1), norm_w (1), running variances (4)
            a_mag_diff = norm_a - 9.81
            w_z = gyro[2]
            vec = np.array([
                acc[0], acc[1], acc[2],
                gyro[0], gyro[1], gyro[2],
                norm_a, norm_w,
                a_mag_diff, abs(w_z),
                acc[0] ** 2, gyro[2] ** 2,
            ], dtype=np.float32)
        elif self.in_channels == 8:
            vec = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], norm_a, norm_w], dtype=np.float32)
        else:
            vec = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]], dtype=np.float32)

        self.buffer.append(vec)

        # Minimum window for reliable inference
        req_len = 20 if self.is_moe else self.window_size
        if len(self.buffer) < req_len:
            return VelocityEstimate(
                timestamp_ns=sample.timestamp_ns,
                forward_speed_mps=0.0,
                speed_variance=5.0,
                motion_state="INITIALIZING",
                lateral_speed_mps=0.0,
                vertical_speed_mps=0.0,
            )

        window_arr = np.array(self.buffer, dtype=np.float32).T

        # 1. Stationary Heuristic Check (low vibration variance)
        accel_stds = np.std(window_arr[:3, -min(20, window_arr.shape[1]):], axis=1)
        gyro_stds = np.std(window_arr[3:6, -min(20, window_arr.shape[1]):], axis=1)
        is_stationary = (float(np.mean(accel_stds)) < self.stationary_thresh) and (float(np.mean(gyro_stds)) < 0.03)

        if is_stationary:
            return VelocityEstimate(
                timestamp_ns=sample.timestamp_ns,
                forward_speed_mps=0.0,
                speed_variance=0.01,
                motion_state="STATIONARY",
                lateral_speed_mps=0.0,
                vertical_speed_mps=0.0,
            )

        if not self.is_model_loaded:
            return VelocityEstimate(
                timestamp_ns=sample.timestamp_ns,
                forward_speed_mps=0.0,
                speed_variance=4.0,
                motion_state="DRIVING_UNTRAINED",
            )

        # 2. Forward Inference
        with torch.inference_mode():
            if self.is_moe:
                # Multi-scale dual windows: short (20 samples = 2s) and long (60 samples = 6s)
                L = window_arr.shape[1]
                if L < 60:
                    pad = np.repeat(window_arr[:, 0:1], 60 - L, axis=1)
                    w_full = np.hstack([pad, window_arr])
                else:
                    w_full = window_arr[:, -60:]

                w_norm = (w_full - self.norm_mean) / self.norm_std
                x_long = torch.from_numpy(w_norm).unsqueeze(0).float().to(self.device)
                x_short = x_long[:, :, -20:]

                v_fused, var_fused, _ = self.model(x_short, x_long)
                speed_mps = max(0.0, float(v_fused[0, 0].item()))
                variance = float(var_fused[0, 0].item())
            else:
                norm_w_arr = (window_arr - self.norm_mean) / self.norm_std
                x_tensor = torch.from_numpy(norm_w_arr).unsqueeze(0).float().to(self.device)
                speed_pred, log_var = self.model(x_tensor)
                speed_mps = max(0.0, float(speed_pred[0, 0].item()))
                variance = float(torch.exp(log_var[0, 0]).item())

        return VelocityEstimate(
            timestamp_ns=sample.timestamp_ns,
            forward_speed_mps=speed_mps,
            speed_variance=variance,
            motion_state="DRIVING",
            lateral_speed_mps=0.0,
            vertical_speed_mps=0.0,
        )


# Register in factory registry
register_velocity_estimator(
    "tcn_attention",
    lambda **params: AIVelocityEstimator(**params)
)
register_velocity_estimator(
    "moe_bayesian",
    lambda **params: AIVelocityEstimator(**params)
)
