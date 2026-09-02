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


class AIVelocityEstimator(IVelocityEstimator):
    """
    Online AI velocity estimator running TCN-Attention inference on rolling IMU buffer.
    """
    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_velocity_model.pt",
        window_size: int = 100,
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

        # Rolling buffer for raw/calibrated samples
        self.buffer = collections.deque(maxlen=window_size)

        # Default model setup
        self.in_channels = 8
        self.norm_mean = np.zeros((8, 1), dtype=np.float32)
        self.norm_std = np.ones((8, 1), dtype=np.float32)
        self.is_model_loaded = False

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.norm_mean = ckpt.get("norm_mean", self.norm_mean)
            self.norm_std = ckpt.get("norm_std", self.norm_std)
            self.in_channels = self.norm_mean.shape[0]

            self.model = TCNAttentionVelocityModel(in_channels=self.in_channels, base_channels=32, num_attention_heads=4)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.is_model_loaded = True
        else:
            self.model = TCNAttentionVelocityModel(in_channels=self.in_channels, base_channels=32, num_attention_heads=4)

        self.model.to(self.device)
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

        if self.in_channels == 8:
            vec = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], norm_a, norm_w], dtype=np.float32)
        else:
            vec = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]], dtype=np.float32)

        self.buffer.append(vec)

        # If buffer not full, output zero with high variance
        if len(self.buffer) < self.window_size:
            return VelocityEstimate(
                timestamp_ns=sample.timestamp_ns,
                forward_speed_mps=0.0,
                speed_variance=5.0,
                motion_state="INITIALIZING",
                lateral_speed_mps=0.0,
                vertical_speed_mps=0.0,
            )

        # Extract window: shape (window_size, in_channels) -> transpose to (in_channels, window_size)
        window_arr = np.array(self.buffer, dtype=np.float32).T

        # 1. Check stationary heuristic (low vibration variance)
        accel_stds = np.std(window_arr[:3, -20:], axis=1)  # Last 2s
        gyro_stds = np.std(window_arr[3:6, -20:], axis=1)
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

        # 2. Neural network forward inference
        if not self.is_model_loaded:
            return VelocityEstimate(
                timestamp_ns=sample.timestamp_ns,
                forward_speed_mps=0.0,
                speed_variance=4.0,
                motion_state="DRIVING_UNTRAINED",
            )

        norm_w_arr = (window_arr - self.norm_mean) / self.norm_std
        x_tensor = torch.from_numpy(norm_w_arr).unsqueeze(0).float().to(self.device)

        with torch.inference_mode():
            speed_pred, log_var = self.model(x_tensor)
            speed_mps = float(speed_pred[0, 0].item())
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
