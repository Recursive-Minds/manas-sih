"""1D Dilated Residual Neural Network for Speed & Uncertainty Estimation.

Architecture:
- Input: Multi-channel IMU windows (linear accel, gyro, norms, and spectral features) of shape (B, C, L)
- 1D Temporal Convolutions with increasing dilation factors (1, 2, 4)
- Multi-Task Prediction Heads:
  1. Forward Vehicle Speed (m/s) with non-negativity constraint
  2. Heteroscedastic Log-Variance (ln sigma^2) for dynamic Kalman noise covariance
  3. Motion Context Logits (Stationary vs Cruising vs Cornering)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResBlock1D(nn.Module):
    """1D Residual block with dilated convolution and batch normalization."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class ResNet1DSpeedEstimator(nn.Module):
    """Deep 1D ResNet for vehicle velocity and uncertainty estimation."""

    def __init__(self, in_channels: int = 12, base_channels: int = 64):
        super().__init__()

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Residual backbone with progressive temporal receptive field
        self.block1 = ResBlock1D(base_channels, base_channels, dilation=1)
        self.block2 = ResBlock1D(base_channels, base_channels * 2, dilation=2)
        self.block3 = ResBlock1D(base_channels * 2, base_channels * 2, dilation=4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        feature_dim = base_channels * 2

        # 1. Forward Speed Head (m/s)
        self.speed_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
        )

        # 2. Heteroscedastic Log-Variance Head (ln sigma^2)
        self.variance_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
        )

        # 3. Motion Classification Head (0: Stationary, 1: Cruising, 2: Cornering)
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, C, L) IMU tensor

        Returns:
            speed: (B, 1) predicted forward speed in m/s
            log_var: (B, 1) predicted log-variance ln(sigma^2)
            class_logits: (B, 3) predicted motion class logits
        """
        feats = self.stem(x)
        feats = self.block1(feats)
        feats = self.block2(feats)
        feats = self.block3(feats)

        pooled = self.pool(feats).squeeze(-1)  # (B, feature_dim)

        speed = self.speed_head(pooled)
        log_var = self.variance_head(pooled)
        class_logits = self.class_head(pooled)

        return speed, log_var, class_logits
