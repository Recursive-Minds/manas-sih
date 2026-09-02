r"""
TCN-Attention Hybrid Neural Network for 1D Forward Velocity & Uncertainty Estimation.

Architecture:
1. Input: (Batch, in_channels, 100) -> 8-channel IMU window [ax, ay, az, gx, gy, gz, ||a||, ||w||]
2. Multi-Scale Dilated TCN Stages: Dilations d in {1, 2, 4, 8, 16} capturing both micro road vibrations and macro maneuvers.
3. Multi-Head Temporal Self-Attention: Global temporal manifold context.
4. Global Dual Pooling (Mean + Max) & Regression Heads:
   - Head 1: Forward Speed \hat{v} >= 0 (m/s)
   - Head 2: Log-Variance s = log(\sigma^2) for Adaptive EKF Downweighting
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Residual 1D Convolution Block with Dilation and Layer Normalization."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class TemporalSelfAttention(nn.Module):
    """Multi-Head Self-Attention over downsampled temporal feature frames."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, d_model, Length) -> transpose to (Batch, Length, d_model)
        x_t = x.transpose(1, 2)
        attn_out, _ = self.mha(x_t, x_t, x_t)
        out = self.norm(x_t + self.dropout(attn_out))
        return out.transpose(1, 2)  # Return (Batch, d_model, Length)


class TCNAttentionVelocityModel(nn.Module):
    """
    Lightweight, edge-deployable Multi-Scale TCN-Attention Velocity Estimator.
    """
    def __init__(
        self,
        in_channels: int = 8,          # 3 accel + 3 gyro + ||a|| + ||w||
        base_channels: int = 32,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # 1. Front-end 1D Convolutions with temporal downsampling (100 -> 50 -> 25)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )

        # Stage 1: Dilation 1 & 2 (100 -> 50)
        self.stage1_1 = Conv1DBlock(base_channels, base_channels * 2, dilation=1, stride=2)
        self.stage1_2 = Conv1DBlock(base_channels * 2, base_channels * 2, dilation=2, stride=1)

        # Stage 2: Multi-Scale Dilations 4, 8, 16 (50 -> 25)
        self.stage2_1 = Conv1DBlock(base_channels * 2, base_channels * 4, dilation=4, stride=2)
        self.stage2_2 = Conv1DBlock(base_channels * 4, base_channels * 4, dilation=8, stride=1)
        self.stage2_3 = Conv1DBlock(base_channels * 4, base_channels * 4, dilation=16, stride=1)

        feature_dim = base_channels * 4  # 128 channels

        # 2. Temporal Self-Attention over the 25 compressed frames
        self.attention = TemporalSelfAttention(d_model=feature_dim, num_heads=num_attention_heads, dropout=dropout)

        # 3. Dense Classification / Regression Head
        pooled_dim = feature_dim * 2  # Mean + Max pooling concatenation = 256
        self.fc_shared = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: Forward Speed \hat{v} >= 0
        self.speed_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # Head 2: Log-Variance s = log(\sigma^2)
        self.variance_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
          speed: (Batch, 1) in m/s
          log_variance: (Batch, 1)
        """
        # 1. Multi-scale TCN Feature extraction
        feat = self.stem(x)
        feat = self.stage1_2(self.stage1_1(feat))
        feat = self.stage2_3(self.stage2_2(self.stage2_1(feat)))  # Shape: (B, 128, 25)

        # 2. Multi-Head Self-Attention
        attn = self.attention(feat)  # Shape: (B, 128, 25)

        # 3. Global Temporal Pooling (Mean + Max)
        mean_pool = torch.mean(attn, dim=2)
        max_pool, _ = torch.max(attn, dim=2)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 256)

        # 4. Heads
        h = self.fc_shared(pooled)
        speed = self.speed_head(h)
        log_var = self.variance_head(h)

        return speed, log_var


def gaussian_nll_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    log_var: torch.Tensor,
    kinematic_weight: float = 0.05
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian Negative Log-Likelihood Loss with Kinematic Consistency.
    Maintains exact linear gradient scaling for high-speed driving while learning uncertainty.
    """
    diff_sq = (y_true - y_pred) ** 2
    log_var_clamped = torch.clamp(log_var, min=-4.0, max=4.0)
    precision = torch.exp(-log_var_clamped)
    
    loss_nll = 0.5 * precision * diff_sq + 0.5 * log_var_clamped
    
    if len(y_true) > 1 and kinematic_weight > 0:
        d_pred = y_pred[1:] - y_pred[:-1]
        d_true = y_true[1:] - y_true[:-1]
        loss_kinematic = F.mse_loss(d_pred, d_true)
        total_loss = torch.mean(loss_nll) + kinematic_weight * loss_kinematic
    else:
        total_loss = torch.mean(loss_nll)
        
    return total_loss
