"""
Causal temporal network mapping invariant IMU features to forward speed and heteroscedastic uncertainty.

Key Guarantees:
1. Strictly Causal: All 1D convolutions are left-padded so timestep t depends strictly on <= t.
2. ChannelNorm: Normalizes only across channels per timestep, preventing future temporal leakage.
3. Sequence output: Emits (speed, log_var) for every timestep along the sequence window.
4. Heteroscedastic Uncertainty: Output log_var directly calibrates the Kalman filter measurement covariance.
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """1D convolution that never reads the future (left-padded)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel_size, padding=0, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(F.pad(x, (self.left_pad, 0)))


class ChannelNorm(nn.Module):
    """Normalise across channels at each timestep.

    nn.BatchNorm / GroupNorm aggregate across time which leaks future samples into present.
    ChannelNorm keeps every timestep strictly independent.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CausalBlock(nn.Module):
    """Pre-norm residual block of two dilated causal convolutions."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.norm1 = ChannelNorm(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm2 = ChannelNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return h + res


class CausalSpeedNet(nn.Module):
    """Dilated causal TCN + unidirectional GRU producing per-timestep speed and log-variance."""

    def __init__(
        self,
        in_channels: int = 14,
        hidden: int = 64,
        levels: int = 5,
        kernel_size: int = 3,
        gru_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden

        self.input_proj = nn.Conv1d(in_channels, hidden, 1)
        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            blocks.append(
                CausalBlock(hidden, hidden, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
        self.tcn = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,  # Strictly unidirectional to enforce causality
        )

        self.head_speed = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, L) where C = in_channels, L = sequence length in timesteps.
        Returns:
            speed: (B, L) non-negative vehicle forward speed in m/s.
            log_var: (B, L) log-variance of speed estimate.
        """
        h = self.input_proj(x)
        h = self.tcn(h)                     # (B, hidden, L)
        h_seq = h.transpose(1, 2)           # (B, L, hidden)
        out_gru, _ = self.gru(h_seq)        # (B, L, gru_hidden)

        speed = F.softplus(self.head_speed(out_gru).squeeze(-1))  # (B, L), strictly >= 0
        log_var = torch.clamp(self.head_logvar(out_gru).squeeze(-1), min=-6.0, max=4.0)
        return speed, log_var


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
