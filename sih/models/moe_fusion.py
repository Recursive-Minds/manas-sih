"""Bayesian Mixture-of-Experts (MoE) Fusion for Vehicle Forward Velocity & Uncertainty.

Implements Minimum-Variance Uncertainty-Weighted Ensembling:
Blends predictions from:
1. Expert 1: 1D Dilated Residual Network (ResNet-1D) - Micro-vibrations and rapid dynamic changes
2. Expert 2: Multi-Scale Dilated TCN with Multi-Head Temporal Self-Attention - 10-second macro context

Mathematically optimal closed-form minimum-variance estimator:
    v_fused = (v_1 / var_1 + v_2 / var_2) / (1 / var_1 + 1 / var_2)
    var_fused = 1 / (1 / var_1 + 1 / var_2)
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sih.models.resnet1d import ResNet1DSpeedEstimator


class BayesianMoEFusion(nn.Module):
    """Minimum-Variance Bayesian Mixture-of-Experts Fusion Head."""

    def __init__(
        self,
        expert_resnet: Optional[nn.Module] = None,
        expert_tcn: Optional[nn.Module] = None,
        min_variance: float = 1e-4,
        max_variance: float = 16.0,
    ):
        super().__init__()
        self.expert_resnet = expert_resnet
        self.expert_tcn = expert_tcn
        self.min_variance = min_variance
        self.max_variance = max_variance

    def fuse_predictions(
        self,
        v_resnet: torch.Tensor,
        log_var_resnet: torch.Tensor,
        v_tcn: torch.Tensor,
        log_var_tcn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine two speed and uncertainty estimates via precision-weighted Bayesian fusion.

        Args:
            v_resnet: (B, 1) predicted forward speed from ResNet-1D (m/s)
            log_var_resnet: (B, 1) predicted log-variance ln(sigma^2)
            v_tcn: (B, 1) predicted forward speed from TCN-Attention (m/s)
            log_var_tcn: (B, 1) predicted log-variance ln(sigma^2)

        Returns:
            v_fused: (B, 1) optimal minimum-variance forward speed
            var_fused: (B, 1) combined uncertainty variance sigma^2
        """
        # Clamp log-variances BEFORE exponentiation to prevent FP16 overflow (max FP16 is 65504, exp(11) ~ 59874)
        c_log_var_res = torch.clamp(log_var_resnet, min=-6.0, max=4.0)
        c_log_var_tcn = torch.clamp(log_var_tcn, min=-6.0, max=4.0)

        var_resnet = torch.clamp(torch.exp(c_log_var_res), min=self.min_variance, max=self.max_variance)
        var_tcn = torch.clamp(torch.exp(c_log_var_tcn), min=self.min_variance, max=self.max_variance)

        prec_resnet = 1.0 / var_resnet
        prec_tcn = 1.0 / var_tcn
        total_prec = prec_resnet + prec_tcn

        # Minimum-variance weighted velocity
        v_fused = (v_resnet * prec_resnet + v_tcn * prec_tcn) / total_prec
        var_fused = 1.0 / total_prec

        return v_fused, var_fused

    def forward(
        self,
        x_short: torch.Tensor,
        x_long: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through both experts and Bayesian fusion.

        Args:
            x_short: (B, C, L_short) short temporal window (e.g. 20 samples = 2s) for ResNet
            x_long: (B, C, L_long) long temporal window (e.g. 60 or 100 samples) for TCN

        Returns:
            v_fused: (B, 1) fused velocity
            var_fused: (B, 1) fused variance
            diagnostics: dictionary of intermediate outputs
        """
        assert self.expert_resnet is not None, "ResNet expert must be assigned"
        assert self.expert_tcn is not None, "TCN expert must be assigned"

        v_res, log_var_res, cls_logits = self.expert_resnet(x_short)
        v_tcn, log_var_tcn = self.expert_tcn(x_long)

        v_fused, var_fused = self.fuse_predictions(v_res, log_var_res, v_tcn, log_var_tcn)

        diag = {
            "v_resnet": v_res,
            "var_resnet": torch.exp(torch.clamp(log_var_res, min=-6.0, max=4.0)),
            "v_tcn": v_tcn,
            "var_tcn": torch.exp(torch.clamp(log_var_tcn, min=-6.0, max=4.0)),
            "class_logits": cls_logits,
        }

        return v_fused, var_fused, diag


def numpy_bayesian_fusion(
    v_1: float,
    sigma_1: float,
    v_2: float,
    sigma_2: float,
) -> Tuple[float, float]:
    """Lightweight pure-NumPy Bayesian fusion function for real-time edge execution."""
    var_1 = max(float(sigma_1) ** 2, 1e-4)
    var_2 = max(float(sigma_2) ** 2, 1e-4)
    prec_1 = 1.0 / var_1
    prec_2 = 1.0 / var_2
    total_prec = prec_1 + prec_2

    v_fused = (v_1 * prec_1 + v_2 * prec_2) / total_prec
    sigma_fused = float(np.sqrt(1.0 / total_prec))
    return float(v_fused), sigma_fused
