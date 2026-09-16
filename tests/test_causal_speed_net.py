"""
Unit tests for CausalSpeedNet and FinalSpeedLoss.

Verifies:
1. Output shape contracts.
2. Strict causality: perturbations at t_k never affect predictions at t < t_k.
3. Non-negativity of speed predictions.
4. Correctness of sequence losses.
"""

import pytest
import torch
import numpy as np

from sih.models.causal_speed_net import CausalSpeedNet, count_parameters
from sih.models.losses_sequence import FinalSpeedLoss, huber_speed, temporal_smoothness, integrated_distance_error


def test_causal_speed_net_shapes():
    net = CausalSpeedNet(in_channels=14, hidden=32, levels=4, kernel_size=3, gru_hidden=32)
    B, C, L = 4, 14, 80
    x = torch.randn(B, C, L)
    speed, logvar = net(x)

    assert speed.shape == (B, L), f"Expected ({B}, {L}), got {speed.shape}"
    assert logvar.shape == (B, L), f"Expected ({B}, {L}), got {logvar.shape}"
    assert (speed >= 0).all(), "Predicted speed must be non-negative"


def test_strict_causality():
    """Perturbing the tail of the sequence must NOT alter predictions for earlier timesteps."""
    net = CausalSpeedNet(in_channels=14, hidden=32, levels=4, kernel_size=3, gru_hidden=32)
    net.eval()

    L = 60
    t_perturb = 40
    x1 = torch.randn(1, 14, L)
    x2 = x1.clone()
    # Perturb all timesteps >= t_perturb
    x2[:, :, t_perturb:] += torch.randn(1, 14, L - t_perturb) * 5.0

    with torch.no_grad():
        s1, l1 = net(x1)
        s2, l2 = net(x2)

    diff_past_speed = torch.max(torch.abs(s1[:, :t_perturb] - s2[:, :t_perturb])).item()
    diff_past_logvar = torch.max(torch.abs(l1[:, :t_perturb] - l2[:, :t_perturb])).item()

    assert diff_past_speed < 1e-6, f"Causality violation! Future perturbation changed past speed by {diff_past_speed}"
    assert diff_past_logvar < 1e-6, f"Causality violation! Future perturbation changed past logvar by {diff_past_logvar}"


def test_losses_sequence():
    loss_fn = FinalSpeedLoss()
    B, L = 4, 50
    pred = torch.ones(B, L) * 10.0
    logvar = torch.zeros(B, L)
    target = torch.ones(B, L) * 10.0

    total_zero, terms_zero = loss_fn(pred, logvar, target)
    assert terms_zero["huber"] == 0.0
    assert terms_zero["distance"] == 0.0

    # Perturb target
    target_offset = torch.ones(B, L) * 12.0
    total_err, terms_err = loss_fn(pred, logvar, target_offset)
    assert total_err > 0.0
    assert terms_err["huber"] > 0.0
    assert terms_err["distance"] > 0.0
