"""Physics-Informed Loss Functions for Vehicle Speed & Dynamics Estimation.

Implements:
1. balanced_velocity_loss: Rule 8 scale-balanced loss preventing high-speed underprediction.
2. l_dynamics_variance_alignment: Asymmetric variance deficit penalty preventing flat predictions.
3. l_centripetal: Centripetal physics constraint enforcing a_lat = v_forward * omega_yaw.
4. l_drift_windowed: Windowed positive signed bias penalty preventing integration dead-reckoning drift.
5. l_jerk_hinge: Central-difference vehicle jerk constraint above physical threshold.
6. phase55_balanced_loss: Multi-objective balanced loss for Dual-Expert MoE.
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from typing import Dict, Optional


def l_dynamics_variance_alignment(v_pred: torch.Tensor, v_true: torch.Tensor) -> torch.Tensor:
    """One-sided variance alignment loss: penalises under-dynamic (flat) speed predictions.

    Fires when Var(v_pred) < Var(v_true). Does NOT penalise when prediction is more variable
    than ground truth.
    """
    vp = v_pred.view(-1)
    vt = v_true.view(-1)
    if len(vp) < 2:
        return torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)
    var_p = torch.var(vp) + 1e-4
    var_t = torch.var(vt) + 1e-4
    deficit = torch.clamp(var_t - var_p, min=0.0)
    norm_denom = torch.clamp(var_t, min=1.0)
    return torch.clamp(deficit / norm_denom, max=10.0)


def l_centripetal(
    v_pred: torch.Tensor,
    a_lateral: torch.Tensor,
    omega_yaw: torch.Tensor,
) -> torch.Tensor:
    """Kinematic Centripetal Consistency Loss: a_lateral = v_forward * omega_yaw."""
    p = v_pred.view(-1)
    a_lat = a_lateral.view(-1)
    w_yaw = omega_yaw.view(-1)
    if len(p) == 0:
        return torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)
    centripetal_err = a_lat - p * w_yaw
    return torch.mean(centripetal_err ** 2)


def l_drift_windowed(
    v_pred: torch.Tensor,
    v_true: torch.Tensor,
    window_size: int = 100,
    low_speed_threshold: float = 5.0,
    weight_multiplier: float = 3.0,
) -> torch.Tensor:
    """Windowed positive signed-bias loss penalizing forward dead-reckoning drift."""
    p = v_pred.view(-1)
    t = v_true.view(-1)
    N = len(p)

    if N == 0:
        return torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

    if N < window_size:
        bias = torch.mean(p - t)
        pos_bias = torch.clamp(bias, min=0.0)
        loss = pos_bias ** 2
        if torch.mean(t) < low_speed_threshold:
            loss = loss * weight_multiplier
        return loss

    num_windows = N // window_size
    p_windows = p[: num_windows * window_size].view(num_windows, window_size)
    t_windows = t[: num_windows * window_size].view(num_windows, window_size)

    win_bias = torch.mean(p_windows - t_windows, dim=1)
    pos_bias = torch.clamp(win_bias, min=0.0)
    win_losses = pos_bias ** 2

    win_t_mean = torch.mean(t_windows, dim=1)
    weights = torch.where(win_t_mean < low_speed_threshold, weight_multiplier, 1.0)
    total_loss = torch.mean(win_losses * weights)

    return total_loss


def l_jerk_hinge(
    v_pred: torch.Tensor,
    dt: float = 0.1,
    j_max: float = 20.0,
) -> torch.Tensor:
    """Central-difference jerk violation penalty."""
    p = v_pred.view(-1)
    if len(p) < 3:
        return torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)
    jerk = (p[2:] - p[:-2]) / (2.0 * dt)
    excess_jerk = torch.clamp(torch.abs(jerk) - j_max, min=0.0)
    return torch.mean(excess_jerk ** 2)


def balanced_velocity_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    log_var: torch.Tensor,
    w_scale: float = 2.0,
    w_dyn: float = 1.0,
) -> torch.Tensor:
    """High-Speed Balanced Velocity Loss enforcing Rule 8 in GEMINI.md.

    Avoids gradient compression on high speeds and forces speed scale ratio sum(v_pred)/sum(v_true) -> 1.00.
    """
    # 1. Primary Huber / SmoothL1 loss across all speed regimes
    mse = F.smooth_l1_loss(y_pred, y_true, beta=1.0)

    # 2. Global speed scale ratio penalty
    sum_true = torch.sum(y_true) + 1e-4
    sum_pred = torch.sum(y_pred)
    scale_penalty = (sum_pred / sum_true - 1.0) ** 2

    # 3. High-speed under-prediction penalty (for v > 8 m/s)
    high_speed_mask = (y_true > 8.0).float()
    if high_speed_mask.sum() > 0:
        high_speed_err = torch.sum(high_speed_mask * (y_pred - y_true) ** 2) / high_speed_mask.sum()
    else:
        high_speed_err = torch.tensor(0.0, device=y_true.device)

    # 4. Asymmetric variance alignment loss (penalizes flat predictions)
    dyn_loss = l_dynamics_variance_alignment(y_pred, y_true)

    # 5. Independent uncertainty learning (detached speed_head gradient)
    var_target = (y_pred.detach() - y_true) ** 2
    var_pred = torch.exp(torch.clamp(log_var, -4.0, 4.0))
    var_loss = F.smooth_l1_loss(var_pred, var_target, beta=1.0)

    return mse + w_scale * scale_penalty + 0.5 * high_speed_err + w_dyn * dyn_loss + 0.1 * var_loss


def phase55_balanced_loss(
    v_fused: torch.Tensor,
    var_fused: torch.Tensor,
    v_res: torch.Tensor,
    var_res: torch.Tensor,
    v_tcn: torch.Tensor,
    var_tcn: torch.Tensor,
    v_gt: torch.Tensor,
    a_lat: Optional[torch.Tensor] = None,
    w_yaw: Optional[torch.Tensor] = None,
    class_logits: Optional[torch.Tensor] = None,
    motion_labels: Optional[torch.Tensor] = None,
    w_dyn: float = 2.0,
    w_cls: float = 0.2,
) -> torch.Tensor:
    """Multi-objective balanced loss for Dual-Expert MoE Fusion."""
    loss_fused = F.smooth_l1_loss(v_fused, v_gt, beta=1.0)
    loss_res = F.smooth_l1_loss(v_res, v_gt, beta=1.0)
    loss_tcn = F.smooth_l1_loss(v_tcn, v_gt, beta=1.0)
    loss_primary = loss_fused + 0.5 * loss_res + 0.5 * loss_tcn

    # Scale penalty with safe clamped denominator
    mean_true = torch.clamp(torch.mean(v_gt), min=0.5)
    mean_pred = torch.clamp(torch.mean(v_fused), min=0.0)
    loss_scale = torch.clamp((mean_pred / mean_true - 1.0) ** 2, max=25.0)

    # High speed penalty
    high_mask = (v_gt > 8.0).float()
    if high_mask.sum() > 0:
        loss_high = torch.clamp(torch.sum(high_mask * (v_fused - v_gt) ** 2) / high_mask.sum(), max=50.0)
    else:
        loss_high = torch.tensor(0.0, device=v_gt.device)

    # Variance target learning
    var_target_fused = (v_fused.detach() - v_gt) ** 2
    var_target_res = (v_res.detach() - v_gt) ** 2
    var_target_tcn = (v_tcn.detach() - v_gt) ** 2
    loss_var = (
        F.smooth_l1_loss(var_fused, var_target_fused, beta=1.0)
        + 0.5 * F.smooth_l1_loss(var_res, var_target_res, beta=1.0)
        + 0.5 * F.smooth_l1_loss(var_tcn, var_target_tcn, beta=1.0)
    )

    # Dynamics variance alignment
    loss_dyn = l_dynamics_variance_alignment(v_fused, v_gt)

    loss_total = loss_primary + 2.0 * loss_scale + 0.5 * loss_high + 0.10 * loss_var + w_dyn * loss_dyn

    if a_lat is not None and w_yaw is not None:
        loss_cent = l_centripetal(v_fused, a_lat, w_yaw)
        loss_total += 0.05 * loss_cent

    if class_logits is not None and motion_labels is not None:
        loss_cls = F.cross_entropy(class_logits, motion_labels)
        loss_total += w_cls * loss_cls

    return loss_total
