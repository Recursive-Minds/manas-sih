"""
Loss functions for sequence-based speed estimation along the genuine temporal axis.

Critical Architecture Invariance:
Every function here takes (B, L) sequence tensors where L indexes time within a window.
Reductions operate over the time axis L before batch averaging.
This prevents the catastrophic collapse that happens when temporal smoothness/continuity
is misapplied across the shuffled batch dimension.
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "huber_speed",
    "relative_speed_loss",
    "temporal_smoothness",
    "integrated_distance_error",
    "dynamics_variance_alignment",
    "windowed_bias",
    "low_speed_emphasis",
    "quadratic_idle_loss",
    "centripetal_physics_loss",
    "heteroscedastic_nll",
    "scale_ratio_penalty",
    "moe_load_balancing_loss",
    "moe_regime_prior_loss",
    "HomoscedasticMultiTaskLoss",
    "FinalSpeedLoss",
]


def _check(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"Expected (B, L) sequence tensors, got {tuple(pred.shape)} and {tuple(target.shape)}. "
            "Point-wise (B, 1) tensors make temporal terms meaningless."
        )


def huber_speed(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Primary per-sample speed error."""
    _check(pred, target)
    return F.smooth_l1_loss(pred, target, beta=beta)


def relative_speed_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Speed-relative log-cosh loss. Equalizes relative gradients across low and high speeds.
    
    Eliminates high-speed gradient compression where 80 km/h errors received the exact same
    gradient as 5 km/h errors.
    """
    _check(pred, target)
    scale_ref = torch.clamp(0.20 * target + 1.5, min=1.5)
    rel_err = (pred - target) / scale_ref
    return torch.mean(torch.log(torch.cosh(rel_err)))


def temporal_smoothness(pred: torch.Tensor) -> torch.Tensor:
    """Total variation along time: penalises non-physical sample-to-sample chatter."""
    if pred.dim() != 2:
        raise ValueError("temporal_smoothness requires (B, L)")
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    d = pred[:, 1:] - pred[:, :-1]
    return torch.mean(d ** 2)


def integrated_distance_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    """Cumulative along-track trajectory distance error: penalises distance shortfall and overshoot."""
    _check(pred, target)
    dist_pred = torch.cumsum(pred * dt, dim=1)
    dist_target = torch.cumsum(target * dt, dim=1)
    endpoint_err = torch.mean(torch.abs(dist_pred[:, -1] - dist_target[:, -1]))
    profile_err = torch.mean(torch.abs(dist_pred - dist_target))
    return endpoint_err + 0.5 * profile_err


def dynamics_variance_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One-sided penalty for predictions that are less dynamic than the truth."""
    _check(pred, target)
    vp = pred.var(dim=1, unbiased=False)
    vt = target.var(dim=1, unbiased=False)
    deficit = torch.clamp(vt - vp, min=0.0)
    return torch.mean(deficit / (vt + 1e-3))


def windowed_bias(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared mean signed error per window."""
    _check(pred, target)
    return torch.mean(torch.mean(pred - target, dim=1) ** 2)


def low_speed_emphasis(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Extra weight on stops and crawl (<2 m/s). Eliminates false stationary creep without quadratic explosion."""
    _check(pred, target)
    mask = (target < threshold).float()
    denom = mask.sum().clamp(min=1.0)
    err = F.smooth_l1_loss(pred, target, beta=0.5, reduction="none")
    return torch.sum(mask * err) / denom


def heteroscedastic_nll(
    pred: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    detach_mean: bool = True,
) -> torch.Tensor:
    """Gaussian negative log-likelihood for calibrating the uncertainty head."""
    _check(pred, target)
    err = (pred.detach() if detach_mean else pred) - target
    inv_var = torch.exp(-log_var)
    return torch.mean(0.5 * (inv_var * err ** 2 + log_var))


def scale_ratio_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic penalty on window distance ratio sum(pred) / sum(target).
    
    Prevents the velocity network from compressing speed scale (under-predicting distance)
    or blowing up scale at high velocities.
    """
    _check(pred, target)
    p_sum = torch.sum(pred, dim=1).clamp(min=1e-3)
    t_sum = torch.sum(target, dim=1).clamp(min=1e-3)
    return torch.mean((torch.log(p_sum + 1.0) - torch.log(t_sum + 1.0)) ** 2)


def quadratic_idle_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Asymmetric quadratic penalty for positive speed predictions during complete stops (target < threshold).
    
    Eliminates engine idle vibration creep (which previously accounted for 25% of benchmark failures).
    """
    _check(pred, target)
    stop_mask = (target < threshold).float()
    denom = stop_mask.sum().clamp(min=1.0)
    idle_err = (stop_mask * (F.relu(pred) ** 2)).sum() / denom
    return idle_err


def centripetal_physics_loss(
    pred: torch.Tensor,
    a_lat: Optional[torch.Tensor] = None,
    omega_z: Optional[torch.Tensor] = None,
    min_turn_rate: float = 0.05,
) -> torch.Tensor:
    """First-principles Newtonian centripetal bound: v <= |a_lat| / |omega_z|.
    
    Penalizes speed predictions that exceed the physical centripetal capacity in curves.
    """
    if a_lat is None or omega_z is None:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    _check(pred, a_lat)
    turn_mask = (torch.abs(omega_z) >= min_turn_rate).float()
    denom = turn_mask.sum().clamp(min=1.0)
    v_max_cent = torch.abs(a_lat) / (torch.abs(omega_z) + 1e-4)
    overshoot = F.relu(pred - v_max_cent)
    return (turn_mask * (overshoot ** 2)).sum() / denom


def moe_load_balancing_loss(gates: torch.Tensor) -> torch.Tensor:
    """Load balancing loss preventing MoE expert collapse.
    
    Args:
        gates: (B, L, K) or (N, K) softmax gating weights.
    """
    if gates.dim() == 3:
        gates = gates.reshape(-1, gates.shape[-1])
    num_experts = gates.shape[-1]
    P_k = torch.mean(gates, dim=0)
    idx_top = torch.argmax(gates, dim=1)
    f_k = torch.zeros_like(P_k)
    for k in range(num_experts):
        f_k[k] = torch.mean((idx_top == k).float())
    return num_experts * torch.sum(f_k * P_k)


def moe_regime_prior_loss(gates: torch.Tensor, target_speed: torch.Tensor) -> torch.Tensor:
    """Soft regime guidance loss during warmup training.
    
    Expert 0: 0-20 km/h (target < 5.5 m/s)
    Expert 1: 20-60 km/h (5.5 <= target <= 16.6 m/s)
    Expert 2: >60 km/h (target > 16.6 m/s)
    """
    if gates.dim() == 3:
        t_flat = target_speed.reshape(-1)
        g_flat = gates.reshape(-1, gates.shape[-1])
    else:
        t_flat = target_speed
        g_flat = gates
    
    priors = torch.zeros_like(g_flat)
    priors[t_flat < 5.5, 0] = 1.0
    priors[(t_flat >= 5.5) & (t_flat <= 16.6), 1] = 1.0
    priors[t_flat > 16.6, 2] = 1.0
    
    log_gates = torch.log(torch.clamp(g_flat, min=1e-6))
    return -torch.mean(torch.sum(priors * log_gates, dim=-1))


class FinalSpeedLoss(nn.Module):
    """
    Balanced multi-objective loss formulated strictly along the temporal sequence axis.
    """
    def __init__(
        self,
        w_huber: float = 0.50,
        w_rel: float = 0.50,
        w_distance: float = 0.15,
        w_dynamics: float = 0.15,
        w_bias: float = 0.10,
        w_low_speed: float = 0.10,
        w_smooth: float = 0.02,
        w_nll: float = 0.05,
        w_scale: float = 0.25,
        w_idle: float = 0.25,
        w_centripetal: float = 0.10,
        dt: float = 0.1,
        low_speed_threshold: float = 2.0,
        idle_threshold: float = 0.5,
    ):
        super().__init__()
        self.w_huber = w_huber
        self.w_rel = w_rel
        self.w_distance = w_distance
        self.w_dynamics = w_dynamics
        self.w_bias = w_bias
        self.w_low_speed = w_low_speed
        self.w_smooth = w_smooth
        self.w_nll = w_nll
        self.w_scale = w_scale
        self.w_idle = w_idle
        self.w_centripetal = w_centripetal
        self.dt = dt
        self.low_speed_threshold = low_speed_threshold
        self.idle_threshold = idle_threshold

    def forward(
        self,
        pred: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
        a_lat: Optional[torch.Tensor] = None,
        omega_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        terms = {
            "huber": self.w_huber * huber_speed(pred, target),
            "rel": self.w_rel * relative_speed_loss(pred, target),
            "distance": self.w_distance * integrated_distance_error(pred, target, self.dt),
            "dynamics": self.w_dynamics * dynamics_variance_alignment(pred, target),
            "bias": self.w_bias * windowed_bias(pred, target),
            "low_speed": self.w_low_speed * low_speed_emphasis(pred, target, self.low_speed_threshold),
            "smooth": self.w_smooth * temporal_smoothness(pred),
            "nll": self.w_nll * heteroscedastic_nll(pred, log_var, target),
            "scale": self.w_scale * scale_ratio_penalty(pred, target),
            "idle": self.w_idle * quadratic_idle_loss(pred, target, self.idle_threshold),
            "centripetal": self.w_centripetal * centripetal_physics_loss(pred, a_lat, omega_z),
        }
        total = sum(terms.values())
        return total, {k: float(v.detach().item()) for k, v in terms.items()}


class HomoscedasticMultiTaskLoss(nn.Module):
    """
    Kendall, Gal & Cipolla (CVPR 2018) Homoscedastic Multi-Task Uncertainty Loss.
    
    Reference:
        Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using
        Uncertainty to Weigh Losses for Scene Geometry and Semantics."
        IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
        
    Mathematical principle:
        For regression tasks with Gaussian observation noise:
        L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
        where s_i = log(sigma_i^2) is an unconstrained learnable parameter (nn.Parameter)
        optimized automatically by gradient descent alongside network weights.
        
        Eliminates manual trial-and-error loss weighting.
    """
    def __init__(self, task_names: list[str], clamp_range: tuple[float, float] = (-4.0, 4.0)):
        super().__init__()
        self.task_names = list(task_names)
        self.clamp_range = clamp_range
        # Initialize s_i = 0.0 (initial effective weight = 0.5)
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_names), dtype=torch.float32))

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        s_clamped = torch.clamp(self.log_vars, self.clamp_range[0], self.clamp_range[1])
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        effective_weights: dict[str, float] = {}

        for i, name in enumerate(self.task_names):
            loss_i = losses[name]
            s_i = s_clamped[i]
            precision = torch.exp(-s_i)
            # Kendall & Gal 2018 formulation
            term = 0.5 * precision * loss_i + 0.5 * s_i
            total_loss = total_loss + term
            effective_weights[name] = float((0.5 * precision).detach().item())

        return total_loss, effective_weights

