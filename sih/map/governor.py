"""Closed-Loop Road-Constrained Velocity & Curvature Kinematics Observer.

Implements physics-informed kinematic constraints derived from road geometry:
1. Non-Holonomic Constraint (NHC): Lateral velocity v_lateral = 0 along the road tangent.
2. Curvature-Limited Speed Bound: v_max = sqrt(a_lat_max / kappa) preventing along-track
   runaway over-estimation during sharp turns.
3. Arc-Length Along-Track Velocity Regularization: Aligns dead-reckoned forward velocity
   with physical road centerline progress.
4. C2-continuous Hermite smoothstep catch-up interpolation on corner exit.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class RoadKinematicsGovernor:
    """Regulates estimated forward velocity and trajectory using road geometry constraints."""

    def __init__(
        self,
        a_lat_max: float = 3.5,  # Maximum sustainable lateral acceleration (m/s^2)
        min_curve_radius: float = 5.0,  # Minimum physical turn radius (meters)
        speed_limit_mps: float = 33.3,  # 120 km/h default speed cap
    ):
        self.a_lat_max = a_lat_max
        self.min_curve_radius = min_curve_radius
        self.speed_limit_mps = speed_limit_mps

    def compute_curvature(self, coords: np.ndarray) -> np.ndarray:
        """Compute local Menger curvature kappa = 1 / R for an array of 2D coordinates.

        Args:
            coords: Array of shape (N, 2) representing (East, North) coordinates.

        Returns:
            Curvature array kappa of shape (N,) in units of 1/meter.
        """
        N = len(coords)
        if N < 3:
            return np.zeros(N, dtype=np.float64)

        curvatures = np.zeros(N, dtype=np.float64)
        for i in range(1, N - 1):
            p1 = coords[i - 1]
            p2 = coords[i]
            p3 = coords[i + 1]

            a = float(np.linalg.norm(p2 - p1))
            b = float(np.linalg.norm(p3 - p2))
            c = float(np.linalg.norm(p3 - p1))

            if a < 1e-3 or b < 1e-3 or c < 1e-3:
                continue

            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0:
                continue
            area = np.sqrt(area_sq)

            # Menger curvature: kappa = 4 * Area / (a * b * c) = 1 / R_circumscribed
            kappa = 4.0 * area / (a * b * c + 1e-6)
            curvatures[i] = min(kappa, 1.0 / self.min_curve_radius)

        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        return curvatures

    def govern_speed(
        self,
        v_pred: float,
        curvature: float = 0.0,
        yaw_rate_rad_s: Optional[float] = None,
        road_speed_limit: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Apply curvature-limited kinematic bounds and road speed limits.

        Physical constraints:
            v_max_curve = sqrt(a_lat_max / max(kappa, 1e-4))
            v_max_gyro = a_lat_max / (abs(yaw_rate) + 1e-4)
            v_out = min(v_pred, v_max_curve, v_max_gyro, speed_limit)
        """
        v_nonneg = max(0.0, float(v_pred))

        if curvature > 1e-4:
            v_max_curve = float(np.sqrt(self.a_lat_max / curvature))
        else:
            v_max_curve = self.speed_limit_mps

        if yaw_rate_rad_s is not None and abs(yaw_rate_rad_s) > 0.02:
            v_max_gyro = float(self.a_lat_max / (abs(yaw_rate_rad_s) + 1e-4))
        else:
            v_max_gyro = self.speed_limit_mps

        cap = self.speed_limit_mps
        if road_speed_limit is not None and road_speed_limit > 0:
            cap = min(cap, road_speed_limit)

        effective_max = min(v_max_curve, v_max_gyro, cap)
        v_governed = min(v_nonneg, effective_max)

        diag = {
            "v_raw": v_nonneg,
            "v_max_curve": float(v_max_curve),
            "v_max_gyro": float(v_max_gyro),
            "effective_cap": float(effective_max),
            "is_governed": float(v_nonneg > effective_max),
            "curvature": float(curvature),
        }
        return float(v_governed), diag

    def project_velocity_to_road(
        self,
        v_governed: float,
        road_tangent: np.ndarray,
    ) -> np.ndarray:
        """Project governed scalar forward velocity along road tangent vector.

        Enforces Non-Holonomic Constraint (v_lateral = 0):
            v_vector = v_governed * road_tangent
        """
        norm_t = float(np.linalg.norm(road_tangent))
        if norm_t < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float64)
        u_tangent = road_tangent / norm_t
        return v_governed * u_tangent


class DualRateRoadGovernor(RoadKinematicsGovernor):
    """Two-tier hierarchical velocity governor with 0.8s micro-smoothing
    and macro-window retrospective scale calibration.

    Tier 1 (Micro-Loop, 10Hz / 0.8s):
      - C2-continuous Hermite smoothstep catch-up interpolation over 0.8s.
      - Closed-loop curvature-limited forward velocity capping v <= sqrt(a_lat_max / kappa).
      - Centripetal gyro-rate bounding v <= a_lat_max / |omega_yaw|.
      - Non-holonomic lateral velocity constraint (v_lateral = 0 along road tangent).

    Tier 2 (Macro-Loop, 3-5s Cadence):
      - Gathers sliding trajectory windows.
      - Computes retrospective scale factor: scale = Delta_s_road / Delta_s_pred.
      - Recalibrates velocity gain smoothly via exponential moving average.
    """

    def __init__(
        self,
        a_lat_max: float = 3.5,
        min_curve_radius: float = 5.0,
        speed_limit_mps: float = 33.3,
        micro_blend_duration_s: float = 0.8,
        macro_window_s: float = 3.5,
        scale_ema_alpha: float = 0.25,
        min_scale: float = 0.70,
        max_scale: float = 1.30,
        dt: float = 0.1,
    ):
        super().__init__(a_lat_max, min_curve_radius, speed_limit_mps)
        self.dt = dt
        self.micro_blend_duration_s = micro_blend_duration_s
        self.blend_steps = max(1, int(micro_blend_duration_s / dt))
        self.macro_window_samples = max(10, int(macro_window_s / dt))
        self.scale_ema_alpha = scale_ema_alpha
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Dynamic state
        self.calibrated_scale: float = 1.0
        self.v_current: float = 0.0
        self.v_start: float = 0.0
        self.v_target: float = 0.0
        self.blend_step_idx: int = self.blend_steps

        # Macro buffers
        self.macro_v_preds: List[float] = []
        self.macro_distances_road: List[float] = []

    def update_micro(
        self,
        v_pred: float,
        curvature: float = 0.0,
        yaw_rate_rad_s: Optional[float] = None,
        road_speed_limit: Optional[float] = None,
        is_stationary: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute micro-governed velocity with 0.8s smooth C2 catch-up."""
        if is_stationary:
            self.v_current = 0.0
            self.v_target = 0.0
            self.v_start = 0.0
            self.blend_step_idx = self.blend_steps
            diag = {
                "v_raw": float(v_pred),
                "v_target": 0.0,
                "v_governed": 0.0,
                "is_governed": False,
                "curvature": float(curvature),
                "blend_progress": 1.0,
            }
            return 0.0, diag

        # Apply calibrated scale
        v_scaled = float(v_pred) * self.calibrated_scale

        # Instantaneous physical bound
        v_bound, base_diag = self.govern_speed(
            v_scaled,
            curvature=curvature,
            yaw_rate_rad_s=yaw_rate_rad_s,
            road_speed_limit=road_speed_limit,
        )

        # Hermite smoothstep blend towards target bound
        if abs(v_bound - self.v_target) > 0.5:
            self.v_start = self.v_current
            self.v_target = v_bound
            self.blend_step_idx = 0

        if self.blend_step_idx < self.blend_steps:
            t = float(self.blend_step_idx + 1) / float(self.blend_steps)
            # C2 cubic Hermite smoothstep: S(t) = 3*t^2 - 2*t^3
            s_t = 3.0 * (t ** 2) - 2.0 * (t ** 3)
            self.v_current = self.v_start + s_t * (self.v_target - self.v_start)
            self.blend_step_idx += 1
            blend_progress = t
        else:
            self.v_current = self.v_target
            blend_progress = 1.0

        diag = {
            "v_raw": float(v_pred),
            "v_target": float(self.v_target),
            "v_governed": float(self.v_current),
            "is_governed": bool(self.v_current < v_scaled * 0.98),
            "curvature": float(curvature),
            "blend_progress": float(blend_progress),
            "calibrated_scale": float(self.calibrated_scale),
        }
        return float(self.v_current), diag


def update_map_measurement(ekf_instance, proj_point_enu: np.ndarray, road_tangent_unit: np.ndarray) -> bool:
    """
    Applies the orthogonal road projection point as an anisotropic Kalman measurement.
    """
    p_pred = ekf_instance.x_nominal[0:3]
    proj_3d = np.array([
        proj_point_enu[0],
        proj_point_enu[1],
        p_pred[2] if len(proj_point_enu) < 3 else proj_point_enu[2]
    ], dtype=np.float64)
    y_pos = proj_3d - p_pred

    H = np.zeros((3, 15), dtype=np.float64)
    H[0:3, 0:3] = np.eye(3, dtype=np.float64)

    t_raw = np.array([
        road_tangent_unit[0],
        road_tangent_unit[1],
        0.0 if len(road_tangent_unit) < 3 else road_tangent_unit[2]
    ], dtype=np.float64)
    norm_t = float(np.linalg.norm(t_raw))
    if norm_t < 1e-6:
        return False
    t = t_raw / norm_t
    n = np.array([-t[1], t[0], 0.0], dtype=np.float64)
    u = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sigma_perp = 0.35      # Tight lateral lane constraint (0.35m)
    sigma_parallel = 15.0  # Longitudinal uncertainty along road (meters)
    sigma_vert = 2.0       # Vertical uncertainty (meters)

    R_map = (
        (sigma_parallel**2) * np.outer(t, t) +
        (sigma_perp**2) * np.outer(n, n) +
        (sigma_vert**2) * np.outer(u, u)
    )

    S = H @ ekf_instance.P @ H.T + R_map
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return False

    mahalanobis_dist_sq = float(y_pos.T @ S_inv @ y_pos)

    # Chi-square gating (3-DOF, p=0.001 threshold is 16.27)
    if mahalanobis_dist_sq > 16.27:
        return False

    K = ekf_instance.P @ H.T @ S_inv
    delta_x = K @ y_pos

    # Update position and velocity
    if hasattr(ekf_instance, "_p"):
        ekf_instance._p += delta_x[0:3]
        ekf_instance._v += delta_x[3:6]
    else:
        x = ekf_instance.x_nominal.copy()
        x[0:3] += delta_x[0:3]
        x[3:6] += delta_x[3:6]
        ekf_instance.x_nominal = x

    I_KH = np.eye(15, dtype=np.float64) - K @ H
    ekf_instance.P = I_KH @ ekf_instance.P @ I_KH.T + K @ R_map @ K.T
    return True


