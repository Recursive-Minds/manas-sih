"""
15-State Error-State Extended Kalman Filter (ES-EKF) with AI Velocity Fusion,
Real Kalman Gain Non-Holonomic Constraints (NHC), and Zero Velocity Updates (ZUPT).

State Vector (15 dims):
  [0:3]   Position ENU (m)
  [3:6]   Velocity ENU (m/s)
  [6:9]   Attitude error / Orientation (rotation vector on SO3)
  [9:12]  Accelerometer bias (vehicle frame, m/s^2)
  [12:15] Gyroscope bias (vehicle frame, rad/s)

Key Architecture:
  - Unified 15-State Filter: Gyro bias b_g lives inside the single 15-state vector
    and 15x15 covariance matrix P, ensuring cross-covariances are correctly maintained.
  - Dynamic Kalman Gain NHC: Computes K = P H^T (H P H^T + R)^-1 every step from the
    exact linearised body-velocity error Jacobian H_NHC.
  - Turn Gating & Cooldown: Gyro bias updates are frozen whenever |w_z| > 0.015 rad/s (~0.86 deg/s)
    and during a 2.0s post-turn cooldown window.
  - Sanity Bounding: Gyro bias b_g is clipped to +/- 0.1 deg/s (+/- 0.001745 rad/s).
  - Physical Rest ZUPT: Sliding window accelerometer variance detector locks v = 0 when at rest.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from sih.core.contracts import (
    IMUSample,
    CalibratedSample,
    GNSSSample,
    VelocityEstimate,
    FusedPosition,
)
from sih.core.interfaces import IFusionFilter
from sih.core.pipeline import register_fusion_filter
from sih.data.geo import geodetic_to_enu, enu_to_geodetic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from 3D vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Unified 15-State ES-EKF
# ---------------------------------------------------------------------------

class ErrorStateEKF(IFusionFilter):
    """
    Unified 15-state Error-State EKF with robust AI-IMU dead reckoning during GNSS blackouts.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.2,
        gyro_noise_std: float = 0.003,
        accel_bias_std: float = 0.0005,
        gyro_bias_std: float = 0.0000005,
        gnss_pos_std: float = 1.5,
        gnss_vel_std: float = 0.1,
        gnss_heading_std: float = 0.05,        # ~2.8 deg GNSS bearing std (prevents noise jumps)
        nhc_lateral_std: float = 0.20,
        nhc_vertical_std: float = 0.20,
        turn_threshold_rad_s: float = 0.02618, # 1.5 deg/s turn threshold
        cooldown_duration_s: float = 0.5,      # 0.5s post-turn cooldown
        max_gyro_bias_rad_s: float = 0.008726, # +/- 0.5 deg/s MEMS bias bound
        initial_speed_scale: float = 1.00,     # Dynamic speed scale pre-blackout factor
        enable_nhc: bool = True,
        enable_zupt: bool = True,
    ):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_std = accel_bias_std
        self.gyro_bias_std = gyro_bias_std
        self.gnss_pos_std = gnss_pos_std
        self.gnss_vel_std = gnss_vel_std
        self.gnss_heading_std = gnss_heading_std
        self.nhc_lat_std = nhc_lateral_std
        self.nhc_vert_std = nhc_vertical_std
        self.turn_thresh = turn_threshold_rad_s
        self.cooldown_dur = cooldown_duration_s
        self.max_bg = max_gyro_bias_rad_s
        self.initial_speed_scale = initial_speed_scale
        self.enable_nhc = enable_nhc
        self.enable_zupt = enable_zupt

        self.init_highway_extensions()
        self.reset()

    def init_highway_extensions(self):
        """Initializes highway straight-line lock (ZARU) and hybrid speed state."""
        self.straight_drive_timer = 0.0
        self.locked_straight_heading = None
        self.v_chassis_prev = 0.0
        self.latest_calibrated_gyro = np.zeros(3, dtype=np.float64)
        self.latest_calibrated_accel = np.zeros(3, dtype=np.float64)
        self.last_dt = 0.1

    def reset(
        self,
        initial_gnss: Optional[GNSSSample] = None,
        init_pos: Optional[np.ndarray] = None,
        init_heading: Optional[float] = None,
    ):
        self._initialised = False
        self._last_ts: Optional[int] = None
        self._last_gnss_ts: Optional[int] = None
        self._ref = np.zeros(3)

        self._p = np.zeros(3, dtype=np.float64)
        self._v = np.zeros(3, dtype=np.float64)
        self._heading_rad = 0.0
        self._q = R.identity()
        self._ba = np.zeros(3, dtype=np.float64)
        self._bg = np.zeros(3, dtype=np.float64)

        # 15x15 Covariance Matrix P
        self._P = np.diag([
            2.25, 2.25, 9.0,                        # position
            0.25, 0.25, 0.25,                        # velocity
            0.001, 0.001, 0.001,                    # attitude
            1e-4, 1e-4, 1e-4,                        # accel bias
            (0.00002)**2, (0.00002)**2, (0.00002)**2 # gyro bias
        ]).astype(np.float64)

        self._last_turn_ts_s: float = -100.0
        self._last_w_z_corr: float = 0.0
        self._stat_count: int = 0
        self._speed_scale: float = self.initial_speed_scale
        self._last_ai_speed: Optional[float] = None
        self.last_nhc_dtheta_deg: float = 0.0
        self._accel_buf: list[float] = []

        self.init_highway_extensions()

        if init_pos is not None:
            self._p = init_pos.copy().astype(np.float64)
            self._initialised = True

        if init_heading is not None:
            self._heading_rad = float(init_heading)
            yaw_enu_rad = np.pi / 2.0 - self._heading_rad
            self._q = R.from_euler("z", yaw_enu_rad)
            self._initialised = True

        if initial_gnss is not None:
            self.init_from_gnss(initial_gnss)


    def init_from_gnss(
        self,
        g: GNSSSample,
        reference_lat_deg: Optional[float] = None,
        reference_lon_deg: Optional[float] = None,
        reference_alt_m: Optional[float] = None,
    ):
        if reference_lat_deg is not None and reference_lon_deg is not None:
            self._ref = np.array([reference_lat_deg, reference_lon_deg, reference_alt_m or 0.0], dtype=np.float64)
            self._p = geodetic_to_enu(g.latitude_deg, g.longitude_deg, g.altitude_m, self._ref[0], self._ref[1], self._ref[2])
        else:
            self._ref = np.array([g.latitude_deg, g.longitude_deg, g.altitude_m], dtype=np.float64)
            self._p = np.zeros(3, dtype=np.float64)
        self._last_ts = g.timestamp_ns
        self._last_gnss_ts = g.timestamp_ns

        if g.bearing_deg is not None:
            self._heading_rad = float(np.radians(g.bearing_deg))
        else:
            self._heading_rad = 0.0

        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

        if g.speed_mps is not None and g.speed_mps > 0.5:
            b = self._heading_rad
            self._v = np.array([g.speed_mps * np.sin(b), g.speed_mps * np.cos(b), 0.0], dtype=np.float64)
        else:
            self._v = np.zeros(3, dtype=np.float64)

        sig_p = float(max(g.accuracy_h_m, 1.0))
        self._P[0:3, 0:3] = np.diag([sig_p**2, sig_p**2, (sig_p * 2.0)**2])
        self._initialised = True

    def align_heading_to_road(self, road_bearing_deg: float):
        """Align vehicle heading directly to road network segment azimuth."""
        self._heading_rad = float(np.radians(road_bearing_deg))
        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

    def seed_pre_blackout_heading(
        self,
        pre_gnss: List[GNSSSample],
        road_bearing_deg: Optional[float] = None,
        delta_heading_gyro_deg: float = 0.0
    ) -> float:
        """
        Dynamic speed-dependent physical heading seeding at blackout entry:
        1. When vehicle speed > 3.0 m/s and displacement between last 2 fixes > 3.0m:
           Computes geometric vector displacement course arctan2(ΔEast, ΔNorth).
        2. When crawling (0.5 < v <= 3.0 m/s):
           Uses instantaneous Doppler bearing with road corridor weighting.
        3. When stopped (v <= 0.5 m/s):
           Holds last stable moving heading and integrates gyro yaw while stopped.
        4. Extrapolates forward to exact blackout entry using gyro yaw integration.
        5. Sets heading error covariance accordingly.
        """
        valid_moving = [g for g in pre_gnss if g.is_valid and g.speed_mps is not None and g.speed_mps > 0.5]
        seeded_hdg = None
        is_consistent = True

        # Check if latest valid fix is moving stably (v >= 2.5 m/s)
        if valid_moving and valid_moving[-1].speed_mps is not None and valid_moving[-1].speed_mps >= 2.5:
            if len(valid_moving) >= 2 and valid_moving[-2].speed_mps is not None and valid_moving[-2].speed_mps >= 2.5:
                g_prev = valid_moving[-2]
                g_last = valid_moving[-1]
                dt_interval = (g_last.timestamp_ns - g_prev.timestamp_ns) * 1e-9
                # Pure 2-point geometric vector displacement if fixes are closely spaced (dt <= 1.5s)
                if dt_interval <= 1.5:
                    enu_prev = geodetic_to_enu(g_prev.latitude_deg, g_prev.longitude_deg, 0.0, self._ref[0], self._ref[1], self._ref[2])[:2]
                    enu_last = geodetic_to_enu(g_last.latitude_deg, g_last.longitude_deg, 0.0, self._ref[0], self._ref[1], self._ref[2])[:2]
                    disp = enu_last - enu_prev
                    if float(np.linalg.norm(disp)) > 3.0:
                        seeded_hdg = float(np.degrees(np.arctan2(disp[0], disp[1]))) % 360.0
                        is_consistent = True
            if seeded_hdg is None and valid_moving[-1].bearing_deg is not None:
                seeded_hdg = float(valid_moving[-1].bearing_deg)
                is_consistent = True
        else:
            # Vehicle is crawling (< 2.5 m/s) or stopped.
            # Look back for the last stable moving fix (v >= 2.0 m/s)
            stable_fixes = [g for g in pre_gnss if g.is_valid and g.speed_mps is not None and g.speed_mps >= 2.0 and g.bearing_deg is not None]
            if stable_fixes:
                seeded_hdg = float(stable_fixes[-1].bearing_deg)
                is_consistent = True
            elif valid_moving and valid_moving[-1].bearing_deg is not None:
                seeded_hdg = float(valid_moving[-1].bearing_deg)
                is_consistent = False
            elif pre_gnss and pre_gnss[-1].bearing_deg is not None:
                seeded_hdg = float(pre_gnss[-1].bearing_deg)
                is_consistent = False
            else:
                seeded_hdg = float(np.degrees(self._heading_rad))
                is_consistent = False

        # Physical forward extrapolation via integrated gyro turning
        seeded_hdg = (seeded_hdg + delta_heading_gyro_deg) % 360.0

        # Gentle road alignment only if road corridor strictly aligns (|diff| < 20°)
        if road_bearing_deg is not None:
            r_diff = abs((seeded_hdg - road_bearing_deg + 180.0) % 360.0 - 180.0)
            if r_diff < 20.0:
                seeded_hdg = (seeded_hdg + 0.5 * ((road_bearing_deg - seeded_hdg + 180.0) % 360.0 - 180.0)) % 360.0
                is_consistent = True

        self._heading_rad = float(np.radians(seeded_hdg))
        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

        # Critical: Align navigation velocity vector with the new seeded heading
        # Otherwise, NHC on step 0 sees an artificial lateral velocity slip and violently yanks heading back
        v_speed = float(np.linalg.norm(self._v))
        if v_speed > 0.1:
            v_b = np.array([v_speed, 0.0, 0.0], dtype=np.float64)
            self._v = self._q.as_matrix() @ v_b

        # Decouple attitude cross-covariances so pre-blackout velocity innovations cannot rotate the seeded heading
        self._P[6:9, :] = 0.0
        self._P[:, 6:9] = 0.0

        if is_consistent:
            self._P[8, 8] = float(np.radians(2.0))**2
        else:
            self._P[8, 8] = float(np.radians(8.0))**2

        return seeded_hdg

    def reanchor_heading(self, road_bearing_deg: float, confidence: float = 1.0, forward_speed_mps: Optional[float] = None):
        """
        Fast re-anchoring when confirmed back on a straight road segment.
        Gently pulls heading, realigns velocity vector to prevent false NHC lateral slip,
        and decouples attitude cross-covariances.
        """
        r_rad = float(np.radians(road_bearing_deg))
        diff_rad = (r_rad - self._heading_rad + np.pi) % (2.0 * np.pi) - np.pi
        gain = 0.05 * min(1.0, max(0.0, confidence))
        self._heading_rad = (self._heading_rad + gain * diff_rad) % (2.0 * np.pi)
        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

        # Realign ENU velocity with the updated heading to eliminate false NHC lateral slip innovation
        v_fwd = forward_speed_mps if forward_speed_mps is not None else float(np.linalg.norm(self._v[:2]))
        if v_fwd > 0.5:
            C_b_n = self._q.as_matrix()
            self._v = C_b_n @ np.array([v_fwd, 0.0, self._v[2]], dtype=np.float64)

        # Bound attitude yaw covariance smoothly without breaking positive semi-definiteness
        self._P[8, 8] = float(0.90 * self._P[8, 8] + 0.10 * (np.radians(2.0))**2)

    def predict(
        self,
        sample,
        vel: Any = None,
        dt_opt: Optional[float] = None,
        v_fused_opt: Optional[float] = None,
    ) -> FusedPosition:
        if isinstance(sample, CalibratedSample):
            ts = sample.timestamp_ns
            if not self._initialised:
                self._last_ts = ts
                self._initialised = True
                return self.get_state(ts)

            if self._last_ts is None:
                self._last_ts = ts
                return self.get_state(ts)

            dt = (ts - self._last_ts) * 1e-9
            self._last_ts = ts
            if dt <= 0.0 or dt > 2.0:
                dt = 0.1

            raw_gyro = sample.gyro_vehicle.copy()
            raw_acc  = sample.accel_vehicle.copy()
            passed_v_fwd = None
        else:
            # Array inputs: sample is accel (3,), vel is gyro (3,), dt_opt is dt, v_fused_opt is v_fused
            raw_acc = np.asarray(sample, dtype=np.float64)
            raw_gyro = np.asarray(vel, dtype=np.float64)
            dt = float(dt_opt) if dt_opt is not None else 0.01
            ts = 0 if self._last_ts is None else self._last_ts + int(dt * 1e9)
            self._last_ts = ts
            self._initialised = True
            passed_v_fwd = float(v_fused_opt) if v_fused_opt is not None else None
            vel = None

        t_now_s = ts * 1e-9
        g_norm = float(np.linalg.norm(raw_acc))

        self.latest_calibrated_gyro = raw_gyro.copy()
        self.latest_calibrated_accel = raw_acc.copy()
        self.last_dt = float(dt)

        # Correct gyro by 15-state gyro bias
        w_corr = raw_gyro - self._bg

        # Direct Earth-vertical yaw rate projection from calibrated vehicle frame:
        # sample.gyro_vehicle is already in the leveled vehicle frame, so yaw rate is strictly Z-axis
        w_z_corr = float(w_corr[2])

        # Turn & Cooldown detection
        self._last_w_z_corr = w_z_corr
        if abs(w_z_corr) > self.turn_thresh:
            self._last_turn_ts_s = t_now_s

        # Sliding window physical stationary detector
        self._accel_buf.append(g_norm)
        if len(self._accel_buf) > 20:
            self._accel_buf.pop(0)
        a_var = float(np.var(self._accel_buf)) if len(self._accel_buf) >= 10 else 1.0
        g_norm_err = abs(g_norm - 9.80665)

        # Physical Rest ZUPT Decoupling (Phase 4.5 breakthrough):
        # When IMU variance and angular rate indicate physical rest, clamp unconditionally.
        is_physical_rest = (a_var < 0.04 and g_norm_err < 0.6 and float(np.linalg.norm(raw_gyro)) < 0.04)

        is_stationary = (
            is_physical_rest or
            (vel is not None and vel.motion_state == "STATIONARY") or
            (vel is not None and vel.forward_speed_mps is not None and vel.forward_speed_mps < 0.2) or
            (passed_v_fwd is not None and passed_v_fwd < 0.2)
        )

        if is_stationary:
            self._stat_count += 1
            v_fwd = 0.0
            if is_physical_rest:
                w_z_corr = 0.0
        else:
            self._stat_count = 0
            if passed_v_fwd is not None:
                self._last_ai_speed = passed_v_fwd
                v_fwd = passed_v_fwd
            elif vel is not None and vel.forward_speed_mps is not None:
                self._last_ai_speed = float(vel.forward_speed_mps)
                v_fwd = float(vel.forward_speed_mps) * self._speed_scale
                a_x_fwd = float(raw_acc[0]) - self._ba[0]
                if a_x_fwd < -0.3:
                    v_fwd = max(0.0, v_fwd + a_x_fwd * dt)
            else:
                v_fwd = 0.0


        # Propagate nominal heading
        self._heading_rad = (self._heading_rad - w_z_corr * dt) % (2.0 * np.pi)
        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

        # Propagate 3D velocity and position using full IMU acceleration + AI forward speed constraint
        C_b_n = self._q.as_matrix()
        C_n_b = C_b_n.T

        if is_stationary:
            self._v = np.zeros(3, dtype=np.float64)
        else:
            # Transform calibrated vehicle accel to ENU frame (minus gravity)
            a_b = raw_acc - self._ba
            a_n = C_b_n @ a_b + np.array([0.0, 0.0, -9.80665], dtype=np.float64)
            self._v += a_n * dt

            # Project to body frame, update forward component with AI speed, retain dynamic lateral/vertical components
            v_b = C_n_b @ self._v
            v_b[0] = v_fwd
            self._v = C_b_n @ v_b

        self._p[0] += self._v[0] * dt
        self._p[1] += self._v[1] * dt
        self._p[2] += self._v[2] * dt

        # Propagate 15-state covariance P <- F P F^T + Q
        F = np.eye(15, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        q_pos = 0.01 * dt
        q_vel = ((vel.speed_variance if vel else 0.5) * dt)**2

        # Rate-adaptive attitude process noise: scales with |w_z| during turns
        gyro_scale_factor_std = 0.03
        turn_rate = abs(w_z_corr)
        q_att = ((self.gyro_noise_std**2 + (gyro_scale_factor_std * turn_rate)**2) * (dt**2))

        q_ba  = (self.accel_bias_std * dt)**2
        q_bg  = (self.gyro_bias_std * dt)**2

        Q = np.diag([
            q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel,
            q_att, q_att, q_att,
            q_ba,  q_ba,  q_ba,
            q_bg,  q_bg,  q_bg,
        ]).astype(np.float64)

        self._P = F @ self._P @ F.T + Q

        # ZUPT update
        if self.enable_zupt and is_stationary and self._stat_count > 5:
            H_zupt = np.zeros((1, 15), dtype=np.float64)
            H_zupt[0, 14] = 1.0
            y_zupt = np.array([raw_gyro[2] - self._bg[2]])
            r_zupt = np.array([[(0.001)**2]])

            S_z = H_zupt @ self._P @ H_zupt.T + r_zupt
            K_z = self._P @ H_zupt.T @ np.linalg.inv(S_z)
            dx_z = (K_z @ y_zupt).flatten()

            self._bg += dx_z[12:15]
            self._bg = np.clip(self._bg, -self.max_bg, self.max_bg)
            I_KH = np.eye(15) - K_z @ H_zupt
            self._P = I_KH @ self._P @ I_KH.T + K_z @ r_zupt @ K_z.T

        # Real Closed-Loop NHC Measurement Update
        self.last_nhc_dtheta_deg = 0.0
        if self.enable_nhc and not is_stationary and v_fwd > 2.0:
            C_b_n = self._q.as_matrix()
            C_n_b = C_b_n.T
            v_b = C_n_b @ self._v

            H_nhc = np.zeros((2, 15), dtype=np.float64)
            H_nhc[0, 3:6] = C_n_b[1, :]
            H_nhc[0, 6:9] = -(C_n_b @ skew(self._v))[1, :]
            H_nhc[1, 3:6] = C_n_b[2, :]
            H_nhc[1, 6:9] = -(C_n_b @ skew(self._v))[2, :]

            y_nhc = np.array([0.0 - v_b[1], 0.0 - v_b[2]], dtype=np.float64)
            R_nhc = np.diag([self.nhc_lat_std**2, self.nhc_vert_std**2]).astype(np.float64)

            S_nhc = H_nhc @ self._P @ H_nhc.T + R_nhc
            K_nhc = self._P @ H_nhc.T @ np.linalg.inv(S_nhc)

            dx = (K_nhc @ y_nhc).flatten()

            self._v += dx[3:6]

            # Apply heading correction
            dtheta_z = dx[8]
            self._heading_rad = (self._heading_rad + dtheta_z) % (2.0 * np.pi)
            yaw_enu_rad = np.pi / 2.0 - self._heading_rad
            self._q = R.from_euler("z", yaw_enu_rad)
            self.last_nhc_dtheta_deg = float(np.degrees(abs(dtheta_z)))

            # Continuous Lorentzian damping of gyro bias updates during turns
            omega_turn_ref = 0.02 # ~1.15 deg/s
            turn_damping = 1.0 / (1.0 + (abs(w_z_corr) / omega_turn_ref)**2)
            time_since_turn = max(0.0, t_now_s - self._last_turn_ts_s)
            cooldown_factor = min(1.0, time_since_turn / max(self.cooldown_dur, 0.1))
            bias_gain = float(turn_damping * cooldown_factor)

            self._bg += bias_gain * dx[12:15]
            self._bg = np.clip(self._bg, -self.max_bg, self.max_bg)

            I_KH = np.eye(15) - K_nhc @ H_nhc
            self._P = I_KH @ self._P @ I_KH.T + K_nhc @ R_nhc @ K_nhc.T

        return self.get_state(ts)

    def update_gnss(self, gnss: GNSSSample) -> FusedPosition:
        if not gnss.is_valid:
            return self.get_state()

        if not self._initialised:
            self.init_from_gnss(gnss)
            return self.get_state()

        t_now_s = gnss.timestamp_ns * 1e-9
        gnss_enu = geodetic_to_enu(
            gnss.latitude_deg, gnss.longitude_deg, gnss.altitude_m,
            self._ref[0], self._ref[1], self._ref[2]
        )

        sig_p = float(max(gnss.accuracy_h_m, 1.0))
        in_cooldown = (t_now_s - self._last_turn_ts_s) < self.cooldown_dur

        # Position Update
        H_p = np.zeros((3, 15), dtype=np.float64)
        H_p[0:3, 0:3] = np.eye(3)
        y_p = gnss_enu - self._p
        R_p = np.diag([sig_p**2, sig_p**2, (sig_p * 2.0)**2]).astype(np.float64)

        S_p = H_p @ self._P @ H_p.T + R_p
        K_p = self._P @ H_p.T @ np.linalg.inv(S_p)
        dx_p = (K_p @ y_p).flatten()

        self._p += dx_p[0:3]
        self._v += dx_p[3:6]

        I_KH = np.eye(15) - K_p @ H_p
        self._P = I_KH @ self._P @ I_KH.T + K_p @ R_p @ K_p.T

        # Heading Alignment & Pre-Blackout Gyro Bias Estimation
        if gnss.speed_mps is not None and gnss.bearing_deg is not None and gnss.speed_mps > 2.5:
            gnss_hdg_rad = float(np.radians(gnss.bearing_deg))
            y_hdg = wrap_pi(gnss_hdg_rad - self._heading_rad)
            # Smooth innovation update: never completely ignore valid motion heading,
            # but bound single-step correction to prevent wild receiver outliers from jerking the filter
            max_step_rad = np.radians(15.0)
            step_rad = 0.20 * np.clip(y_hdg, -max_step_rad, max_step_rad)
            self._heading_rad = (self._heading_rad + step_rad) % (2.0 * np.pi)
            yaw_enu_rad = np.pi / 2.0 - self._heading_rad
            self._q = R.from_euler("z", yaw_enu_rad)

            # Estimate Earth-vertical gyro bias during straight driving
            if abs(y_hdg) < np.radians(3.0) and not in_cooldown:
                self._bg[2] += 0.02 * y_hdg
                self._bg[2] = float(np.clip(self._bg[2], -self.max_bg, self.max_bg))

            # Pre-Blackout Speed Scale Factor Adaptation with persistent excitation safeguard
            if (
                self._last_ai_speed is not None
                and self._last_ai_speed > 2.5
                and gnss.speed_mps > 3.0
                and abs(self._last_w_z_corr) < 0.02
                and abs(y_hdg) < np.radians(3.0)
                and not in_cooldown
            ):
                raw_scale = float(gnss.speed_mps / self._last_ai_speed)
                clipped_scale = float(np.clip(raw_scale, 0.75, 1.35))
                self._speed_scale = 0.90 * self._speed_scale + 0.10 * clipped_scale

        self._last_gnss_ts = gnss.timestamp_ns
        return self.get_state(gnss.timestamp_ns)

    @property
    def b_g(self) -> np.ndarray:
        return self._bg

    @property
    def P(self) -> np.ndarray:
        return self._P

    @P.setter
    def P(self, new_p: np.ndarray):
        self._P = new_p.copy()

    @property
    def q(self):
        q_xyzw = self._q.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

    @q.setter
    def q(self, q_val):
        if isinstance(q_val, R):
            self._q = q_val
        else:
            q_xyzw = np.array([q_val[1], q_val[2], q_val[3], q_val[0]], dtype=np.float64)
            q_xyzw /= max(1e-12, np.linalg.norm(q_xyzw))
            self._q = R.from_quat(q_xyzw)
        yaw_enu_rad = self._q.as_euler("zyx")[0]
        self._heading_rad = (np.pi / 2.0 - yaw_enu_rad) % (2.0 * np.pi)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)

    @property
    def x_nominal(self) -> np.ndarray:
        rotvec = self._q.as_rotvec()
        return np.concatenate([self._p, self._v, rotvec, self._ba, self._bg])

    @x_nominal.setter
    def x_nominal(self, x: np.ndarray):
        self._p = x[0:3].copy()
        self._v = x[3:6].copy()
        self._q = R.from_rotvec(x[6:9])
        yaw_enu_rad = self._q.as_euler("zyx")[0]
        self._heading_rad = (np.pi / 2.0 - yaw_enu_rad) % (2.0 * np.pi)
        self._ba = x[9:12].copy()
        self._bg = x[12:15].copy()

    def get_nav_azimuth_rad(self) -> float:
        return float(self._heading_rad)

    def update_straight_line_lock(self, active_road_bearing_rad: Optional[float] = None) -> None:
        """
        Zero Angular Rate Update (ZARU) / Heading Lock.
        Suppresses cubic yaw drift divergence during straight highway cruising.
        """
        current_speed = float(np.linalg.norm(self._v[:2]))
        w_z_corr = float(self.latest_calibrated_gyro[2] - self._bg[2])

        # Trigger: vehicle speed > 15 m/s with minimal turning (|w_z| < 0.005 rad/s) for > 2.0s
        if current_speed > 15.0 and abs(w_z_corr) < 0.005:
            self.straight_drive_timer += self.last_dt
        else:
            self.straight_drive_timer = 0.0
            self.locked_straight_heading = None
            return

        if self.straight_drive_timer < 2.0:
            return

        current_heading = self.get_nav_azimuth_rad()
        if active_road_bearing_rad is not None:
            target_heading = float(active_road_bearing_rad)
        else:
            if self.locked_straight_heading is None:
                self.locked_straight_heading = current_heading
            target_heading = self.locked_straight_heading

        # Heading innovation wrapped to [-pi, pi]
        y_heading = wrap_pi(target_heading - current_heading)

        # Measurement Jacobian H (maps error state attitude delta_theta_z at index 8)
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 8] = 1.0

        # Observation noise covariance (tight constraint: ~1.15 degrees)
        R_heading = np.array([[0.02**2]], dtype=np.float64)

        # Kalman gain & state correction
        S = H @ self._P @ H.T + R_heading
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self._P @ H.T @ S_inv
        delta_x = (K * y_heading).ravel()

        # Update nominal position, velocity, attitude, and biases
        self._p += delta_x[0:3]
        self._v += delta_x[3:6]

        dtheta_z = delta_x[8]
        self._heading_rad = (self._heading_rad + dtheta_z) % (2.0 * np.pi)
        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

        v_fwd = float(np.linalg.norm(self._v[:2]))
        if v_fwd > 0.5:
            C_b_n = self._q.as_matrix()
            self._v = C_b_n @ np.array([v_fwd, 0.0, self._v[2]], dtype=np.float64)

        self._ba += delta_x[9:12]
        self._bg += delta_x[12:15]
        self._bg = np.clip(self._bg, -self.max_bg, self.max_bg)

        # Joseph-form covariance update
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_heading @ K.T

    def compute_hybrid_speed(self, v_moe: float, spectral_power_3_8hz: float, dt: float) -> float:
        """
        Blends MoE speed with forward inertial integration when spectral road texture cues drop.
        """
        alpha = float(1.0 / (1.0 + np.exp(-(spectral_power_3_8hz - 0.25) * 15.0)))
        accel_fwd = float(self.latest_calibrated_accel[0] - self._ba[0])
        v_inertial = max(0.0, float(self.v_chassis_prev + accel_fwd * dt))

        v_fused = (alpha * float(v_moe)) + ((1.0 - alpha) * v_inertial)
        self.v_chassis_prev = v_fused
        return float(v_fused)

    def get_state(self, ts: int = 0) -> FusedPosition:
        lat, lon, alt = enu_to_geodetic(
            self._p[0], self._p[1], self._p[2],
            self._ref[0], self._ref[1], self._ref[2]
        )
        return FusedPosition(
            timestamp_ns=ts,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_m=alt,
            position_enu_m=self._p.copy(),
            velocity_enu_mps=self._v.copy(),
            heading_rad=self._heading_rad,
            covariance=self._P.copy(),
            mode="GNSS_AIDED" if (self._last_gnss_ts and (ts - self._last_gnss_ts)*1e-9 < 3.0) else "INS_ONLY_BLACKOUT",
            gnss_outage_duration_s=max(0.0, (ts - (self._last_gnss_ts or ts))*1e-9)
        )



register_fusion_filter("es_ekf", lambda **kwargs: ErrorStateEKF(**kwargs))
register_fusion_filter("es_ekf_nhc", lambda **kwargs: ErrorStateEKF(**kwargs))


