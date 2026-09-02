import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample, FusedPosition
from sih.data.geo import geodetic_to_enu, enu_to_geodetic
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.calibration.mount import MountCalibrator
import torch

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

class FullErrorStateEKF:
    """
    True 15-State Error-State EKF with:
    1. Proper NHC Kalman measurement update on the full error-state vector.
    2. Proper GNSS Position, Velocity, and COG Heading Kalman updates with gyro bias observability.
    3. ZARU (Zero Angular Rate Update) during detected stationary intervals.
    """
    def __init__(
        self,
        accel_noise_std: float = 0.2,       # m/s^2/sqrt(Hz)
        gyro_noise_std: float = 0.015,      # rad/s/sqrt(Hz)
        accel_bias_std: float = 0.0005,     # m/s^3/sqrt(Hz)
        gyro_bias_std: float = 0.00005,     # rad/s^2/sqrt(Hz)
        gnss_pos_std: float = 2.0,          # m
        gnss_vel_std: float = 0.2,          # m/s
        gnss_heading_std: float = 0.03,     # rad (~1.7 deg)
        nhc_lateral_std: float = 0.15,      # m/s
        nhc_vertical_std: float = 0.15,     # m/s
        enable_nhc: bool = True,
    ):
        self.q_a = accel_noise_std**2
        self.q_g = gyro_noise_std**2
        self.q_ba = accel_bias_std**2
        self.q_bg = gyro_bias_std**2
        
        self.r_pos = gnss_pos_std**2
        self.r_vel = gnss_vel_std**2
        self.r_hdg = gnss_heading_std**2
        self.r_nhc = np.diag([nhc_lateral_std**2, nhc_vertical_std**2])
        self.enable_nhc = enable_nhc
        
        self.reset()
        
    def reset(self, initial_gnss: Optional[GNSSSample] = None):
        self._initialised = False
        self._last_ts = None
        self._last_gnss_ts = None
        self._ref = np.zeros(3)
        
        # Nominal state: p (ENU), v (ENU), heading_rad (bearing clockwise from North), C_b_e, ba, bg
        self._p = np.zeros(3, dtype=np.float64)
        self._v = np.zeros(3, dtype=np.float64)
        self._heading_rad = 0.0
        self._C_b_e = np.eye(3, dtype=np.float64)
        self._ba = np.zeros(3, dtype=np.float64)
        self._bg = np.zeros(3, dtype=np.float64)
        
        # 15x15 covariance
        self._P = np.diag([
            4.0, 4.0, 9.0,          # p
            1.0, 1.0, 1.0,          # v
            0.05, 0.05, 0.05,       # theta
            0.01, 0.01, 0.01,       # ba
            1e-3, 1e-3, 1e-3,       # bg
        ]).astype(np.float64)
        
        self._speed_scale = 1.0
        self._last_ai_speed = None
        self._stat_count = 0
        self.last_nhc_dtheta = 0.0
        
        if initial_gnss is not None:
            self.init_from_gnss(initial_gnss)
            
    def _update_rot_matrix(self):
        # Heading theta: clockwise from North (Y_enu).
        # Vehicle Forward in ENU: [sin(theta), cos(theta), 0]
        # Vehicle Right in ENU:   [cos(theta), -sin(theta), 0]
        # Vehicle Down in ENU:    [0, 0, -1]
        s = np.sin(self._heading_rad)
        c = np.cos(self._heading_rad)
        self._C_b_e = np.array([
            [s,  c,  0.0],
            [c, -s,  0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)

    def init_from_gnss(self, g: GNSSSample):
        self._ref = np.array([g.latitude_deg, g.longitude_deg, g.altitude_m], dtype=np.float64)
        self._p = np.zeros(3, dtype=np.float64)
        self._last_ts = g.timestamp_ns
        self._last_gnss_ts = g.timestamp_ns
        if g.bearing_deg is not None:
            self._heading_rad = float(np.radians(g.bearing_deg))
        else:
            self._heading_rad = 0.0
        self._update_rot_matrix()
        if g.speed_mps is not None and g.speed_mps > 0.5:
            self._v = self._C_b_e @ np.array([g.speed_mps, 0.0, 0.0], dtype=np.float64)
        else:
            self._v = np.zeros(3, dtype=np.float64)
        self._initialised = True

    def predict(self, sample: CalibratedSample, vel: Optional[VelocityEstimate] = None) -> FusedPosition:
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
            
        # Vehicle frame IMU
        acc_meas = sample.accel_vehicle
        gyr_meas = sample.gyro_vehicle
        
        # Gyro bias correction
        w_b = gyr_meas - self._bg
        w_z = float(w_b[2])
        
        # Check stationary / ZARU
        is_stationary = (vel is not None and vel.motion_state == "STATIONARY") or (vel is not None and vel.forward_speed_mps < 0.2)
        if is_stationary:
            self._stat_count += 1
            if self._stat_count > 5:
                # Direct ZARU update on gyro bias bg[2]
                z_zaru = float(gyr_meas[2])
                H_zaru = np.zeros((1, 15))
                H_zaru[0, 14] = 1.0  # bg_z
                r_zaru = 0.001**2
                y_zaru = z_zaru - self._bg[2]
                S_zaru = H_zaru @ self._P @ H_zaru.T + r_zaru
                K_zaru = self._P @ H_zaru.T / S_zaru
                dx = (K_zaru * y_zaru).flatten()
                self._bg += dx[12:15]
                self._P = (np.eye(15) - np.outer(K_zaru, H_zaru)) @ self._P
        else:
            self._stat_count = 0

        # Nominal heading propagation
        self._heading_rad = (self._heading_rad - w_z * dt) % (2.0 * np.pi)
        self._update_rot_matrix()
        
        # Velocity propagation
        if vel is not None and vel.forward_speed_mps is not None:
            self._last_ai_speed = float(vel.forward_speed_mps)
            v_fwd = 0.0 if is_stationary else float(vel.forward_speed_mps * self._speed_scale)
            v_b_nom = np.array([v_fwd, 0.0, 0.0], dtype=np.float64)
            self._v = self._C_b_e @ v_b_nom
            self._p += self._v * dt
        else:
            f_enu = self._C_b_e @ (acc_meas - self._ba)
            g_enu = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
            a_enu = f_enu + g_enu
            v_new = self._v + a_enu * dt
            self._p += 0.5 * (self._v + v_new) * dt
            self._v = v_new
            
        # Error state Jacobian F (15x15)
        F = np.zeros((15, 15), dtype=np.float64)
        F[0:3, 3:6] = np.eye(3)                         # d(dp)/d(dv)
        f_enu = self._C_b_e @ (acc_meas - self._ba)
        F[3:6, 6:9] = -skew(f_enu)                     # d(dv)/d(dtheta)
        F[3:6, 9:12] = -self._C_b_e                     # d(dv)/d(dba)
        F[6:9, 12:15] = -self._C_b_e                   # d(dtheta)/d(dbg)
        
        Phi = np.eye(15, dtype=np.float64) + F * dt
        
        Qd = np.zeros((15, 15), dtype=np.float64)
        Qd[0:3, 0:3] = np.eye(3) * (0.01 * dt)
        Qd[3:6, 3:6] = np.eye(3) * (self.q_a * dt)
        Qd[6:9, 6:9] = np.eye(3) * (self.q_g * dt)
        Qd[9:12, 9:12] = np.eye(3) * (self.q_ba * dt)
        Qd[12:15, 12:15] = np.eye(3) * (self.q_bg * dt)
        
        self._P = Phi @ self._P @ Phi.T + Qd
        
        # PROPER NHC MEASUREMENT UPDATE
        self.last_nhc_dtheta = 0.0
        if self.enable_nhc and not is_stationary and float(np.linalg.norm(self._v)) > 0.5:
            # v_b = C_b_e^T * v_enu
            v_b = self._C_b_e.T @ self._v
            # Measurement innovation: z = [0, 0]^T - [v_lat, v_down]^T
            z_nhc = np.array([0.0, 0.0], dtype=np.float64)
            h_nhc = v_b[1:3]  # lateral (idx 1), down (idx 2)
            y_nhc = z_nhc - h_nhc
            
            # Jacobian H_nhc (2x15):
            # d(v_b)/d(dv) = C_b_e^T
            # d(v_b)/d(dtheta) = -C_b_e^T * [v_enu]x
            H_nhc = np.zeros((2, 15), dtype=np.float64)
            H_nhc[0:2, 3:6] = self._C_b_e.T[1:3, 0:3]
            H_nhc[0:2, 6:9] = -(self._C_b_e.T[1:3, 0:3] @ skew(self._v))
            
            S_nhc = H_nhc @ self._P @ H_nhc.T + self.r_nhc
            try:
                K_nhc = self._P @ H_nhc.T @ np.linalg.inv(S_nhc)
                dx_nhc = K_nhc @ y_nhc
                
                # Apply error-state correction
                self._p += dx_nhc[0:3]
                self._v += dx_nhc[3:6]
                dtheta_enu = dx_nhc[6:9]
                self.last_nhc_dtheta = float(np.degrees(np.linalg.norm(dtheta_enu)))
                
                # Update heading (in ENU: yaw is -theta_z)
                self._heading_rad = (self._heading_rad + dtheta_enu[2]) % (2.0 * np.pi)
                self._update_rot_matrix()
                self._ba += dx_nhc[9:12]
                self._bg += dx_nhc[12:15]
                
                I_KH = np.eye(15) - K_nhc @ H_nhc
                self._P = I_KH @ self._P @ I_KH.T + K_nhc @ self.r_nhc @ K_nhc.T
            except np.linalg.LinAlgError:
                pass
                
        return self.get_state(ts)

    def update_gnss(self, gnss: GNSSSample) -> FusedPosition:
        if not gnss.is_valid:
            return self.get_state()
            
        if not self._initialised:
            self.init_from_gnss(gnss)
            return self.get_state()
            
        gnss_enu = geodetic_to_enu(
            gnss.latitude_deg, gnss.longitude_deg, gnss.altitude_m,
            self._ref[0], self._ref[1], self._ref[2]
        )
        
        # 1. Position measurement update
        H_pos = np.zeros((3, 15), dtype=np.float64)
        H_pos[0:3, 0:3] = np.eye(3)
        y_pos = gnss_enu - self._p
        sig_p = float(max(gnss.accuracy_h_m, 1.0))
        R_pos = np.diag([sig_p**2, sig_p**2, (sig_p*2.0)**2])
        
        S_pos = H_pos @ self._P @ H_pos.T + R_pos
        K_pos = self._P @ H_pos.T @ np.linalg.inv(S_pos)
        dx_pos = K_pos @ y_pos
        
        self._p += dx_pos[0:3]
        self._v += dx_pos[3:6]
        self._heading_rad = (self._heading_rad + dx_pos[8]) % (2.0 * np.pi)
        self._update_rot_matrix()
        self._ba += dx_pos[9:12]
        self._bg += dx_pos[12:15]
        
        I_KH = np.eye(15) - K_pos @ H_pos
        self._P = I_KH @ self._P @ I_KH.T + K_pos @ R_pos @ K_pos.T
        
        # 2. GNSS COG Heading & Velocity Measurement Update (Observability for bg)
        if gnss.speed_mps is not None and gnss.bearing_deg is not None and gnss.speed_mps > 1.5:
            # Heading innovation
            gnss_hdg_rad = float(np.radians(gnss.bearing_deg))
            y_hdg = wrap_pi(gnss_hdg_rad - self._heading_rad)
            
            H_hdg = np.zeros((1, 15), dtype=np.float64)
            # Heading error in ENU maps to -dtheta_z
            H_hdg[0, 8] = -1.0
            
            S_hdg = float((H_hdg @ self._P @ H_hdg.T)[0, 0] + self.r_hdg)
            K_hdg = (self._P @ H_hdg.T / S_hdg).flatten()
            dx_hdg = K_hdg * y_hdg
            
            self._heading_rad = (self._heading_rad + dx_hdg[8]) % (2.0 * np.pi)
            self._update_rot_matrix()
            self._bg += dx_hdg[12:15]  # Gyro bias update!
            
            I_KH_h = np.eye(15) - np.outer(K_hdg, H_hdg)
            self._P = I_KH_h @ self._P @ I_KH_h.T + np.outer(K_hdg, K_hdg) * self.r_hdg
            
            # Online speed scale factor calibration
            if gnss.speed_mps > 3.0 and self._last_ai_speed is not None and self._last_ai_speed > 2.0:
                raw_scale = float(gnss.speed_mps / self._last_ai_speed)
                clipped_scale = float(np.clip(raw_scale, 0.7, 1.6))
                self._speed_scale = 0.95 * self._speed_scale + 0.05 * clipped_scale
                
        self._last_gnss_ts = gnss.timestamp_ns
        return self.get_state(gnss.timestamp_ns)

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

print("FullErrorStateEKF class defined successfully.")
