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

class TrueBodyVelESEKF:
    """
    Standard Error-State Extended Kalman Filter where:
    1. Propagation is driven by calibrated IMU strapdown (accel + gyro).
    2. AI forward velocity and NHC lateral/vertical constraints are fused as a 3D body velocity measurement update:
       z_b = [v_fwd_ai, 0, 0]^T
       h(x) = C_b_e^T * v_enu
       H = [0_3x3, C_b_e^T, -C_b_e^T * [v_enu]x, 0_3x3, 0_3x3]
    3. GNSS updates correct position, velocity, and heading with full error-state covariance coupling into gyro bias bg.
    """
    def __init__(
        self,
        accel_noise_std: float = 0.3,       # m/s^2/sqrt(Hz)
        gyro_noise_std: float = 0.02,       # rad/s/sqrt(Hz)
        accel_bias_std: float = 0.001,      # m/s^3/sqrt(Hz)
        gyro_bias_std: float = 0.0001,      # rad/s^2/sqrt(Hz)
        gnss_pos_std: float = 2.0,          # m
        gnss_vel_std: float = 0.2,          # m/s
        gnss_heading_std: float = 0.03,     # rad
        ai_vel_std: float = 0.5,            # m/s forward speed noise
        nhc_lateral_std: float = 0.2,       # m/s lateral noise
        nhc_vertical_std: float = 0.2,      # m/s vertical noise
        gravity: float = 9.80665,
    ):
        self.q_a = accel_noise_std**2
        self.q_g = gyro_noise_std**2
        self.q_ba = accel_bias_std**2
        self.q_bg = gyro_bias_std**2
        self.g = gravity
        
        self.r_pos = gnss_pos_std**2
        self.r_vel = gnss_vel_std**2
        self.r_hdg = gnss_heading_std**2
        self.R_b = np.diag([ai_vel_std**2, nhc_lateral_std**2, nhc_vertical_std**2])
        
        self.reset()
        
    def reset(self, initial_gnss: Optional[GNSSSample] = None):
        self._initialised = False
        self._last_ts = None
        self._last_gnss_ts = None
        self._ref = np.zeros(3)
        
        self._p = np.zeros(3, dtype=np.float64)
        self._v = np.zeros(3, dtype=np.float64)
        self._heading_rad = 0.0
        self._C_b_e = np.eye(3, dtype=np.float64)
        self._ba = np.zeros(3, dtype=np.float64)
        self._bg = np.zeros(3, dtype=np.float64)
        
        self._P = np.diag([
            4.0, 4.0, 9.0,          # p
            1.0, 1.0, 1.0,          # v
            0.05, 0.05, 0.05,       # theta
            0.01, 0.01, 0.01,       # ba
            1e-3, 1e-3, 1e-3,       # bg
        ]).astype(np.float64)
        
        self._speed_scale = 1.0
        self._last_ai_speed = None
        self.last_nhc_dtheta = 0.0
        
        if initial_gnss is not None:
            self.init_from_gnss(initial_gnss)
            
    def _update_rot_matrix(self):
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
            
        acc_meas = sample.accel_vehicle
        gyr_meas = sample.gyro_vehicle
        
        # 1. Nominal heading propagation
        w_b = gyr_meas - self._bg
        w_z = float(w_b[2])
        self._heading_rad = (self._heading_rad - w_z * dt) % (2.0 * np.pi)
        self._update_rot_matrix()
        
        # 2. Nominal strapdown acceleration & velocity propagation
        f_b = acc_meas - self._ba
        f_enu = self._C_b_e @ f_b
        g_enu = np.array([0.0, 0.0, -self.g], dtype=np.float64)
        a_enu = f_enu + g_enu
        
        v_new = self._v + a_enu * dt
        self._p += 0.5 * (self._v + v_new) * dt
        self._v = v_new
        
        # 3. Covariance propagation
        F = np.zeros((15, 15), dtype=np.float64)
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -skew(f_enu)
        F[3:6, 9:12] = -self._C_b_e
        F[6:9, 12:15] = -self._C_b_e
        
        Phi = np.eye(15, dtype=np.float64) + F * dt
        Qd = np.zeros((15, 15), dtype=np.float64)
        Qd[0:3, 0:3] = np.eye(3) * (0.01 * dt)
        Qd[3:6, 3:6] = np.eye(3) * (self.q_a * dt)
        Qd[6:9, 6:9] = np.eye(3) * (self.q_g * dt)
        Qd[9:12, 9:12] = np.eye(3) * (self.q_ba * dt)
        Qd[12:15, 12:15] = np.eye(3) * (self.q_bg * dt)
        
        self._P = Phi @ self._P @ Phi.T + Qd
        
        # 4. 3D Body Velocity Measurement Update (AI forward speed + NHC lateral/vertical)
        self.last_nhc_dtheta = 0.0
        if vel is not None and vel.forward_speed_mps is not None:
            self._last_ai_speed = float(vel.forward_speed_mps)
            v_ai = float(vel.forward_speed_mps * self._speed_scale)
            if vel.motion_state == "STATIONARY" or v_ai < 0.2:
                v_ai = 0.0
                
            z_b = np.array([v_ai, 0.0, 0.0], dtype=np.float64)
            h_b = self._C_b_e.T @ self._v
            y_b = z_b - h_b
            
            H_b = np.zeros((3, 15), dtype=np.float64)
            H_b[0:3, 3:6] = self._C_b_e.T
            H_b[0:3, 6:9] = -(self._C_b_e.T @ skew(self._v))
            
            # Measurement noise covariance
            Rb = self.R_b.copy()
            if v_ai < 0.2:
                Rb[0, 0] = 0.05**2  # tight stationary constraint
                
            S_b = H_b @ self._P @ H_b.T + Rb
            try:
                K_b = self._P @ H_b.T @ np.linalg.inv(S_b)
                dx_b = K_b @ y_b
                
                self._p += dx_b[0:3]
                self._v += dx_b[3:6]
                dtheta_enu = dx_b[6:9]
                self.last_nhc_dtheta = float(np.degrees(np.linalg.norm(dtheta_enu)))
                
                self._heading_rad = (self._heading_rad + dtheta_enu[2]) % (2.0 * np.pi)
                self._update_rot_matrix()
                self._ba += dx_b[9:12]
                self._bg += dx_b[12:15]
                
                I_KH = np.eye(15) - K_b @ H_b
                self._P = I_KH @ self._P @ I_KH.T + K_b @ Rb @ K_b.T
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
        
        # Position update
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
        
        # GNSS COG Heading update (observability for bg)
        if gnss.speed_mps is not None and gnss.bearing_deg is not None and gnss.speed_mps > 2.0:
            gnss_hdg_rad = float(np.radians(gnss.bearing_deg))
            y_hdg = wrap_pi(gnss_hdg_rad - self._heading_rad)
            
            H_hdg = np.zeros((1, 15), dtype=np.float64)
            H_hdg[0, 8] = -1.0
            
            S_hdg = float((H_hdg @ self._P @ H_hdg.T)[0, 0] + self.r_hdg)
            K_hdg = (self._P @ H_hdg.T / S_hdg).flatten()
            dx_hdg = K_hdg * y_hdg
            
            self._heading_rad = (self._heading_rad + dx_hdg[8]) % (2.0 * np.pi)
            self._update_rot_matrix()
            self._bg += dx_hdg[12:15]
            
            I_KH_h = np.eye(15) - np.outer(K_hdg, H_hdg)
            self._P = I_KH_h @ self._P @ I_KH_h.T + np.outer(K_hdg, K_hdg) * self.r_hdg
            
            if self._last_ai_speed is not None and self._last_ai_speed > 2.0:
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

print("TrueBodyVelESEKF defined successfully.")
