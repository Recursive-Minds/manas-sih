import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92")
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample, FusedPosition
from sih.data.geo import geodetic_to_enu, enu_to_geodetic
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.calibration.mount import MountCalibrator
import torch

def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

class GatedHeadingAndNHCEKF:
    def __init__(
        self,
        gnss_pos_std: float = 2.0,
        gnss_hdg_std: float = 0.02,        # rad (~1.1 deg)
        gyro_noise_std: float = 0.005,     # rad/s/sqrt(Hz)
        gyro_bias_rw_std: float = 0.00001, # rad/s^2/sqrt(Hz)
        nhc_lateral_std: float = 0.15,     # m/s
        nhc_heading_gain: float = 0.015,
    ):
        self.r_pos = gnss_pos_std**2
        self.r_hdg = gnss_hdg_std**2
        self.q_theta = gyro_noise_std**2
        self.q_bg = gyro_bias_rw_std**2
        self.r_nhc = nhc_lateral_std**2
        self.nhc_gain = nhc_heading_gain
        self.reset()
        
    def reset(self, initial_gnss: Optional[GNSSSample] = None):
        self._initialised = False
        self._last_ts = None
        self._last_gnss_ts = None
        self._ref = np.zeros(3)
        
        self._p = np.zeros(3, dtype=np.float64)
        self._v = np.zeros(3, dtype=np.float64)
        self._heading_rad = 0.0
        self._bg_z = 0.0
        self._w_raw_last = 0.0
        
        self._P_hdg = np.diag([0.05**2, (0.0005)**2]).astype(np.float64)
        self._P_pos = np.diag([4.0, 4.0, 9.0]).astype(np.float64)
        
        self._speed_scale = 1.0
        self._last_ai_speed = None
        self._stat_count = 0
        self.last_nhc_dtheta = 0.0
        
        if initial_gnss is not None:
            self.init_from_gnss(initial_gnss)

    def init_from_gnss(self, g: GNSSSample):
        self._ref = np.array([g.latitude_deg, g.longitude_deg, g.altitude_m], dtype=np.float64)
        self._p = np.zeros(3, dtype=np.float64)
        self._last_ts = g.timestamp_ns
        self._last_gnss_ts = g.timestamp_ns
        if g.bearing_deg is not None:
            self._heading_rad = float(np.radians(g.bearing_deg))
        else:
            self._heading_rad = 0.0
        if g.speed_mps is not None and g.speed_mps > 0.5:
            self._v = np.array([g.speed_mps * np.sin(self._heading_rad), g.speed_mps * np.cos(self._heading_rad), 0.0], dtype=np.float64)
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
            
        w_raw = float(sample.gyro_vehicle[2])
        self._w_raw_last = w_raw
        w_corrected = w_raw - self._bg_z
        
        # 1. ZARU when stationary
        is_stationary = (vel is not None and vel.motion_state == "STATIONARY") or (vel is not None and vel.forward_speed_mps < 0.2)
        if is_stationary:
            self._stat_count += 1
            if self._stat_count > 5:
                # Direct ZARU update on bg_z
                H = np.array([[0.0, 1.0]])
                y = w_raw - self._bg_z
                S = float((H @ self._P_hdg @ H.T)[0, 0] + 0.001**2)
                K = (self._P_hdg @ H.T / S).flatten()
                self._heading_rad = (self._heading_rad + K[0] * y) % (2.0 * np.pi)
                self._bg_z += K[1] * y
                I_KH = np.eye(2) - np.outer(K, H)
                self._P_hdg = I_KH @ self._P_hdg @ I_KH.T + np.outer(K, K) * (0.001**2)
                w_corrected = 0.0
        else:
            self._stat_count = 0

        # 2. Heading & Bias Propagation:
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self._heading_rad = (self._heading_rad - w_corrected * dt) % (2.0 * np.pi)
        Q = np.array([[self.q_theta * dt, 0.0], [0.0, self.q_bg * dt]], dtype=np.float64)
        self._P_hdg = F @ self._P_hdg @ F.T + Q
        
        # 3. Forward Speed Propagation
        if vel is not None and vel.forward_speed_mps is not None:
            self._last_ai_speed = float(vel.forward_speed_mps)
            v_fwd = 0.0 if is_stationary else float(vel.forward_speed_mps * self._speed_scale)
        else:
            v_fwd = 0.0
            
        ve = v_fwd * np.sin(self._heading_rad)
        vn = v_fwd * np.cos(self._heading_rad)
        self._v = np.array([ve, vn, 0.0], dtype=np.float64)
        
        self._p[0] += ve * dt
        self._p[1] += vn * dt
        self._P_pos[0:2, 0:2] += np.eye(2) * ((0.5 * dt)**2 + 0.01)

        # 4. NHC MEASUREMENT UPDATE
        self.last_nhc_dtheta = 0.0
        if not is_stationary and v_fwd > 2.0 and abs(w_corrected) > 0.015:
            # Expected centripetal acceleration: a_lat = v_fwd * w_corrected
            # In our vehicle coordinate frame:
            # Right axis is index 1. Turning counter-clockwise (positive w_z) produces negative lateral acceleration (to the right).
            a_lat_meas = float(sample.accel_vehicle[1])
            # Check slip innovation:
            y_slip = a_lat_meas + (v_fwd * w_corrected)
            
            dtheta = float(np.clip((y_slip / (v_fwd * 5.0)) * self.nhc_gain * dt, -0.02, 0.02))
            self._heading_rad = (self._heading_rad + dtheta) % (2.0 * np.pi)
            self.last_nhc_dtheta = float(np.degrees(abs(dtheta)))
            
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
        self._p = gnss_enu.copy()
        sig_p = float(max(gnss.accuracy_h_m, 1.0))
        self._P_pos = np.diag([sig_p**2, sig_p**2, (sig_p*2.0)**2])
        
        # Straight-line gated GNSS COG Heading update
        if gnss.speed_mps is not None and gnss.bearing_deg is not None and gnss.speed_mps > 3.0:
            gnss_hdg_rad = float(np.radians(gnss.bearing_deg))
            y_hdg = wrap_pi(gnss_hdg_rad - self._heading_rad)
            
            is_straight = abs(self._w_raw_last) < 0.02  # < 1.1 deg/s straight line
            if is_straight:
                # Full update on heading and gyro bias bg_z
                H = np.array([[1.0, 0.0]], dtype=np.float64)
                S = float((H @ self._P_hdg @ H.T)[0, 0] + self.r_hdg)
                K = (self._P_hdg @ H.T / S).flatten()
                
                self._heading_rad = (self._heading_rad + K[0] * y_hdg) % (2.0 * np.pi)
                self._bg_z += K[1] * y_hdg
                
                I_KH = np.eye(2) - np.outer(K, H)
                self._P_hdg = I_KH @ self._P_hdg @ I_KH.T + np.outer(K, K) * self.r_hdg
            else:
                # During turns: update heading only, do not contaminate bg_z
                self._heading_rad = (self._heading_rad + 0.3 * y_hdg) % (2.0 * np.pi)
            
            # Speed scale update
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
            covariance=np.eye(15),
            mode="GNSS_AIDED" if (self._last_gnss_ts and (ts - self._last_gnss_ts)*1e-9 < 3.0) else "INS_ONLY_BLACKOUT",
            gnss_outage_duration_s=max(0.0, (ts - (self._last_gnss_ts or ts))*1e-9)
        )

trip_s1 = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

def get_trip_inferences(trip):
    ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std = ckpt.get("norm_std", np.ones((8, 1), dtype=np.float32))
    
    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples:
        calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    
    acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    
    N = len(feats)
    window_size = 100
    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w = np.vstack([pad, feats[:i+1]]).T
        else:
            w = feats[i - window_size + 1 : i + 1].T
        windows.append(w)
        
    windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
    preds = []
    with torch.no_grad():
        for b in range(0, N, 1024):
            x = torch.from_numpy(windows_norm[b : b + 1024]).to(device)
            p, _ = model(x)
            preds.extend(p.cpu().numpy().flatten())
    return calib_samples, np.array(preds, dtype=np.float32)

print("Precomputing inferences...")
calib_s1, v_s1 = get_trip_inferences(trip_s1)
calib_s2, v_s2 = get_trip_inferences(trip_s2)

def run_single_eval(trip, calib_samples, v_preds, bo_start_s, bo_dur_s):
    ekf = GatedHeadingAndNHCEKF()
    ekf.reset(trip.gnss_samples[0])
    
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(bo_start_s * 1e9)
    bo_end_ns = bo_start_ns + int(bo_dur_s * 1e9)
    
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    
    est_pts = []
    gt_pts = []
    nhc_dthetas = []
    bg_z_history = []
    
    for i, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr > bo_end_ns + int(2e9):
            break
            
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                ekf.update_gnss(g)
            gnss_idx += 1
            
        calib = calib_samples[i]
        fwd_v = float(v_preds[i])
        m_state = "STATIONARY" if fwd_v < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=fwd_v, speed_variance=1.0, motion_state=m_state)
        fused = ekf.predict(calib, vel)
        
        bg_z_history.append((float((t_curr - t0_ns)*1e-9), float(np.degrees(ekf._bg_z))))
        
        if bo_start_ns <= t_curr <= bo_end_ns:
            est_pts.append(fused.position_enu_m[:2])
            g_enu = geodetic_to_enu(
                trip.gnss_samples[min(gnss_idx, n_gnss-1)].latitude_deg,
                trip.gnss_samples[min(gnss_idx, n_gnss-1)].longitude_deg,
                0.0,
                trip.reference_lat_deg,
                trip.reference_lon_deg,
                0.0
            )
            gt_pts.append(g_enu[:2])
            nhc_dthetas.append(ekf.last_nhc_dtheta)
            
    est_pts = np.array(est_pts)
    gt_pts = np.array(gt_pts)
    
    dists = np.sqrt(np.sum(np.diff(gt_pts, axis=0)**2, axis=1))
    tot_dist = max(float(np.sum(dists)), 1.0)
    
    final_err = float(np.linalg.norm(est_pts[-1] - gt_pts[-1]))
    drift_pct = (final_err / tot_dist) * 100.0
    
    return {
        "final_error_m": final_err,
        "drift_pct": drift_pct,
        "dist_m": tot_dist,
        "nhc_dthetas": nhc_dthetas,
        "bg_z_history": bg_z_history,
        "pre_bo_bg_z": bg_z_history[-len(nhc_dthetas)][1]
    }

print("\nEvaluating Scenario A (30s @ 120s on S-S1)...")
res_sc1 = run_single_eval(trip_s1, calib_s1, v_s1, 120.0, 30.0)
print(f"Scenario A with Gated Heading & NHC EKF:")
print(f"  Final Position Error: {res_sc1['final_error_m']:.2f} m (Old was 119.08 m)")
print(f"  Drift Percentage:     {res_sc1['drift_pct']:.2f} % (Old was 23.34 %)")
print(f"  Pre-Blackout bg_z:    {res_sc1['pre_bo_bg_z']:.4f} deg/s")
print(f"  NHC dtheta (mean):    {np.mean(res_sc1['nhc_dthetas']):.4f} deg/step")
print(f"  NHC dtheta (max):     {np.max(res_sc1['nhc_dthetas']):.4f} deg/step")

print("\nEvaluating Scenario B (60s @ 300s on S-S1)...")
res_sc2 = run_single_eval(trip_s1, calib_s1, v_s1, 300.0, 60.0)
print(f"Scenario B with Gated Heading & NHC EKF:")
print(f"  Final Position Error: {res_sc2['final_error_m']:.2f} m (Old was 254.07 m)")
print(f"  Drift Percentage:     {res_sc2['drift_pct']:.2f} % (Old was 31.21 %)")
print(f"  Pre-Blackout bg_z:    {res_sc2['pre_bo_bg_z']:.4f} deg/s")
print(f"  NHC dtheta (mean):    {np.mean(res_sc2['nhc_dthetas']):.4f} deg/step")
print(f"  NHC dtheta (max):     {np.max(res_sc2['nhc_dthetas']):.4f} deg/step")
