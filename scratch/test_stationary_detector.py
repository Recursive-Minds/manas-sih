import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Optional
from scipy.spatial.transform import Rotation as R
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample, FusedPosition
from sih.data.geo import geodetic_to_enu, enu_to_geodetic

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_PATH   = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class Unified15StateEKF:
    """
    Unified 15-State Error-State EKF with Real Kalman Gain NHC,
    Corrected Heading Innovation Sign, Robust Physical Stationary Detection,
    Gated Gyro Bias Learning, and Sanity Clipping.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.2,
        gyro_noise_std: float = 0.005,
        accel_bias_std: float = 0.0005,
        gyro_bias_std: float = 0.000005,
        gnss_pos_std: float = 2.0,
        gnss_vel_std: float = 0.2,
        gnss_heading_std: float = 0.02,
        nhc_lateral_std: float = 0.25,
        nhc_vertical_std: float = 0.25,
        turn_threshold_rad_s: float = 0.015,   # ~0.86 deg/s
        cooldown_duration_s: float = 2.0,      # 2s cooldown
        max_gyro_bias_rad_s: float = 0.01745,  # +/- 1.0 deg/s MEMS bound
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
        self.enable_nhc = enable_nhc
        self.enable_zupt = enable_zupt

        self.reset()

    def reset(self, initial_gnss: Optional[GNSSSample] = None):
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

        # 15x15 Covariance
        self._P = np.diag([
            4.0, 4.0, 9.0,               # pos
            1.0, 1.0, 1.0,               # vel
            0.01, 0.01, 0.01,            # att
            1e-4, 1e-4, 1e-4,            # ba
            (0.0002)**2, (0.0002)**2, (0.0002)**2, # bg
        ]).astype(np.float64)

        self._last_turn_ts_s: float = -100.0
        self._stat_count: int = 0
        self._speed_scale: float = 1.0
        self._last_ai_speed: Optional[float] = None
        self.last_nhc_dtheta_deg: float = 0.0
        self._accel_buf: list[float] = []

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

        t_now_s = ts * 1e-9

        raw_gyro = sample.gyro_vehicle
        raw_acc  = sample.accel_vehicle
        w_corr   = raw_gyro - self._bg
        w_z_corr = float(w_corr[2])

        # Turn & Cooldown detection
        if abs(w_z_corr) > self.turn_thresh:
            self._last_turn_ts_s = t_now_s

        # Sliding window physical stationary detector
        a_norm = float(np.linalg.norm(raw_acc))
        self._accel_buf.append(a_norm)
        if len(self._accel_buf) > 20:
            self._accel_buf.pop(0)
        a_var = float(np.var(self._accel_buf)) if len(self._accel_buf) >= 10 else 1.0
        g_norm_err = abs(a_norm - 9.80665)

        is_physical_rest = (a_var < 0.05 and g_norm_err < 0.6 and float(np.linalg.norm(raw_gyro)) < 0.04)

        is_stationary = (
            is_physical_rest or
            (vel is not None and vel.motion_state == "STATIONARY") or
            (vel is not None and vel.forward_speed_mps < 0.2)
        )

        if is_stationary:
            self._stat_count += 1
            v_fwd = 0.0
            w_z_corr = 0.0
        else:
            self._stat_count = 0
            if vel is not None and vel.forward_speed_mps is not None:
                self._last_ai_speed = float(vel.forward_speed_mps)
                v_fwd = float(vel.forward_speed_mps) * self._speed_scale
            else:
                v_fwd = 0.0

        # Propagate nominal heading
        self._heading_rad = (self._heading_rad - w_z_corr * dt) % (2.0 * np.pi)
        yaw_enu_rad = np.pi / 2.0 - self._heading_rad
        self._q = R.from_euler("z", yaw_enu_rad)

        ve = v_fwd * np.sin(self._heading_rad)
        vn = v_fwd * np.cos(self._heading_rad)
        self._v = np.array([ve, vn, 0.0], dtype=np.float64)

        self._p[0] += ve * dt
        self._p[1] += vn * dt

        # Propagate 15-state covariance P <- F P F^T + Q
        F = np.eye(15, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        q_pos = 0.01 * dt
        q_vel = ((vel.speed_variance if vel else 0.5) * dt)**2
        q_att = (self.gyro_noise_std * dt)**2
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

            in_cooldown = (t_now_s - self._last_turn_ts_s) < self.cooldown_dur
            if not in_cooldown:
                self._bg += dx[12:15]
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
        if not in_cooldown:
            self._bg += dx_p[12:15]
            self._bg = np.clip(self._bg, -self.max_bg, self.max_bg)

        I_KH = np.eye(15) - K_p @ H_p
        self._P = I_KH @ self._P @ I_KH.T + K_p @ R_p @ K_p.T

        # Heading & Speed Update when moving
        if gnss.speed_mps is not None and gnss.bearing_deg is not None and gnss.speed_mps > 3.0:
            gnss_hdg_rad = float(np.radians(gnss.bearing_deg))
            y_hdg = wrap_pi(gnss_hdg_rad - self._heading_rad)

            H_hdg = np.zeros((1, 15), dtype=np.float64)
            H_hdg[0, 8] = 1.0
            R_hdg = np.array([[self.gnss_heading_std**2]], dtype=np.float64)

            S_hdg = H_hdg @ self._P @ H_hdg.T + R_hdg
            K_hdg = self._P @ H_hdg.T @ np.linalg.inv(S_hdg)
            dx_h = (K_hdg * y_hdg).flatten()

            self._heading_rad = (self._heading_rad + dx_h[8]) % (2.0 * np.pi)
            yaw_enu_rad = np.pi / 2.0 - self._heading_rad
            self._q = R.from_euler("z", yaw_enu_rad)

            if not in_cooldown:
                self._bg += dx_h[12:15]
                self._bg = np.clip(self._bg, -self.max_bg, self.max_bg)

            I_KH = np.eye(15) - K_hdg @ H_hdg
            self._P = I_KH @ self._P @ I_KH.T + K_hdg @ R_hdg @ K_hdg.T

            if self._last_ai_speed is not None and self._last_ai_speed > 2.0:
                raw_scale = float(gnss.speed_mps / self._last_ai_speed)
                clipped_scale = float(np.clip(raw_scale, 0.7, 1.6))
                self._speed_scale = 0.95 * self._speed_scale + 0.05 * clipped_scale

        self._last_gnss_ts = gnss.timestamp_ns
        return self.get_state(gnss.timestamp_ns)

    @property
    def b_g(self) -> np.ndarray:
        return self._bg

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


def main():
    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model  = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    def get_inferences(trip):
        calibrator = MountCalibrator(window_size=100)
        for g in trip.gnss_samples:
            calibrator.observe_gnss(g)
        calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
        acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
        gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
        feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
        N = len(feats)
        window_size = 100
        windows = []
        norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
        norm_std  = ckpt.get("norm_std",  np.ones((8, 1),  dtype=np.float32))
        for i in range(N):
            if i < window_size:
                pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
                w   = np.vstack([pad, feats[:i+1]]).T
            else:
                w   = feats[i - window_size + 1 : i + 1].T
            windows.append(w)
        windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
        preds = []
        with torch.no_grad():
            for b in range(0, N, 1024):
                x = torch.from_numpy(windows_norm[b : b + 1024]).to(device)
                p, _ = model(x)
                preds.extend(p.cpu().numpy().flatten())
        return calib_samples, np.array(preds, dtype=np.float32)

    calib_s1, v_s1 = get_inferences(trip_s1)
    calib_s2, v_s2 = get_inferences(trip_s2)

    sweep_df = pd.read_csv(SWEEP_CSV)

    def eval_trip(trip, calib, v_preds, sub_df):
        results = []
        for _, row in sub_df.iterrows():
            t_start  = float(row["start_time_s"])
            duration = float(row["duration_s"])

            ekf = Unified15StateEKF(
                turn_threshold_rad_s=np.radians(0.86),
                cooldown_duration_s=2.0,
                max_gyro_bias_rad_s=np.radians(0.5), # Tight clip +/- 0.5 deg/s
                nhc_lateral_std=0.25,
                nhc_vertical_std=0.25,
            )
            ekf.init_from_gnss(trip.gnss_samples[0])

            t0_ns = trip.imu_samples[0].timestamp_ns
            bo_start_ns = t0_ns + int(t_start * 1e9)
            bo_end_ns   = bo_start_ns + int(duration * 1e9)

            gnss_idx = 0
            n_gnss   = len(trip.gnss_samples)
            est_pts  = []
            gt_pts   = []

            for j, imu in enumerate(trip.imu_samples):
                t_curr = imu.timestamp_ns
                if t_curr > bo_end_ns + int(2e9):
                    break

                while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                    g = trip.gnss_samples[gnss_idx]
                    if not (bo_start_ns <= g.timestamp_ns <= bo_end_ns):
                        ekf.update_gnss(g)
                    gnss_idx += 1

                cal = calib[j]
                v_fwd = float(v_preds[j])
                m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
                vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)
                fused = ekf.predict(cal, vel)

                if bo_start_ns <= t_curr <= bo_end_ns:
                    est_pts.append(fused.position_enu_m[:2])
                    g_curr = trip.gnss_samples[min(gnss_idx, n_gnss-1)]
                    g_enu  = geodetic_to_enu(g_curr.latitude_deg, g_curr.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)
                    gt_pts.append(g_enu[:2])

            est_pts = np.array(est_pts)
            gt_pts  = np.array(gt_pts)

            if len(gt_pts) < 2: continue
            dists    = np.sqrt(np.sum(np.diff(gt_pts, axis=0)**2, axis=1))
            tot_dist = max(float(np.sum(dists)), 1.0)
            final_err = float(np.linalg.norm(est_pts[-1] - gt_pts[-1]))
            drift_pct = (final_err / tot_dist) * 100.0

            results.append({
                "scenario": row["scenario"], "trip": row["trip"],
                "old_error_m": row["final_error_m"], "old_drift_pct": row["drift_pct"],
                "new_error_m": final_err, "new_drift_pct": drift_pct
            })
        return results

    sc1_df = sweep_df[sweep_df["trip"].str.contains("S-S1")]
    sc2_df = sweep_df[sweep_df["trip"].str.contains("S-S2")]

    print("\nEvaluating physical rest detector & position-uncorrupted EKF...", flush=True)
    t0 = time.time()
    r1 = eval_trip(trip_s1, calib_s1, v_s1, sc1_df)
    r2 = eval_trip(trip_s2, calib_s2, v_s2, sc2_df)
    df_res = pd.DataFrame(r1 + r2)

    med   = df_res["new_drift_pct"].median()
    worst = df_res["new_drift_pct"].max()

    print(f"Time: {time.time()-t0:.2f}s", flush=True)
    print(f"Original Baseline Target:  Median < 73.54%,  Worst-Case < 806.03%", flush=True)
    print(f"Physical Rest EKF:         Median = {med:.2f}%,   Worst-Case = {worst:.2f}%", flush=True)

    print("\nScenario breakdown:", flush=True)
    for i, r in df_res.iterrows():
        print(f"  [{i+1:2d}/50] {r['scenario'][:32]:32s} old={r['old_drift_pct']:7.2f}% ({r['old_error_m']:6.1f}m) -> new={r['new_drift_pct']:7.2f}% ({r['new_error_m']:6.1f}m)", flush=True)

if __name__ == "__main__":
    main()
