import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF, skew
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PURE 6-AXIS IMU ADVANCED BREAKTHROUGH EVALUATION ({device}) ===", flush=True)

    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    def get_inferences(trip):
        calibrator = MountCalibrator(window_size=100)
        for g in trip.gnss_samples: calibrator.observe_gnss(g)
        calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
        acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
        gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)

        # Zero-phase Savitzky-Golay filtering on gyro z to remove high-freq mechanical vibration without lag
        w_z_raw = gyr[:, 2]
        if len(w_z_raw) > 15:
            w_z_filt = savgol_filter(w_z_raw, window_length=15, polyorder=2)
            gyr[:, 2] = w_z_filt

        feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
        N = len(feats)
        window_size = 100

        norm_mean = ckpt_v.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
        norm_std  = ckpt_v.get("norm_std",  np.ones((8, 1),  dtype=np.float32))

        windows = []
        for i in range(N):
            if i < window_size:
                pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
                w   = np.vstack([pad, feats[:i+1]]).T
            else:
                w   = feats[i - window_size + 1 : i + 1].T
            windows.append((w - norm_mean) / norm_std)

        preds_v = []
        with torch.no_grad():
            for b in range(0, N, 2048):
                xv = torch.from_numpy(np.squeeze(np.array(windows[b : b + 2048], dtype=np.float32), axis=1) if len(np.array(windows[b : b + 2048]).shape) == 4 else np.array(windows[b : b + 2048], dtype=np.float32)).to(device)
                pv, _ = model_v(xv)
                preds_v.extend(pv.cpu().numpy().flatten())

        return calib_samples, np.array(preds_v, dtype=np.float32)

    calib_s1, v_s1 = get_inferences(trip_s1)
    calib_s2, v_s2 = get_inferences(trip_s2)

    sweep_df = pd.read_csv(SWEEP_CSV)

    def run_sweep(enable_high_speed_anchor=True):
        records = []
        for idx, row in sweep_df.iterrows():
            trip_name = str(row["trip"])
            t_start   = float(row["start_time_s"])
            duration  = float(row["duration_s"])
            dist_m    = float(row["distance_m"])

            if "S-S1" in trip_name:
                trip = trip_s1; calib = calib_s1; v_preds = v_s1
            else:
                trip = trip_s2; calib = calib_s2; v_preds = v_s2

            t0_ns = trip.imu_samples[0].timestamp_ns
            bo_start_ns = t0_ns + int(t_start * 1e9)
            bo_end_ns   = bo_start_ns + int(duration * 1e9)

            gt_start_sample = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
            gt_end_sample   = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

            gt_start_enu = geodetic_to_enu(gt_start_sample.latitude_deg, gt_start_sample.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_end_enu   = geodetic_to_enu(gt_end_sample.latitude_deg, gt_end_sample.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_disp      = gt_end_enu - gt_start_enu

            warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
            warmup_gnss = min([g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=trip.gnss_samples[0])

            ekf = ErrorStateEKF(
                turn_threshold_rad_s=np.radians(1.5),
                cooldown_duration_s=0.5,
                max_gyro_bias_rad_s=np.radians(0.5),
                initial_speed_scale=1.00
            )
            ekf.init_from_gnss(warmup_gnss)

            gnss_idx = 0
            n_gnss   = len(trip.gnss_samples)
            est_pts  = []

            for j, imu in enumerate(trip.imu_samples):
                t_curr = imu.timestamp_ns
                if t_curr < warmup_start_ns: continue
                if t_curr > bo_end_ns + int(1e9): break

                while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                    g = trip.gnss_samples[gnss_idx]
                    if g.timestamp_ns < bo_start_ns:
                        ekf.update_gnss(g)
                        if g.speed_mps is not None and g.speed_mps > 3.0 and g.bearing_deg is not None:
                            ekf._heading_rad = float(np.radians(g.bearing_deg))
                    gnss_idx += 1

                cal = calib[j]
                v_fwd = float(v_preds[j])

                m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
                vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)
                fused = ekf.predict(cal, vel)

                # High-Speed Kinematic Direction Stiffness Update (Var(psi) ~ 1 / v^2)
                if enable_high_speed_anchor and bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 12.0: # v > 12 m/s (43 km/h)
                    # Velocity direction in ENU frame: psi_kin = atan2(v_east, v_north)
                    v_enu = ekf._v[:2]
                    v_norm = np.linalg.norm(v_enu)
                    if v_norm > 5.0:
                        psi_kin = np.arctan2(v_enu[0], v_enu[1])
                        y_psi = (psi_kin - ekf._heading_rad + np.pi) % (2.0 * np.pi) - np.pi
                        
                        # Variance scales inversely with speed squared: r_psi = (0.02 / (v / 10))^2
                        r_psi = max(0.005, 0.05 / (v_fwd / 10.0))**2
                        
                        H_p = np.zeros((1, 15), dtype=np.float64)
                        H_p[0, 8] = 1.0
                        R_p = np.array([[r_psi]])
                        
                        S_p = H_p @ ekf._P @ H_p.T + R_p
                        K_p = ekf._P @ H_p.T @ np.linalg.inv(S_p)
                        dx = (K_p @ np.array([y_psi])).flatten()

                        ekf._heading_rad = (ekf._heading_rad + dx[8]) % (2.0 * np.pi)
                        ekf._bg += dx[12:15]
                        ekf._bg = np.clip(ekf._bg, -ekf.max_bg, ekf.max_bg)

                if bo_start_ns <= t_curr <= bo_end_ns:
                    est_pts.append(fused.position_enu_m[:2])

            est_pts = np.array(est_pts)
            if len(est_pts) < 2: continue
            est_disp = est_pts[-1] - est_pts[0]
            final_err_m = float(np.linalg.norm(est_disp - gt_disp))
            drift_pct   = (final_err_m / max(dist_m, 1.0)) * 100.0

            records.append({"distance_m": dist_m, "error_m": final_err_m, "drift_pct": drift_pct})

        df = pd.DataFrame(records)
        long_df = df[df["distance_m"] > 500.0]

        return {
            "median_drift_pct": float(df["drift_pct"].median()),
            "worst_drift_pct": float(df["drift_pct"].max()),
            "long_median_drift_pct": float(long_df["drift_pct"].median()) if len(long_df) > 0 else float(df["drift_pct"].median()),
            "median_err_m": float(df["error_m"].median()),
            "long_median_err_m": float(long_df["error_m"].median()) if len(long_df) > 0 else float(df["error_m"].median())
        }

    print("\n==========================================================================")
    print("      PURE 6-AXIS IMU ADVANCED SIGNAL PROCESSING BENCHMARK                ")
    print("==========================================================================")
    res_base = run_sweep(enable_high_speed_anchor=False)
    res_adv  = run_sweep(enable_high_speed_anchor=True)

    print(f"1. Pure 6-Axis IMU (Base Physical 3D EKF):")
    print(f"   - Combined Median Drift %:        {res_base['median_drift_pct']:.2f}%")
    print(f"   - Long Outages (>500m) Median %:  {res_base['long_median_drift_pct']:.2f}%")
    print(f"   - Median Position Error:          {res_base['median_err_m']:.2f} m")

    print(f"\n2. + Zero-Phase SG Filter + High-Speed Kinematic Stiffness (Pure IMU):")
    print(f"   - Combined Median Drift %:        {res_adv['median_drift_pct']:.2f}%")
    print(f"   - Long Outages (>500m) Median %:  {res_adv['long_median_drift_pct']:.2f}%")
    print(f"   - Median Position Error:          {res_adv['median_err_m']:.2f} m")
    print("==========================================================================")

if __name__ == "__main__":
    main()
