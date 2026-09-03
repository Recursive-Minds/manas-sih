import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.tcn_heading import TCNAttentionHeadingModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
VEL_MODEL_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"
HDG_MODEL_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_heading_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SIH NON-MAP HEADING ENGINE EVALUATION ({device}) ===", flush=True)

    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")

    # Load Velocity Model
    ckpt_v = torch.load(VEL_MODEL_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    # Load AI Heading Model
    ckpt_h = torch.load(HDG_MODEL_PATH, map_location=device, weights_only=False)
    model_h = TCNAttentionHeadingModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_h.load_state_dict(ckpt_h["model_state_dict"])
    model_h.to(device)
    model_h.eval()

    def get_inferences(trip):
        calibrator = MountCalibrator(window_size=100)
        for g in trip.gnss_samples: calibrator.observe_gnss(g)
        calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
        acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
        gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
        feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
        N = len(feats)
        window_size = 100
        windows = []

        # Standardize for velocity model
        norm_mean_v = ckpt_v.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
        norm_std_v  = ckpt_v.get("norm_std",  np.ones((8, 1),  dtype=np.float32))

        # Standardize for heading model
        norm_mean_h = ckpt_h.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
        norm_std_h  = ckpt_h.get("norm_std",  np.ones((8, 1),  dtype=np.float32))

        windows_v = []
        windows_h = []
        for i in range(N):
            if i < window_size:
                pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
                w   = np.vstack([pad, feats[:i+1]]).T
            else:
                w   = feats[i - window_size + 1 : i + 1].T
            windows_v.append((w - norm_mean_v) / norm_std_v)
            windows_h.append((w - norm_mean_h) / norm_std_h)

        preds_v = []
        preds_w = []
        preds_var = []

        with torch.no_grad():
            for b in range(0, N, 2048):
                xv = torch.from_numpy(np.array(windows_v[b : b + 2048], dtype=np.float32)).to(device)
                pv, _ = model_v(xv)
                preds_v.extend(pv.cpu().numpy().flatten())

                xh = torch.from_numpy(np.squeeze(np.array(windows_h[b : b + 2048], dtype=np.float32), axis=1) if len(np.array(windows_h[b : b + 2048]).shape) == 4 else np.array(windows_h[b : b + 2048], dtype=np.float32)).to(device)
                pw, pvar = model_h(xh)
                preds_w.extend(pw.cpu().numpy().flatten())
                preds_var.extend(torch.exp(pvar).cpu().numpy().flatten())

        return calib_samples, np.array(preds_v, dtype=np.float32), np.array(preds_w, dtype=np.float32), np.array(preds_var, dtype=np.float32)

    calib_s1, v_s1, w_s1, var_s1 = get_inferences(trip_s1)
    calib_s2, v_s2, w_s2, var_s2 = get_inferences(trip_s2)

    sweep_df = pd.read_csv(SWEEP_CSV)

    def run_sweep(enable_zaru=True, enable_centripetal=True, enable_ai_w=True):
        records = []
        for idx, row in sweep_df.iterrows():
            trip_name = str(row["trip"])
            t_start   = float(row["start_time_s"])
            duration  = float(row["duration_s"])
            dist_m    = float(row["distance_m"])

            if "S-S1" in trip_name:
                trip = trip_s1; calib = calib_s1; v_preds = v_s1; w_preds = w_s1; var_preds = var_s1
            else:
                trip = trip_s2; calib = calib_s2; v_preds = v_s2; w_preds = w_s2; var_preds = var_s2

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

            # Sliding window gyro buffer for ZARU straight detection
            w_buf = []

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
                w_ai  = float(w_preds[j])
                v_var = float(var_preds[j])

                w_z_meas = float(cal.gyro_vehicle[2])
                a_y_meas = float(cal.accel_vehicle[1])

                w_buf.append(w_z_meas)
                if len(w_buf) > 20: w_buf.pop(0)

                w_var = np.var(w_buf) if len(w_buf) >= 10 else 1.0

                # 1. ZARU Straight Motion Lock
                if enable_zaru and w_var < 0.0005 and abs(a_y_meas) < 0.3 and v_fwd > 3.0:
                    y_zaru = 0.0 - (w_z_meas - ekf._bg[2])
                    H_z = np.zeros((1, 15), dtype=np.float64)
                    H_z[0, 14] = 1.0
                    R_z = np.array([[(0.005)**2]])
                    S_z = H_z @ ekf._P @ H_z.T + R_z
                    K_z = ekf._P @ H_z.T @ np.linalg.inv(S_z)
                    dx = (K_z @ np.array([y_zaru])).flatten()
                    ekf._bg += dx[12:15]
                    ekf._bg = np.clip(ekf._bg, -ekf.max_bg, ekf.max_bg)

                # 2. Kinematic Centripetal Coupling (w_kin = a_y / v_fwd)
                if enable_centripetal and v_fwd > 5.0 and abs(a_y_meas) > 0.5:
                    w_kin = a_y_meas / v_fwd
                    y_kin = w_kin - (w_z_meas - ekf._bg[2])
                    H_k = np.zeros((1, 15), dtype=np.float64)
                    H_k[0, 14] = 1.0
                    R_k = np.array([[(0.05)**2]])
                    S_k = H_k @ ekf._P @ H_k.T + R_k
                    K_k = ekf._P @ H_k.T @ np.linalg.inv(S_k)
                    dx = (K_k @ np.array([y_kin])).flatten()
                    ekf._bg += dx[12:15]
                    ekf._bg = np.clip(ekf._bg, -ekf.max_bg, ekf.max_bg)

                # 3. AI Neural Yaw-Rate Fusion
                if enable_ai_w and v_fwd > 2.0:
                    y_ai = w_ai - (w_z_meas - ekf._bg[2])
                    H_ai = np.zeros((1, 15), dtype=np.float64)
                    H_ai[0, 14] = 1.0
                    R_ai = np.array([[max(v_var, 0.001)**2]])
                    S_ai = H_ai @ ekf._P @ H_ai.T + R_ai
                    K_ai = ekf._P @ H_ai.T @ np.linalg.inv(S_ai)
                    dx = (K_ai @ np.array([y_ai])).flatten()
                    ekf._bg += dx[12:15]
                    ekf._bg = np.clip(ekf._bg, -ekf.max_bg, ekf.max_bg)

                m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
                vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)
                fused = ekf.predict(cal, vel)

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

    # Execute combinations
    res_base = run_sweep(enable_zaru=False, enable_centripetal=False, enable_ai_w=False)
    res_zaru = run_sweep(enable_zaru=True,  enable_centripetal=False, enable_ai_w=False)
    res_full = run_sweep(enable_zaru=True,  enable_centripetal=True,  enable_ai_w=True)

    print("\n==========================================================================")
    print("        NON-MAP HEADING ENGINE 50-SCENARIO BENCHMARK RESULTS              ")
    print("==========================================================================")
    print(f"1. EKF Baseline (Physical 3D NHC):")
    print(f"   - Combined Median Drift %:        {res_base['median_drift_pct']:.2f}%")
    print(f"   - Long Outages (>500m) Median %:  {res_base['long_median_drift_pct']:.2f}%")
    print(f"   - Worst-Case Drift %:            {res_base['worst_drift_pct']:.2f}%")
    print(f"   - Median Position Error:          {res_base['median_err_m']:.2f} m")

    print(f"\n2. + ZARU Straight Motion Lock:")
    print(f"   - Combined Median Drift %:        {res_zaru['median_drift_pct']:.2f}%")
    print(f"   - Long Outages (>500m) Median %:  {res_zaru['long_median_drift_pct']:.2f}%")
    print(f"   - Worst-Case Drift %:            {res_zaru['worst_drift_pct']:.2f}%")
    print(f"   - Median Position Error:          {res_zaru['median_err_m']:.2f} m")

    print(f"\n3. + Full Non-Map Heading Engine (ZARU + Centripetal + AI Heading Model):")
    print(f"   - Combined Median Drift %:        {res_full['median_drift_pct']:.2f}%")
    print(f"   - Long Outages (>500m) Median %:  {res_full['long_median_drift_pct']:.2f}%")
    print(f"   - Worst-Case Drift %:            {res_full['worst_drift_pct']:.2f}%")
    print(f"   - Median Position Error:          {res_full['median_err_m']:.2f} m")
    print("==========================================================================")

if __name__ == "__main__":
    main()
