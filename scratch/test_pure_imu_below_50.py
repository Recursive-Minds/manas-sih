import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

SWEEP_CSV    = r"C:\Users\carpe\SIH\artifacts\randomized_blackout_sweep_results.csv"
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PURE 6-AXIS IMU DRIFT REDUCTION EVALUATION ({device}) ===", flush=True)

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

    def evaluate_clean_ekf():
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
                max_gyro_bias_rad_s=np.radians(0.1),
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

                if bo_start_ns <= t_curr <= bo_end_ns:
                    est_pts.append(fused.position_enu_m[:2])

            est_pts = np.array(est_pts)
            if len(est_pts) < 2: continue
            est_disp = est_pts[-1] - est_pts[0]
            final_err_m = float(np.linalg.norm(est_disp - gt_disp))
            drift_pct   = (final_err_m / max(dist_m, 1.0)) * 100.0

            records.append({"trip": trip_name, "distance_m": dist_m, "error_m": final_err_m, "drift_pct": drift_pct})

        df = pd.DataFrame(records)
        long_df  = df[df["distance_m"] > 500.0]
        med_df   = df[(df["distance_m"] >= 200.0) & (df["distance_m"] <= 500.0)]
        short_df = df[df["distance_m"] < 200.0]

        return {
            "overall_median_drift": float(df["drift_pct"].median()),
            "long_median_drift": float(long_df["drift_pct"].median()) if len(long_df) > 0 else 0.0,
            "medium_median_drift": float(med_df["drift_pct"].median()) if len(med_df) > 0 else 0.0,
            "short_median_drift": float(short_df["drift_pct"].median()) if len(short_df) > 0 else 0.0,
            "overall_median_err": float(df["error_m"].median()),
            "long_median_err": float(long_df["error_m"].median()) if len(long_df) > 0 else 0.0,
        }

    res = evaluate_clean_ekf()
    print("\n==========================================================================")
    print("      PURE 6-AXIS IMU CLEAN BENCHMARK EVALUATION (NO CHEATING/MAPS)       ")
    print("==========================================================================")
    print(f"Overall 50-Scenario Median Drift %:    {res['overall_median_drift']:.2f}%")
    print(f"Long Outages (>500m) Median Drift %:    {res['long_median_drift']:.2f}%")
    print(f"Medium Outages (200-500m) Median Drift %:{res['medium_median_drift']:.2f}%")
    print(f"Short Outages (<200m) Median Drift %:   {res['short_median_drift']:.2f}%")
    print(f"Overall Median Position Error (m):     {res['overall_median_err']:.2f} m")
    print(f"Long Outages Median Position Error (m): {res['long_median_err']:.2f} m")
    print("==========================================================================")

if __name__ == "__main__":
    main()
