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
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample, FusedPosition
from sih.data.geo import geodetic_to_enu, enu_to_geodetic

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_PATH   = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

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

    def eval_trip(trip, calib, v_preds, sub_df, offset_idx=0):
        results = []
        for idx, (_, row) in enumerate(sub_df.iterrows()):
            t_start  = float(row["start_time_s"])
            duration = float(row["duration_s"])

            ekf = ErrorStateEKF(enable_nhc=True, enable_zupt=True)
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
                    in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
                    if not in_bo:
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
                    g_enu  = geodetic_to_enu(
                        g_curr.latitude_deg, g_curr.longitude_deg, 0.0,
                        trip.reference_lat_deg, trip.reference_lon_deg, 0.0
                    )
                    gt_pts.append(g_enu[:2])

            est_pts = np.array(est_pts)
            gt_pts  = np.array(gt_pts)

            if len(gt_pts) < 2: continue

            dists    = np.sqrt(np.sum(np.diff(gt_pts, axis=0)**2, axis=1))
            tot_dist = max(float(np.sum(dists)), 1.0)

            final_err = float(np.linalg.norm(est_pts[-1] - gt_pts[-1]))
            drift_pct = (final_err / tot_dist) * 100.0

            cur_num = offset_idx + idx + 1
            pct = (cur_num / 50.0) * 100.0
            print(f"  [{cur_num:2d}/50] ({pct:3.0f}%) {row['scenario'][:32]:32s} old={row['drift_pct']:7.2f}% ({row['final_error_m']:6.1f}m) -> new={drift_pct:7.2f}% ({final_err:6.1f}m)", flush=True)

            results.append({
                "scenario": row["scenario"], "trip": row["trip"],
                "old_error_m": row["final_error_m"], "old_drift_pct": row["drift_pct"],
                "new_error_m": final_err, "new_drift_pct": drift_pct
            })
        return results

    sc1_df = sweep_df[sweep_df["trip"].str.contains("S-S1")]
    sc2_df = sweep_df[sweep_df["trip"].str.contains("S-S2")]

    print("\nRunning Production ErrorStateEKF Baseline Sweep...", flush=True)
    t0 = time.time()
    r1 = eval_trip(trip_s1, calib_s1, v_s1, sc1_df, offset_idx=0)
    r2 = eval_trip(trip_s2, calib_s2, v_s2, sc2_df, offset_idx=len(r1))
    df_res = pd.DataFrame(r1 + r2)

    med   = df_res["new_drift_pct"].median()
    worst = df_res["new_drift_pct"].max()

    print(f"\nCompleted in {time.time()-t0:.2f}s", flush=True)
    print(f"Original Baseline Target:  Median < 73.54%,  Worst-Case < 806.03%", flush=True)
    print(f"Production ErrorStateEKF:  Median = {med:.2f}%,   Worst-Case = {worst:.2f}%", flush=True)

if __name__ == "__main__":
    main()
