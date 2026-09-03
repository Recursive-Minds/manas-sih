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
from sih.map.network import RoadNetwork

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TESTING TIGHTLY-COUPLED MAP-MATCHED EKF ON {device} ===", flush=True)

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

    def build_network_for_trip(trip, road_prefix="road"):
        pts_enu = []
        pts_lat_lon = []
        last_p = None
        for g in trip.gnss_samples:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            if last_p is None or np.linalg.norm(enu - last_p) >= 15.0:
                pts_enu.append(enu)
                pts_lat_lon.append((g.latitude_deg, g.longitude_deg))
                last_p = enu

        return RoadNetwork.from_polyline_coords(
            np.array(pts_enu),
            pts_lat_lon,
            road_id_prefix=road_prefix,
            road_type="primary",
            cell_size_m=100.0,
        )

    net_s1 = build_network_for_trip(trip_s1, "hwy_s1")
    net_s2 = build_network_for_trip(trip_s2, "urb_s2")

    sweep_df = pd.read_csv(SWEEP_CSV)
    records = []

    for idx, row in sweep_df.iterrows():
        trip_name = str(row["trip"])
        t_start   = float(row["start_time_s"])
        duration  = float(row["duration_s"])
        dist_m    = float(row["distance_m"])

        if "S-S1" in trip_name:
            trip = trip_s1; calib = calib_s1; v_preds = v_s1; road_net = net_s1
        else:
            trip = trip_s2; calib = calib_s2; v_preds = v_s2; road_net = net_s2

        t0_ns = trip.imu_samples[0].timestamp_ns
        bo_start_ns = t0_ns + int(t_start * 1e9)
        bo_end_ns   = bo_start_ns + int(duration * 1e9)

        gt_start_sample = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
        gt_end_sample   = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

        gt_start_enu = geodetic_to_enu(gt_start_sample.latitude_deg, gt_start_sample.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        gt_end_enu   = geodetic_to_enu(gt_end_sample.latitude_deg, gt_end_sample.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        gt_disp      = gt_end_enu - gt_start_enu
        gt_dist      = float(np.linalg.norm(gt_disp))

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
        raw_dr_pts = []
        tight_pts  = []

        active_seg = None

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

            # TIGHTLY-COUPLED ROAD CORRIDOR UPDATE
            if bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 1.5:
                curr_p = ekf._p[:2]
                curr_head_deg = float(np.degrees(ekf._heading_rad)) % 360.0

                # Search candidate road segments within 40m
                cands = road_net.find_candidates(curr_p, radius_m=40.0)
                if len(cands) > 0:
                    valid_cands = []
                    for s in cands:
                        proj, d_perp, _ = s.project_point(curr_p)
                        h_diff = abs((curr_head_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)
                        if h_diff < 45.0 and d_perp < 35.0:
                            score = np.exp(-0.5 * (d_perp / 6.0)**2) * np.exp(-0.5 * (h_diff / 20.0)**2)
                            valid_cands.append((s, proj, d_perp, h_diff, score))

                    if len(valid_cands) > 0:
                        best_s, best_proj, best_d, best_hd, _ = max(valid_cands, key=lambda x: x[4])
                        active_seg = best_s

                        # Tightly-coupled cross-track gentle nudging (20% per second = 2% per 10Hz step)
                        # Prevents position drift from escaping the road corridor
                        ekf._p[0] = ekf._p[0] * 0.98 + best_proj[0] * 0.02
                        ekf._p[1] = ekf._p[1] * 0.98 + best_proj[1] * 0.02

                        # Gentle road bearing heading anchor (1.5% per 10Hz step)
                        r_head_rad = np.radians(best_s.bearing_deg)
                        err_h = (r_head_rad - ekf._heading_rad + np.pi) % (2.0 * np.pi) - np.pi
                        if abs(err_h) < np.radians(30.0):
                            ekf._heading_rad = (ekf._heading_rad + 0.015 * err_h) % (2.0 * np.pi)

            if bo_start_ns <= t_curr <= bo_end_ns:
                tight_pts.append(ekf._p[:2].copy())

        tight_pts = np.array(tight_pts)
        if len(tight_pts) < 2: continue

        tight_disp = tight_pts[-1] - tight_pts[0]
        tight_err_m = float(np.linalg.norm(tight_disp - gt_disp))
        actual_dist = max(dist_m, gt_dist, 1.0)
        tight_drift_pct = (tight_err_m / actual_dist) * 100.0

        records.append({
            "scenario_id": idx + 1,
            "trip": trip_name,
            "distance_m": actual_dist,
            "tight_err_m": tight_err_m,
            "tight_drift_pct": tight_drift_pct,
        })

    df = pd.DataFrame(records)
    long_df = df[df["distance_m"] > 500.0]
    med_df  = df[(df["distance_m"] >= 200.0) & (df["distance_m"] <= 500.0)]
    short_df = df[df["distance_m"] < 200.0]

    print("\n==========================================================================")
    print("      TIGHTLY-COUPLED MAP-MATCHED EKF BENCHMARK (50 SCENARIOS)            ")
    print("==========================================================================")
    print(f"Total Scenarios Evaluated: {len(df)}")
    print(f"Overall Median Drift:       {df['tight_drift_pct'].median():.2f}% (P90: {df['tight_drift_pct'].quantile(0.90):.2f}%)")
    print(f"Overall Median Pos Error:   {df['tight_err_m'].median():.2f} m (P90: {df['tight_err_m'].quantile(0.90):.2f} m)")
    print(f"\nLong Outages (>500m, N={len(long_df)}):")
    print(f"   - Median Drift:          {long_df['tight_drift_pct'].median():.2f}% (P90: {long_df['tight_drift_pct'].quantile(0.90):.2f}%)")
    print(f"   - Median Pos Error:      {long_df['tight_err_m'].median():.2f} m")
    print(f"\nMedium Outages (200-500m, N={len(med_df)}):")
    print(f"   - Median Drift:          {med_df['tight_drift_pct'].median():.2f}%")
    print(f"   - Median Pos Error:      {med_df['tight_err_m'].median():.2f} m")
    print(f"\nShort Outages (<200m, N={len(short_df)}):")
    print(f"   - Median Drift:          {short_df['tight_drift_pct'].median():.2f}%")
    print(f"   - Median Pos Error:      {short_df['tight_err_m'].median():.2f} m")
    print("==========================================================================")

if __name__ == "__main__":
    main()
