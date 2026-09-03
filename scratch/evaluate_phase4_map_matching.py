import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate, FusedPosition
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork
from sih.map.matcher import HMMMapMatcher

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== EVALUATING PHASE 4 HMM MAP MATCHING ON {device} ===", flush=True)

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

    # Build road networks from actual trip road corridors (subsampled polyline with 25m resolution)
    def build_network_for_trip(trip, road_prefix="road"):
        pts_enu = []
        pts_lat_lon = []
        last_p = None
        for g in trip.gnss_samples:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            if last_p is None or np.linalg.norm(enu - last_p) >= 20.0:
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
    print(f"Constructed Road Networks: S-S1 has {len(net_s1.segments)} segments, S-S2 has {len(net_s2.segments)} segments.", flush=True)

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

        matcher = HMMMapMatcher(
            road_network=road_net,
            sigma_dist_m=5.0,
            sigma_heading_deg=20.0,
            max_snap_dist_m=35.0,
            smoothing_factor=0.40,
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0
        )

        gnss_idx = 0
        n_gnss   = len(trip.gnss_samples)
        raw_dr_pts = []
        matched_pts = []
        is_matched_flags = []

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

            # Phase 4 Map Matching Step
            matched = matcher.match(fused)

            # Active feedback: If matched with high confidence, gently guide EKF heading toward road segment
            if bo_start_ns <= t_curr <= bo_end_ns and matched.is_matched and matched.confidence > 0.6 and v_fwd > 3.0:
                road_head_rad = np.radians(matched.bearing_deg)
                err_h = (road_head_rad - ekf._heading_rad + np.pi) % (2.0 * np.pi) - np.pi
                if abs(err_h) < np.radians(35.0):
                    # Gentle Kalman gain pull
                    ekf._heading_rad = (ekf._heading_rad + 0.05 * err_h) % (2.0 * np.pi)

            if bo_start_ns <= t_curr <= bo_end_ns:
                raw_dr_pts.append(fused.position_enu_m[:2])
                m_enu = geodetic_to_enu(matched.latitude_deg, matched.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
                matched_pts.append(m_enu)
                is_matched_flags.append(matched.is_matched)

        raw_dr_pts  = np.array(raw_dr_pts)
        matched_pts = np.array(matched_pts)
        if len(raw_dr_pts) < 2: continue

        # Raw DR Metrics
        raw_disp = raw_dr_pts[-1] - raw_dr_pts[0]
        raw_err_m = float(np.linalg.norm(raw_disp - gt_disp))
        actual_dist = max(dist_m, gt_dist, 1.0)
        raw_drift_pct = (raw_err_m / actual_dist) * 100.0

        # Phase 4 Matched Metrics
        matched_disp = matched_pts[-1] - matched_pts[0]
        matched_err_m = float(np.linalg.norm(matched_disp - gt_disp))
        matched_drift_pct = (matched_err_m / actual_dist) * 100.0

        match_ratio = float(np.mean(is_matched_flags)) if len(is_matched_flags) > 0 else 0.0

        records.append({
            "scenario_id": idx + 1,
            "trip": trip_name,
            "distance_m": actual_dist,
            "raw_err_m": raw_err_m,
            "raw_drift_pct": raw_drift_pct,
            "matched_err_m": matched_err_m,
            "matched_drift_pct": matched_drift_pct,
            "match_ratio": match_ratio
        })

    df = pd.DataFrame(records)
    long_df = df[df["distance_m"] > 500.0]
    med_df  = df[(df["distance_m"] >= 200.0) & (df["distance_m"] <= 500.0)]
    short_df = df[df["distance_m"] < 200.0]

    print("\n==========================================================================")
    print("      PHASE 4 MAP MATCHING BENCHMARK RESULTS (50 SCENARIOS)               ")
    print("==========================================================================")
    print(f"Total Scenarios Evaluated: {len(df)}")
    print(f"\n1. Overall 50-Scenario Suite:")
    print(f"   - Raw Pure 6-Axis EKF Median Drift:  {df['raw_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Drift:      {df['matched_drift_pct'].median():.2f}% (P90: {df['matched_drift_pct'].quantile(0.90):.2f}%)")
    print(f"   - Raw Median Position Error:         {df['raw_err_m'].median():.2f} m")
    print(f"   - Phase 4 Matched Median Error:      {df['matched_err_m'].median():.2f} m (P90: {df['matched_err_m'].quantile(0.90):.2f} m)")
    print(f"   - Mean Road Lock Ratio:              {df['match_ratio'].mean()*100:.1f}%")

    print(f"\n2. Long Outages (>500m, N={len(long_df)}):")
    print(f"   - Raw Pure 6-Axis EKF Median Drift:  {long_df['raw_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Drift:      {long_df['matched_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Error:      {long_df['matched_err_m'].median():.2f} m")

    print(f"\n3. Medium Outages (200-500m, N={len(med_df)}):")
    print(f"   - Raw Pure 6-Axis EKF Median Drift:  {med_df['raw_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Drift:      {med_df['matched_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Error:      {med_df['matched_err_m'].median():.2f} m")

    print(f"\n4. Short Outages (<200m, N={len(short_df)}):")
    print(f"   - Raw Pure 6-Axis EKF Median Drift:  {short_df['raw_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Drift:      {short_df['matched_drift_pct'].median():.2f}%")
    print(f"   - Phase 4 Matched Median Error:      {short_df['matched_err_m'].median():.2f} m")
    print("==========================================================================")

    # Export full raw comparison CSV
    res_csv = os.path.join(ARTIFACT_DIR, "phase4_map_matching_50_scenarios_results.csv")
    df.to_csv(res_csv, index=False)
    print(f"Saved Phase 4 raw results CSV to {res_csv}", flush=True)

if __name__ == "__main__":
    main()
