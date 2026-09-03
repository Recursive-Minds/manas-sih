"""
Diagnostic script for inspecting initial heading offset at exact blackout entry timestamp.
Compares:
1. EKF prior heading before any seeding
2. Seeded heading (current seed_pre_blackout_heading)
3. Last GNSS fix bearing
4. True ground truth bearing at exact bo_start_ns (interpolated from ground truth trajectory)
5. Gyro-extrapolated heading from last GNSS fix to bo_start_ns
For all 35 scenarios of S-M.csv, highlighting #20, #31, #18, #14.
"""

import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork

DATA_PATH = r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv"
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def build_road_network(trip, prefix="road", min_step_m=15.0):
    pts_enu = []
    pts_lat_lon = []
    last_p = None
    for g in trip.gnss_samples:
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        if last_p is None or np.linalg.norm(enu - last_p) >= min_step_m:
            pts_enu.append(enu)
            pts_lat_lon.append((g.latitude_deg, g.longitude_deg))
            last_p = enu

    return RoadNetwork.from_polyline_coords(
        np.array(pts_enu),
        pts_lat_lon,
        road_id_prefix=prefix,
        road_type="primary",
        cell_size_m=100.0
    ), np.array(pts_enu)

def compute_gt_bearings(valid_gnss, ref_lat, ref_lon):
    times_s = np.array([(g.timestamp_ns - valid_gnss[0].timestamp_ns) * 1e-9 for g in valid_gnss])
    enus = np.array([geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, ref_lat, ref_lon, 0.0)[:2] for g in valid_gnss])
    
    # Cubic spline of East and North vs time
    cs_e = CubicSpline(times_s, enus[:, 0])
    cs_n = CubicSpline(times_s, enus[:, 1])
    
    return times_s, enus, cs_e, cs_n, valid_gnss[0].timestamp_ns

def main():
    print("Loading data for heading offset diagnosis...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = GenericDataLoader()
    trip = loader.load_file(DATA_PATH)

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    calibrator = MountCalibrator(window_size=100)
    if "mount_pitch_deg" in trip.metadata and "mount_roll_deg" in trip.metadata:
        calibrator.calibrate_from_mount_angles(trip.metadata["mount_pitch_deg"], trip.metadata["mount_roll_deg"])
    else:
        for g in trip.gnss_samples: calibrator.observe_gnss(g)

    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    N = len(feats)
    norm_mean = ckpt_v.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std  = ckpt_v.get("norm_std",  np.ones((8, 1),  dtype=np.float32))

    windows = []
    window_size = 100
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
    v_preds = np.array(preds_v, dtype=np.float32)

    road_net, all_road_pts = build_road_network(trip, prefix="sm_road", min_step_m=20.0)

    results_csv = r"C:\Users\carpe\SIH\artifacts\phase4_unseen_sm_benchmark_results.csv"
    df = pd.read_csv(results_csv)

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    gt_times_s, gt_enus, cs_e, cs_n, t_gnss0_ns = compute_gt_bearings(valid_gnss, trip.reference_lat_deg, trip.reference_lon_deg)

    t0_ns = trip.imu_samples[0].timestamp_ns
    records = []

    for idx, row in df.iterrows():
        sc_id = int(row["scenario_id"])
        bo_start_s = float(row["start_time_s"])
        duration_s = float(row["duration_s"])
        dist_m = float(row["distance_m"])

        bo_start_ns = t0_ns + int(bo_start_s * 1e9)
        bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

        # True GT velocity at exact bo_start_ns
        t_bo_rel_s = (bo_start_ns - t_gnss0_ns) * 1e-9
        v_e_gt = float(cs_e.derivative(1)(t_bo_rel_s))
        v_n_gt = float(cs_n.derivative(1)(t_bo_rel_s))
        true_bearing_deg = float(np.degrees(np.arctan2(v_e_gt, v_n_gt))) % 360.0

        # EKF run up to bo_start_ns
        ekf = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.5))
        warmup_start_ns = bo_start_ns - int(30 * 1e9)
        warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns))
        ekf.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)

        n_gnss = len(trip.gnss_samples)
        gnss_idx = 0
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
            gnss_idx += 1

        last_gnss_fix = None
        for j, imu in enumerate(trip.imu_samples):
            t_curr = imu.timestamp_ns
            if t_curr < warmup_start_ns: continue
            if t_curr > bo_start_ns: break

            while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                g = trip.gnss_samples[gnss_idx]
                if g.timestamp_ns <= bo_start_ns:
                    ekf.update_gnss(g)
                    last_gnss_fix = g
                gnss_idx += 1

            cal = calib_samples[j]
            v_fwd = float(v_preds[j])
            m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
            vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)
            ekf.predict(cal, vel)

        # Prior heading in EKF right at bo_start_ns (before reset)
        hdg_ekf_prior_deg = float(np.degrees(ekf._heading_rad)) % 360.0

        # GNSS staleness
        gnss_staleness_s = (bo_start_ns - last_gnss_fix.timestamp_ns) * 1e-9 if last_gnss_fix else 0.0
        last_gnss_bearing_deg = last_gnss_fix.bearing_deg if last_gnss_fix and last_gnss_fix.bearing_deg is not None else np.nan

        # Gyro extrapolation from last GNSS fix to bo_start_ns
        accum_dtheta_deg = 0.0
        for j, imu in enumerate(trip.imu_samples):
            if last_gnss_fix and last_gnss_fix.timestamp_ns < imu.timestamp_ns <= bo_start_ns:
                dt = (imu.timestamp_ns - trip.imu_samples[j-1].timestamp_ns) * 1e-9
                accum_dtheta_deg += np.degrees(calib_samples[j].gyro_vehicle[2] * dt)

        extrapolated_hdg_deg = (last_gnss_bearing_deg + accum_dtheta_deg) % 360.0 if not np.isnan(last_gnss_bearing_deg) else np.nan

        # Instantaneous GNSS bearing with gyro forward extrapolation (NO forced road snapping)
        pre_gnss = [g for g in valid_gnss if g.timestamp_ns <= bo_start_ns]
        last_g = pre_gnss[-1] if pre_gnss else None
        
        # Calculate gyro turning from last_g to bo_start_ns
        delta_gyro = 0.0
        if last_g:
            for k_imu in range(len(trip.imu_samples)):
                t_k = trip.imu_samples[k_imu].timestamp_ns
                if last_g.timestamp_ns < t_k <= bo_start_ns:
                    dt_k = (t_k - trip.imu_samples[k_imu-1].timestamp_ns) * 1e-9
                    delta_gyro += np.degrees(calib_samples[k_imu].gyro_vehicle[2] * dt_k)

        seeded_hdg_deg = ekf.seed_pre_blackout_heading(pre_gnss[-3:], road_bearing_deg=None, delta_heading_gyro_deg=delta_gyro)

        def diff_angle(a, b):
            return (a - b + 180.0) % 360.0 - 180.0

        offset_prior   = diff_angle(hdg_ekf_prior_deg, true_bearing_deg)
        offset_seeded  = diff_angle(seeded_hdg_deg, true_bearing_deg)
        offset_last_g  = diff_angle(last_gnss_bearing_deg, true_bearing_deg)
        offset_extrap  = diff_angle(extrapolated_hdg_deg, true_bearing_deg)

        records.append({
            "scenario_id": sc_id,
            "dist_m": dist_m,
            "dur_s": duration_s,
            "staleness_s": gnss_staleness_s,
            "true_bearing_deg": true_bearing_deg,
            "ekf_prior_deg": hdg_ekf_prior_deg,
            "seeded_deg": seeded_hdg_deg,
            "last_gnss_deg": last_gnss_bearing_deg,
            "extrap_deg": extrapolated_hdg_deg,
            "road_bearing_cand": np.nan,
            "offset_prior": offset_prior,
            "offset_seeded": offset_seeded,
            "offset_last_g": offset_last_g,
            "offset_extrap": offset_extrap,
        })

    diag_df = pd.DataFrame(records)
    print("\n" + "="*90)
    print("INITIAL HEADING OFFSET DIAGNOSTIC REPORT (UNSEEN S-M TRIP)")
    print("="*90)
    
    target_ids = [20, 31, 18, 14, 2, 30]
    print("\nTARGET FOCUS SCENARIOS (#20, #31, #18, #14, and #2, #30):")
    print("-" * 110)
    header = f"{'ID':>3} | {'Dist(m)':>7} | {'Stale(s)':>8} | {'True(deg)':>9} | {'Prior Off':>10} | {'Seed Off':>10} | {'LastG Off':>10} | {'Extrap Off':>10} | {'Road Cand':>9}"
    print(header)
    print("-" * 110)
    for tid in target_ids:
        r = diag_df[diag_df["scenario_id"] == tid].iloc[0]
        rcand_str = f"{r['road_bearing_cand']:.1f}" if r['road_bearing_cand'] is not None else "None"
        print(f"{int(r['scenario_id']):3d} | {r['dist_m']:7.1f} | {r['staleness_s']:8.3f} | {r['true_bearing_deg']:9.2f} | {r['offset_prior']:+10.2f}° | {r['offset_seeded']:+10.2f}° | {r['offset_last_g']:+10.2f}° | {r['offset_extrap']:+10.2f}° | {rcand_str:>9}")

    print("-" * 110)
    print("\nSTATISTICAL SUMMARY ACROSS ALL 35 SCENARIOS:")
    print("-" * 65)
    print(f"GNSS Staleness at Blackout Entry: Mean={diag_df['staleness_s'].mean():.3f}s, Max={diag_df['staleness_s'].max():.3f}s")
    print(f"Offset (EKF Prior vs True GT):    Mean={diag_df['offset_prior'].mean():+.2f}°, Std={diag_df['offset_prior'].std():.2f}°, AbsMean={diag_df['offset_prior'].abs().mean():.2f}°")
    print(f"Offset (Seeded vs True GT):       Mean={diag_df['offset_seeded'].mean():+.2f}°, Std={diag_df['offset_seeded'].std():.2f}°, AbsMean={diag_df['offset_seeded'].abs().mean():.2f}°")
    print(f"Offset (Last GNSS vs True GT):    Mean={diag_df['offset_last_g'].mean():+.2f}°, Std={diag_df['offset_last_g'].std():.2f}°, AbsMean={diag_df['offset_last_g'].abs().mean():.2f}°")
    print(f"Offset (Extrapolated vs True GT): Mean={diag_df['offset_extrap'].mean():+.2f}°, Std={diag_df['offset_extrap'].std():.2f}°, AbsMean={diag_df['offset_extrap'].abs().mean():.2f}°")
    print("-" * 65)

    diag_df.to_csv(r"C:\Users\carpe\SIH\artifacts\initial_heading_offset_diagnosis.csv", index=False)
    print("Saved diagnosis to artifacts/initial_heading_offset_diagnosis.csv", flush=True)

if __name__ == "__main__":
    main()
