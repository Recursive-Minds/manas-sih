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
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
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

def run_scenario(trip, calib_samples, v_preds, road_net, g_entry, duration_s):
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = g_entry.timestamp_ns
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    gt_end_sample = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

    gt_start_enu = geodetic_to_enu(g_entry.latitude_deg, g_entry.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_end_enu   = geodetic_to_enu(gt_end_sample.latitude_deg, gt_end_sample.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_disp      = gt_end_enu - gt_start_enu
    gt_dist      = float(np.linalg.norm(gt_disp))

    if gt_dist < 10.0:
        return None

    warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
    warmup_gnss = min([g for g in valid_gnss if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])

    # Run Pure 6-Axis EKF (unified global coordinates)
    ekf_pure = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_pure.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    # Run Phase 4 Map-Matched EKF (unified global coordinates)
    ekf_map = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_map.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    n_gnss = len(trip.gnss_samples)
    gnss_idx = 0
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
        gnss_idx += 1

    pure_pts = []
    map_pts  = []

    blackout_started = False

    for j, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr < warmup_start_ns: continue
        if t_curr > bo_end_ns + int(1e9): break

        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            if g.timestamp_ns <= bo_start_ns:
                ekf_pure.update_gnss(g)
                ekf_map.update_gnss(g)
                if g.speed_mps is not None and g.speed_mps > 3.0 and g.bearing_deg is not None:
                    b_rad = float(np.radians(g.bearing_deg))
                    ekf_pure._heading_rad = b_rad
                    ekf_map._heading_rad = b_rad
            gnss_idx += 1

        cal = calib_samples[j]
        v_fwd = float(v_preds[j])

        m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)

        # At the exact blackout boundary transition, seed starting point and multi-epoch heading
        if not blackout_started and t_curr >= bo_start_ns:
            blackout_started = True
            ekf_pure._p[0] = gt_start_enu[0]
            ekf_pure._p[1] = gt_start_enu[1]
            ekf_map._p[0]  = gt_start_enu[0]
            ekf_map._p[1]  = gt_start_enu[1]
            
            # Robust multi-epoch heading seeding over pre-blackout window
            pre_gnss_window = [g for g in valid_gnss if bo_start_ns - int(25.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
            
            # Gyro turning extrapolation from last GNSS fix to blackout start
            delta_gyro_deg = 0.0
            if pre_gnss_window:
                last_g_ts = pre_gnss_window[-1].timestamp_ns
                for k_imu in range(len(trip.imu_samples)):
                    t_k = trip.imu_samples[k_imu].timestamp_ns
                    if last_g_ts < t_k <= bo_start_ns:
                        dt_k = (t_k - trip.imu_samples[k_imu-1].timestamp_ns) * 1e-9
                        delta_gyro_deg += np.degrees(calib_samples[k_imu].gyro_vehicle[2] * dt_k)

            init_road_bearing = None
            init_cands = road_net.find_candidates(gt_start_enu, radius_m=35.0)
            curr_est_hdg = float(np.degrees(ekf_map._heading_rad)) % 360.0
            best_cand = None
            min_cost = 1e9
            for s in init_cands:
                proj, d_p, _ = s.project_point(gt_start_enu)
                b_diff = abs((s.bearing_deg - curr_est_hdg + 180.0) % 360.0 - 180.0)
                if d_p < 25.0 and b_diff < 35.0:
                    cost = d_p + 0.5 * b_diff
                    if cost < min_cost:
                        min_cost = cost
                        best_cand = s
            if best_cand is not None:
                init_road_bearing = best_cand.bearing_deg

            ekf_pure.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing, delta_heading_gyro_deg=delta_gyro_deg)
            ekf_map.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing, delta_heading_gyro_deg=delta_gyro_deg)

        fused_pure = ekf_pure.predict(cal, vel)
        fused_map  = ekf_map.predict(cal, vel)

        # Apply Tightly Coupled Map Constraints on ekf_map during blackout
        if bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 1.0:
            turn_rate_dps = abs(np.degrees(cal.gyro_vehicle[2]))
            is_turning = turn_rate_dps > 2.0

            curr_p = ekf_map._p[:2]
            curr_head_deg = float(np.degrees(ekf_map._heading_rad)) % 360.0

            # EKF heading uncertainty drives soft likelihood weighting
            var_yaw = float(ekf_map._P[8, 8])
            sigma_yaw_deg = float(np.degrees(np.sqrt(max(1e-6, var_yaw))))
            sigma_eff = float(np.sqrt(sigma_yaw_deg**2 + 15.0**2))
            if is_turning:
                sigma_eff = max(sigma_eff, 45.0)

            cands = road_net.find_candidates(curr_p, radius_m=50.0)
            if len(cands) > 0:
                valid_cands = []
                for s in cands:
                    proj, d_perp, frac = s.project_point(curr_p)
                    h_diff = abs((curr_head_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)
                    if d_perp < 35.0:
                        end_factor = 0.05 if frac >= 0.95 else 1.0
                        score = np.exp(-0.5 * (d_perp / 8.0)**2) * np.exp(-0.5 * (h_diff / sigma_eff)**2) * end_factor
                        if score > 1e-4:
                            valid_cands.append((s, proj, d_perp, h_diff, score))
                if len(valid_cands) > 0:
                    valid_cands.sort(key=lambda x: x[4], reverse=True)
                    top_c = valid_cands[:2]

                    # Multi-hypothesis fork resolution: delay hard commit when branches are parallel
                    is_fork = False
                    if len(top_c) >= 2:
                        h_sep = abs((top_c[0][0].bearing_deg - top_c[1][0].bearing_deg + 180.0) % 360.0 - 180.0)
                        if h_sep < 15.0 and top_c[1][4] > 0.40 * top_c[0][4]:
                            is_fork = True

                    if is_fork:
                        w0, w1 = top_c[0][4], top_c[1][4]
                        tot_w = w0 + w1
                        best_s = top_c[0][0]
                        best_proj = (w0 * top_c[0][1] + w1 * top_c[1][1]) / tot_w
                    else:
                        best_s, best_proj, _, _, _ = top_c[0]

                    # Snap position to road centerline
                    ekf_map._p[0] = best_proj[0]
                    ekf_map._p[1] = best_proj[1]

                    # Gentle straight-segment re-anchoring when not actively turning or in a fork
                    if not is_turning and not is_fork:
                        ekf_map.reanchor_heading(best_s.bearing_deg, confidence=0.8)

        if bo_start_ns <= t_curr <= bo_end_ns:
            pure_pts.append(fused_pure.position_enu_m[:2].copy())
            map_pts.append(ekf_map._p[:2].copy())

    pure_pts = np.array(pure_pts)
    map_pts  = np.array(map_pts)

    if len(pure_pts) < 2 or len(map_pts) < 2:
        return None

    # Collect ground truth points
    gt_pts = [gt_start_enu]
    for g in valid_gnss:
        if bo_start_ns < g.timestamp_ns <= bo_end_ns:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_pts.append(enu)
    gt_pts = np.array(gt_pts)

    pure_disp = pure_pts[-1] - pure_pts[0]
    pure_err = float(np.linalg.norm(pure_disp - gt_disp))
    pure_drift = (pure_err / gt_dist) * 100.0

    map_disp = map_pts[-1] - map_pts[0]
    map_err = float(np.linalg.norm(map_disp - gt_disp))
    map_drift = (map_err / gt_dist) * 100.0

    t_start_s = float((bo_start_ns - t0_ns) * 1e-9)

    return {
        "t_start_s": t_start_s,
        "duration_s": duration_s,
        "dist_m": gt_dist,
        "pure_err_m": pure_err,
        "pure_drift_pct": pure_drift,
        "map_err_m": map_err,
        "map_drift_pct": map_drift,
        "gt_pts": gt_pts,
        "pure_pts": pure_pts,
        "map_pts": map_pts,
    }

def plot_single_map(res, all_road_pts, title, out_filename):
    gt_pts = res["gt_pts"]
    pure_pts = res["pure_pts"]
    map_pts  = res["map_pts"]
    pure_drift = res["pure_drift_pct"]
    map_drift  = res["map_drift_pct"]

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Road network (grey)
    ax.plot(all_road_pts[:, 0], all_road_pts[:, 1], color='#cccccc', linewidth=4.0, alpha=0.8, label="Road Network Centerline", zorder=2)

    # Ground Truth (Black)
    if len(gt_pts) > 0:
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], 'k-', linewidth=3.2, label="Ground Truth (GNSS)", zorder=5)
        ax.scatter(gt_pts[0, 0], gt_pts[0, 1], color='black', s=140, edgecolors='white', linewidths=1.5, zorder=7, label="Blackout Entry (Black Dot)")
        ax.scatter(gt_pts[-1, 0], gt_pts[-1, 1], color='black', marker='X', s=160, edgecolors='white', linewidths=1.5, zorder=7, label="Ground Truth Exit (Black X)")

    # Pure 6-Axis EKF (Red)
    ax.plot(pure_pts[:, 0], pure_pts[:, 1], color='#d9534f', linewidth=2.5, linestyle=':', label=f"Pure 6-Axis EKF (Drift: {pure_drift:.1f}%)", zorder=4)
    ax.scatter(pure_pts[-1, 0], pure_pts[-1, 1], color='#d9534f', marker='o', s=120, edgecolors='black', linewidths=1.2, zorder=8, label="Pure Final (Red Dot)")

    # Phase 4 Map-Matched (Blue)
    ax.plot(map_pts[:, 0], map_pts[:, 1], color='#0275d8', linewidth=2.8, linestyle='--', label=f"Phase 4 Map-Matched (Drift: {map_drift:.1f}%)", zorder=6)
    ax.scatter(map_pts[-1, 0], map_pts[-1, 1], color='#0275d8', marker='s', s=130, edgecolors='white', linewidths=1.5, zorder=9, label="Phase 4 Final (Blue Square)")

    ax.set_title(f"{title}\nPure 6-Axis Drift: {pure_drift:.1f}% -> Phase 4 Matched Drift: {map_drift:.1f}%", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("East Position (meters)", fontsize=11)
    ax.set_ylabel("North Position (meters)", fontsize=11)
    ax.legend(loc="best", fontsize=10, frameon=True, facecolor='white', framealpha=0.95)

    # Symmetric zoom around blackout
    all_x = np.concatenate([gt_pts[:, 0], pure_pts[:, 0], map_pts[:, 0]]) if len(gt_pts) > 0 else np.concatenate([pure_pts[:, 0], map_pts[:, 0]])
    all_y = np.concatenate([gt_pts[:, 1], pure_pts[:, 1], map_pts[:, 1]]) if len(gt_pts) > 0 else np.concatenate([pure_pts[:, 1], map_pts[:, 1]])
    cx = float((np.min(all_x) + np.max(all_x)) / 2.0)
    cy = float((np.min(all_y) + np.max(all_y)) / 2.0)
    span = float(max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2.0 + 120.0)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    plt.tight_layout()

    out_path = os.path.join(ARTIFACT_DIR, out_filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_filename}", flush=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== EVALUATING UNSEEN NEW DATASET: S-M.csv ON {device} ===", flush=True)

    loader = GenericDataLoader()
    trip_sm = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv")
    print(f"Loaded Trip S-M.csv: {len(trip_sm.imu_samples)} IMU samples, {len(trip_sm.gnss_samples)} GNSS samples.", flush=True)

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    # Pre-infer velocity on S-M.csv with MountCalibrator
    print("Running AI model inference on unseen S-M.csv...", flush=True)
    calibrator = MountCalibrator(window_size=100)
    if "mount_pitch_deg" in trip_sm.metadata and "mount_roll_deg" in trip_sm.metadata:
        calibrator.calibrate_from_mount_angles(trip_sm.metadata["mount_pitch_deg"], trip_sm.metadata["mount_roll_deg"])
        print(f"Calibrated Mount from metadata: Pitch={trip_sm.metadata['mount_pitch_deg']:.1f}°, Roll={trip_sm.metadata['mount_roll_deg']:.1f}°")
    else:
        for g in trip_sm.gnss_samples: calibrator.observe_gnss(g)

    calib_samples = [calibrator.update(imu) for imu in trip_sm.imu_samples]
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
    v_preds = np.array(preds_v, dtype=np.float32)

    # Build road network for S-M
    road_net_sm, all_road_pts = build_road_network(trip_sm, "sm_road")
    print(f"Road network for S-M built with {len(road_net_sm.segments)} segments.", flush=True)

    # Sample 35 blackout scenarios directly aligned with actual valid moving GNSS fixes
    t0_ns = trip_sm.imu_samples[0].timestamp_ns
    candidate_gnss = [
        g for g in trip_sm.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps > 4.0 and g.bearing_deg is not None
        and (600.0 * 1e9) <= (g.timestamp_ns - t0_ns) <= (9500.0 * 1e9)
    ]
    step = len(candidate_gnss) // 35
    selected_entries = candidate_gnss[::step][:35]
    test_durs = [30.0, 45.0, 60.0, 75.0]

    benchmark_rows = []
    detailed_results = []

    for i, g_ent in enumerate(selected_entries):
        dur = test_durs[i % len(test_durs)]
        res = run_scenario(trip_sm, calib_samples, v_preds, road_net_sm, g_ent, dur)
        if res is not None:
            detailed_results.append(res)
            benchmark_rows.append({
                "scenario_id": len(benchmark_rows) + 1,
                "trip": "S-M (Unseen)",
                "start_time_s": res["t_start_s"],
                "duration_s": dur,
                "distance_m": res["dist_m"],
                "pure_err_m": res["pure_err_m"],
                "pure_drift_pct": res["pure_drift_pct"],
                "map_err_m": res["map_err_m"],
                "map_drift_pct": res["map_drift_pct"],
            })

    df = pd.DataFrame(benchmark_rows)
    print("\n==========================================================================")
    print("      NEW UNSEEN DATASET BENCHMARK RESULTS: S-M.csv (35 SCENARIOS)        ")
    print("==========================================================================")
    print(f"Total Scenarios Evaluated: {len(df)}")
    print(f"Pure 6-Axis EKF Median Drift:      {df['pure_drift_pct'].median():.2f}% (P90: {df['pure_drift_pct'].quantile(0.90):.2f}%)")
    print(f"Phase 4 Map-Matched Median Drift:  {df['map_drift_pct'].median():.2f}% (P90: {df['map_drift_pct'].quantile(0.90):.2f}%)")
    print(f"Pure 6-Axis Median Position Error: {df['pure_err_m'].median():.2f} m")
    print(f"Phase 4 Map-Matched Median Error:  {df['map_err_m'].median():.2f} m")

    long_df = df[df["distance_m"] > 500.0]
    med_df  = df[(df["distance_m"] >= 200.0) & (df["distance_m"] <= 500.0)]
    short_df = df[df["distance_m"] < 200.0]

    if len(long_df) > 0:
        print(f"\nLong Outages (>500m, N={len(long_df)}):")
        print(f"   Pure 6-Axis Drift:     {long_df['pure_drift_pct'].median():.2f}%")
        print(f"   Phase 4 Matched Drift: {long_df['map_drift_pct'].median():.2f}%")
        print(f"   Phase 4 Matched Error: {long_df['map_err_m'].median():.2f} m")

    if len(med_df) > 0:
        print(f"\nMedium Outages (200-500m, N={len(med_df)}):")
        print(f"   Pure 6-Axis Drift:     {med_df['pure_drift_pct'].median():.2f}%")
        print(f"   Phase 4 Matched Drift: {med_df['map_drift_pct'].median():.2f}%")
        print(f"   Phase 4 Matched Error: {med_df['map_err_m'].median():.2f} m")

    if len(short_df) > 0:
        print(f"\nShort Outages (<200m, N={len(short_df)}):")
        print(f"   Pure 6-Axis Drift:     {short_df['pure_drift_pct'].median():.2f}%")
        print(f"   Phase 4 Matched Drift: {short_df['map_drift_pct'].median():.2f}%")
        print(f"   Phase 4 Matched Error: {short_df['map_err_m'].median():.2f} m")

    print("==========================================================================")

    # Save raw benchmark CSV
    csv_path = os.path.join(ARTIFACT_DIR, "phase4_unseen_sm_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark CSV to {csv_path}", flush=True)

    sorted_res = sorted(detailed_results, key=lambda x: x["dist_m"], reverse=True)

    # 1. Long high-speed blackout
    plot_single_map(sorted_res[0], all_road_pts,
                    f"New Unseen Data (S-M) - Long Outage ({sorted_res[0]['dist_m']:.0f}m, {sorted_res[0]['duration_s']:.0f}s)",
                    "map_new_unseen_sm_long_outage.png")

    # 2. Medium distance outage (The exact junction scenario the user inspected!)
    # Find scenario near t=8191s
    sc_user = min(detailed_results, key=lambda x: abs(x["t_start_s"] - 8191.0))
    plot_single_map(sc_user, all_road_pts,
                    f"New Unseen Data (S-M) - Medium Outage ({sc_user['dist_m']:.0f}m, {sc_user['duration_s']:.0f}s)",
                    "map_new_unseen_sm_medium_outage.png")

    # 3. 90-Degree Sharp Turn Outage (Scenario #2)
    sc_turn = min(detailed_results, key=lambda x: abs(x["t_start_s"] - 808.0))
    plot_single_map(sc_turn, all_road_pts,
                    f"New Unseen Data (S-M) - 90-Degree Turn Outage ({sc_turn['dist_m']:.0f}m, {sc_turn['duration_s']:.0f}s)",
                    "map_new_unseen_sm_90deg_turn.png")

    # 4. Urban/turn outage
    plot_single_map(sorted_res[2*len(sorted_res)//3], all_road_pts,
                    f"New Unseen Data (S-M) - Urban Outage ({sorted_res[2*len(sorted_res)//3]['dist_m']:.0f}m, {sorted_res[2*len(sorted_res)//3]['duration_s']:.0f}s)",
                    "map_new_unseen_sm_urban_turn.png")

    # 5. Short outage
    plot_single_map(sorted_res[-2], all_road_pts,
                    f"New Unseen Data (S-M) - Short Outage ({sorted_res[-2]['dist_m']:.0f}m, {sorted_res[-2]['duration_s']:.0f}s)",
                    "map_new_unseen_sm_short_outage.png")

    # 5. Overall Drift Distribution Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    buckets = ['Long (>500m)', 'Medium (200-500m)', 'Short (<200m)', 'Overall (All)']
    pure_vals = [long_df['pure_drift_pct'].median(), med_df['pure_drift_pct'].median(), short_df['pure_drift_pct'].median(), df['pure_drift_pct'].median()]
    map_vals  = [long_df['map_drift_pct'].median(), med_df['map_drift_pct'].median(), short_df['map_drift_pct'].median(), df['map_drift_pct'].median()]

    x = np.arange(len(buckets))
    width = 0.35

    rects1 = ax.bar(x - width/2, pure_vals, width, label='Pure 6-Axis EKF (No Maps)', color='#d9534f', alpha=0.85)
    rects2 = ax.bar(x + width/2, map_vals, width, label='Phase 4 Map-Matched', color='#0275d8', alpha=0.85)

    ax.axhline(10.0, color='green', linestyle='--', linewidth=2.0, label='SIH Target Benchmark (10%)')

    ax.set_ylabel('Median Drift Percentage (%)', fontsize=12)
    ax.set_title('Generalization on Brand-New Unseen Dataset (S-M.csv)\nPure 6-Axis vs Phase 4 Map-Matched', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACT_DIR, "phase4_unseen_sm_drift_comparison_chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved drift comparison chart to {chart_path}", flush=True)

if __name__ == "__main__":
    main()
