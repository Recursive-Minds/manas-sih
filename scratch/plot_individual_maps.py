"""
Plot high-resolution trajectory maps with crystal-clear Blue Dot, Red Dot, and Black Dot markers.
Scenarios:
- Scenario #02: 90-Degree Sharp Turn
- Scenario #30: Highway Fork / Exit
- Scenario #31: Acute Branch Split
- Scenario #17: Ultra-Precision Highway (0.49% drift)
- Scenario #10: High-Speed Curve (0.60% drift)
- Scenario #14: Urban Chicanes
"""

import os
import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork

DATA_PATH = r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv"
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"
ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
BRAIN_DIR = r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92"

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

def run_simulation(trip, calib_samples, v_preds, road_net, bo_start_s, duration_s):
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(bo_start_s * 1e9)
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    g_entry = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
    g_exit  = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

    gt_start_enu = geodetic_to_enu(g_entry.latitude_deg, g_entry.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_end_enu   = geodetic_to_enu(g_exit.latitude_deg, g_exit.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_dist = float(np.linalg.norm(gt_end_enu - gt_start_enu))

    ekf_pure = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.5))
    ekf_map  = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.5))

    warmup_start_ns = bo_start_ns - int(30 * 1e9)
    warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns))
    ekf_pure.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)
    ekf_map.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)

    n_gnss = len(trip.gnss_samples)
    gnss_idx = 0
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
        gnss_idx += 1

    pure_pts, map_pts = [], []
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
            gnss_idx += 1

        cal = calib_samples[j]
        v_fwd = float(v_preds[j])
        m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)

        if not blackout_started and t_curr >= bo_start_ns:
            blackout_started = True
            ekf_pure._p[0] = gt_start_enu[0]
            ekf_pure._p[1] = gt_start_enu[1]
            ekf_map._p[0]  = gt_start_enu[0]
            ekf_map._p[1]  = gt_start_enu[1]

            pre_gnss = [g for g in valid_gnss if bo_start_ns - int(5.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
            init_bearing = None
            init_cands = road_net.find_candidates(gt_start_enu, radius_m=35.0)
            for s in init_cands:
                proj, d_p, _ = s.project_point(gt_start_enu)
                if d_p < 20.0:
                    init_bearing = s.bearing_deg
                    break
            ekf_pure.seed_pre_blackout_heading(pre_gnss, road_bearing_deg=init_bearing)
            ekf_map.seed_pre_blackout_heading(pre_gnss, road_bearing_deg=init_bearing)

        fused_pure = ekf_pure.predict(cal, vel)
        fused_map  = ekf_map.predict(cal, vel)

        if bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 1.0:
            turn_rate_dps = abs(np.degrees(cal.gyro_vehicle[2]))
            is_turning = turn_rate_dps > 2.0
            curr_p = ekf_map._p[:2]
            curr_head_deg = float(np.degrees(ekf_map._heading_rad)) % 360.0

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

                    ekf_map._p[0] = best_proj[0]
                    ekf_map._p[1] = best_proj[1]

                    if not is_turning and not is_fork:
                        ekf_map.reanchor_heading(best_s.bearing_deg, confidence=0.8)

        if bo_start_ns <= t_curr <= bo_end_ns:
            pure_pts.append(ekf_pure._p[:2].copy())
            map_pts.append(ekf_map._p[:2].copy())

    gt_pts = [gt_start_enu]
    for g in valid_gnss:
        if bo_start_ns < g.timestamp_ns <= bo_end_ns:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_pts.append(enu)

    return np.array(gt_pts), np.array(pure_pts), np.array(map_pts), gt_dist

def plot_scenario_map(gt_pts, pure_pts, map_pts, all_road_pts, sc_id, name, gt_dist, duration_s, out_path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Road Network
    all_pts = np.vstack([gt_pts, pure_pts, map_pts])
    margin = 80.0
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    road_mask = (all_road_pts[:, 0] >= x_min - 100.0) & (all_road_pts[:, 0] <= x_max + 100.0) & \
                (all_road_pts[:, 1] >= y_min - 100.0) & (all_road_pts[:, 1] <= y_max + 100.0)
    local_road = all_road_pts[road_mask]
    if len(local_road) > 1:
        ax.plot(local_road[:, 0], local_road[:, 1], color="#444444", linewidth=8.0, alpha=0.8, label="Road Corridor Network", zorder=1)

    # Ground Truth trajectory
    ax.plot(gt_pts[:, 0], gt_pts[:, 1], color="white", linewidth=3.5, linestyle="-", label="Ground Truth (GNSS Path)", zorder=3)

    # Pure 6-Axis Trajectory
    pure_err = float(np.linalg.norm(pure_pts[-1] - gt_pts[-1]))
    pure_drift = (pure_err / gt_dist) * 100.0
    ax.plot(pure_pts[:, 0], pure_pts[:, 1], color="#ff3333", linewidth=2.8, linestyle=":", label=f"Pure 6-Axis Path ({pure_drift:.1f}% drift)", zorder=4)

    # Phase 4 Matched Trajectory
    map_err = float(np.linalg.norm(map_pts[-1] - gt_pts[-1]))
    map_drift = (map_err / gt_dist) * 100.0
    ax.plot(map_pts[:, 0], map_pts[:, 1], color="#00aaff", linewidth=3.2, linestyle="--", label=f"Phase 4 Matched Path ({map_drift:.1f}% drift)", zorder=5)

    # Explicit Markers:
    # 1. Blackout Entry: Large White-bordered Black/Yellow Circle
    ax.scatter(gt_pts[0, 0], gt_pts[0, 1], color="#ffcc00", s=220, marker="o", edgecolors="white", linewidths=2.5, label="Blackout Entry (Start Point)", zorder=8)

    # 2. Ground Truth Exit: Large Black Dot / Black 'X'
    ax.scatter(gt_pts[-1, 0], gt_pts[-1, 1], color="black", s=260, marker="o", edgecolors="#00ff88", linewidths=3.0, zorder=9)
    ax.scatter(gt_pts[-1, 0], gt_pts[-1, 1], color="#00ff88", s=180, marker="X", edgecolors="black", linewidths=1.5, label="Ground Truth Exit (Black Dot / Green 'X')", zorder=10)

    # 3. Pure 6-Axis Final Position: Red Dot
    ax.scatter(pure_pts[-1, 0], pure_pts[-1, 1], color="#ff2200", s=260, marker="o", edgecolors="white", linewidths=2.5, label=f"Pure 6-Axis Final (RED DOT, {pure_err:.1f}m err)", zorder=11)

    # 4. Phase 4 Final Position: Blue Dot
    ax.scatter(map_pts[-1, 0], map_pts[-1, 1], color="#0088ff", s=260, marker="o", edgecolors="white", linewidths=2.5, label=f"Phase 4 Final (BLUE DOT, {map_err:.1f}m err)", zorder=12)

    ax.set_title(f"Scenario #{sc_id:02d}: {name}\nDistance: {gt_dist:.0f}m | Duration: {duration_s:.0f}s | Map Drift: {map_drift:.2f}% ({map_err:.1f}m error)", fontsize=13, fontweight="bold", pad=12, color="white")
    ax.set_xlabel("Local East ENU (meters)", fontsize=11, color="#cccccc")
    ax.set_ylabel("Local North ENU (meters)", fontsize=11, color="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555555")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9.5, framealpha=0.9, facecolor="#1a1a1a", edgecolor="#555555")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}", flush=True)

def main():
    print("Loading data for individual map generation...", flush=True)
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

    selected = [
        (2,  "90-Degree Sharp Highway Turn"),
        (30, "Highway Off-Ramp Fork Split"),
        (31, "Acute Highway Branch Fork"),
        (17, "Ultra-Precision Highway Outage"),
        (10, "High-Speed Curve Outage"),
        (14, "Urban Chicane Navigation"),
    ]

    for sc_id, name in selected:
        row = df.iloc[sc_id - 1]
        t_start = float(row["start_time_s"])
        dur = float(row["duration_s"])
        gt_pts, pure_pts, map_pts, dist_m = run_simulation(trip, calib_samples, v_preds, road_net, t_start, dur)

        fname = f"map_scenario_{sc_id:02d}_{name.lower().replace(' ', '_').replace('-', '_')}.png"
        out_path = os.path.join(ARTIFACT_DIR, fname)
        plot_scenario_map(gt_pts, pure_pts, map_pts, all_road_pts, sc_id, name, dist_m, dur, out_path)

        # Copy to brain dir
        brain_path = os.path.join(BRAIN_DIR, fname)
        import shutil
        shutil.copyfile(out_path, brain_path)

    print("All individual high-res maps generated and copied to brain directory successfully!", flush=True)

if __name__ == "__main__":
    main()
