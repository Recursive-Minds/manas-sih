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

def run_and_plot(trip_path, t_start_s, duration_s, title, out_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

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
    v_preds = np.array(preds_v, dtype=np.float32)

    # Build road network
    pts_enu = []
    pts_lat_lon = []
    last_p = None
    for g in trip.gnss_samples:
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        if last_p is None or np.linalg.norm(enu - last_p) >= 15.0:
            pts_enu.append(enu)
            pts_lat_lon.append((g.latitude_deg, g.longitude_deg))
            last_p = enu

    road_net = RoadNetwork.from_polyline_coords(np.array(pts_enu), pts_lat_lon, "road", "primary", 100.0)

    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(t_start_s * 1e9)
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

    gt_pts = []
    gt_ts  = []
    for g in trip.gnss_samples:
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        gt_pts.append(enu)
        gt_ts.append(g.timestamp_ns)
    gt_pts = np.array(gt_pts)
    gt_ts  = np.array(gt_ts)

    gt_bo_mask = (gt_ts >= bo_start_ns) & (gt_ts <= bo_end_ns)
    gt_bo_pts  = gt_pts[gt_bo_mask]

    warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
    warmup_gnss = min([g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=trip.gnss_samples[0])

    # Run 1: Pure 6-Axis EKF (No Map) - initialized in global Trip Reference ENU frame
    ekf_pure = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_pure.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    # Run 2: Phase 4 Tightly-Coupled Map Matched EKF - initialized in global Trip Reference ENU frame
    ekf_map = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_map.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
        gnss_idx += 1

    pure_pts = []
    map_pts  = []

    for j, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr < warmup_start_ns: continue
        if t_curr > bo_end_ns + int(1e9): break

        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            if g.timestamp_ns < bo_start_ns:
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

        fused_pure = ekf_pure.predict(cal, vel)
        fused_map  = ekf_map.predict(cal, vel)

        # Apply Tightly Coupled Map Constraints on ekf_map
        if bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 1.5:
            curr_p = ekf_map._p[:2]
            curr_head_deg = float(np.degrees(ekf_map._heading_rad)) % 360.0
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
                    best_s, best_proj, _, _, _ = max(valid_cands, key=lambda x: x[4])
                    # Cross-track corridor nudge
                    ekf_map._p[0] = ekf_map._p[0] * 0.98 + best_proj[0] * 0.02
                    ekf_map._p[1] = ekf_map._p[1] * 0.98 + best_proj[1] * 0.02
                    # Road bearing heading pull
                    r_head_rad = np.radians(best_s.bearing_deg)
                    err_h = (r_head_rad - ekf_map._heading_rad + np.pi) % (2.0 * np.pi) - np.pi
                    if abs(err_h) < np.radians(30.0):
                        ekf_map._heading_rad = (ekf_map._heading_rad + 0.015 * err_h) % (2.0 * np.pi)

        if bo_start_ns <= t_curr <= bo_end_ns:
            pure_pts.append(fused_pure.position_enu_m[:2])
            map_pts.append(ekf_map._p[:2].copy())

    pure_pts = np.array(pure_pts)
    map_pts  = np.array(map_pts)

    # Compute Final Drift Metrics
    gt_disp = gt_bo_pts[-1] - gt_bo_pts[0]
    gt_dist = float(np.linalg.norm(gt_disp))

    pure_disp = pure_pts[-1] - pure_pts[0]
    pure_err = float(np.linalg.norm(pure_disp - gt_disp))
    pure_drift = (pure_err / gt_dist) * 100.0

    map_disp = map_pts[-1] - map_pts[0]
    map_err = float(np.linalg.norm(map_disp - gt_disp))
    map_drift = (map_err / gt_dist) * 100.0

    print(f"Scenario: {title}")
    print(f"  - Total Distance: {gt_dist:.1f} m")
    print(f"  - Pure 6-Axis EKF: Error = {pure_err:.1f} m, Drift = {pure_drift:.2f}%")
    print(f"  - Phase 4 Map-Matched: Error = {map_err:.1f} m, Drift = {map_drift:.2f}%")

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot Road Network Edges (Grey)
    road_pts = np.array(pts_enu)
    ax.plot(road_pts[:, 0], road_pts[:, 1], color='#cccccc', linewidth=4.0, alpha=0.8, label="Road Network Centerline", zorder=2)

    # Plot Ground Truth (Black)
    ax.plot(gt_bo_pts[:, 0], gt_bo_pts[:, 1], 'k-', linewidth=3.0, label="Ground Truth (GNSS)", zorder=5)
    ax.scatter(gt_bo_pts[0, 0], gt_bo_pts[0, 1], color='black', s=130, zorder=7, label="Blackout Entry")
    ax.scatter(gt_bo_pts[-1, 0], gt_bo_pts[-1, 1], color='black', marker='X', s=150, zorder=7, label="Ground Truth Exit")

    # Plot Pure 6-Axis EKF (Red)
    ax.plot(pure_pts[:, 0], pure_pts[:, 1], color='#d9534f', linewidth=2.5, linestyle=':', label=f"Pure 6-Axis EKF (Drift: {pure_drift:.1f}%)", zorder=4)
    ax.scatter(pure_pts[-1, 0], pure_pts[-1, 1], color='#d9534f', marker='o', s=100, zorder=6)

    # Plot Phase 4 Map-Matched EKF (Blue)
    ax.plot(map_pts[:, 0], map_pts[:, 1], color='#0275d8', linewidth=2.8, linestyle='--', label=f"Phase 4 Map-Matched (Drift: {map_drift:.1f}%)", zorder=6)
    ax.scatter(map_pts[-1, 0], map_pts[-1, 1], color='#0275d8', marker='s', s=120, zorder=8)

    ax.set_title(f"{title}\nPure 6-Axis Drift: {pure_drift:.1f}% -> Phase 4 Matched Drift: {map_drift:.1f}%", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("East Position (meters)", fontsize=11)
    ax.set_ylabel("North Position (meters)", fontsize=11)
    ax.legend(loc="best", fontsize=10, frameon=True, facecolor='white', framealpha=0.95)

    # Zoom in around the blackout with a clean symmetric margin
    all_x = np.concatenate([gt_bo_pts[:, 0], pure_pts[:, 0], map_pts[:, 0]])
    all_y = np.concatenate([gt_bo_pts[:, 1], pure_pts[:, 1], map_pts[:, 1]])
    cx = float((np.min(all_x) + np.max(all_x)) / 2.0)
    cy = float((np.min(all_y) + np.max(all_y)) / 2.0)
    span = float(max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2.0 + 120.0)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    plt.tight_layout()

    out_path = os.path.join(ARTIFACT_DIR, out_filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}", flush=True)

def main():
    print("=== PLOTTING PHASE 4 MAP-MATCHED TRAJECTORY COMPARISONS ===", flush=True)
    # 1. Highway 75s blackout (S-S1)
    run_and_plot(
        trip_path=r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv",
        t_start_s=1956.0, duration_s=75.0,
        title="Phase 4 Map Matching Comparison (Highway 75s Outage)",
        out_filename="map_matching_comparison_highway_75s.png"
    )
    # 2. Urban 45s blackout (S-S2)
    run_and_plot(
        trip_path=r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv",
        t_start_s=8303.0, duration_s=45.0,
        title="Phase 4 Map Matching Comparison (Urban 45s Outage)",
        out_filename="map_matching_comparison_urban_45s.png"
    )

if __name__ == "__main__":
    main()
