"""
Generate Comprehensive Unseen Dataset (S-M.csv) Map Trajectory Visualizations:
Includes:
- Tier 1: Sub-10% Drift (Passed SIH Benchmark Target < 10%)
- Tier 2: 10% - 30% Drift (Strong Dead Reckoning)
- Tier 3: > 30% Drift (Challenging Multi-Branch Intersections & Stops)

Features:
- Black Dot: Blackout Entry
- Black X: Ground Truth Exit
- Red Dot: Pure 6-Axis Final Position
- Blue Square: Phase 4 Map-Matched Final Position
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
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


def run_single_scenario(trip, calib_samples, v_preds, road_net, g_entry, duration_s):
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

    ekf_pure = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_pure.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

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

        if not blackout_started and t_curr >= bo_start_ns:
            blackout_started = True
            ekf_pure._p[0] = gt_start_enu[0]
            ekf_pure._p[1] = gt_start_enu[1]
            ekf_map._p[0]  = gt_start_enu[0]
            ekf_map._p[1]  = gt_start_enu[1]

            pre_gnss_window = [g for g in valid_gnss if bo_start_ns - int(5.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
            init_road_bearing = None
            init_cands = road_net.find_candidates(gt_start_enu, radius_m=35.0)
            for s in init_cands:
                proj, d_p, _ = s.project_point(gt_start_enu)
                if d_p < 20.0:
                    init_road_bearing = s.bearing_deg
                    break
            ekf_pure.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing)
            ekf_map.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing)

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
            pure_pts.append(fused_pure.position_enu_m[:2].copy())
            map_pts.append(ekf_map._p[:2].copy())

    pure_pts = np.array(pure_pts)
    map_pts  = np.array(map_pts)

    if len(pure_pts) < 2 or len(map_pts) < 2:
        return None

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

    return {
        "gt_pts": gt_pts,
        "pure_pts": pure_pts,
        "map_pts": map_pts,
        "dist_m": gt_dist,
        "duration_s": duration_s,
        "pure_err_m": pure_err,
        "pure_drift_pct": pure_drift,
        "map_err_m": map_err,
        "map_drift_pct": map_drift,
        "t_start_s": (bo_start_ns - t0_ns) * 1e-9,
        "start_enu": gt_start_enu,
        "end_enu": gt_end_enu,
    }


def draw_panel(ax, res, all_road_pts, title, subtitle=""):
    gt_pts = res["gt_pts"]
    pure_pts = res["pure_pts"]
    map_pts = res["map_pts"]

    # Corridor Road network
    all_pts_to_fit = np.vstack([gt_pts, pure_pts, map_pts])
    margin = 80.0
    x_min, x_max = all_pts_to_fit[:, 0].min() - margin, all_pts_to_fit[:, 0].max() + margin
    y_min, y_max = all_pts_to_fit[:, 1].min() - margin, all_pts_to_fit[:, 1].max() + margin

    road_mask = (all_road_pts[:, 0] >= x_min - 100.0) & (all_road_pts[:, 0] <= x_max + 100.0) & \
                (all_road_pts[:, 1] >= y_min - 100.0) & (all_road_pts[:, 1] <= y_max + 100.0)
    local_road = all_road_pts[road_mask]
    if len(local_road) > 1:
        ax.plot(local_road[:, 0], local_road[:, 1], color="#d0d0d0", linewidth=6.0, alpha=0.9, label="Road Centerline", zorder=1)

    # Ground Truth
    ax.plot(gt_pts[:, 0], gt_pts[:, 1], color="black", linewidth=3.0, label="Ground Truth (GNSS)", zorder=3)
    ax.scatter(gt_pts[0, 0], gt_pts[0, 1], color="black", s=130, marker="o", edgecolors="white", linewidths=1.5, label="Blackout Entry (Black Dot)", zorder=6)
    ax.scatter(gt_pts[-1, 0], gt_pts[-1, 1], color="black", s=150, marker="X", edgecolors="white", linewidths=1.5, label="Ground Truth Exit (Black X)", zorder=6)

    # Pure 6-Axis
    ax.plot(pure_pts[:, 0], pure_pts[:, 1], color="#d9534f", linestyle=":", linewidth=2.8, label=f"Pure 6-Axis ({res['pure_drift_pct']:.1f}%)", zorder=4)
    ax.scatter(pure_pts[-1, 0], pure_pts[-1, 1], color="#d9534f", s=120, marker="o", edgecolors="black", linewidths=1.2, label="Pure Final (Red Dot)", zorder=7)

    # Phase 4 Map-Matched
    ax.plot(map_pts[:, 0], map_pts[:, 1], color="#0275d8", linestyle="--", linewidth=3.0, label=f"Phase 4 Matched ({res['map_drift_pct']:.1f}%)", zorder=5)
    ax.scatter(map_pts[-1, 0], map_pts[-1, 1], color="#0275d8", s=120, marker="s", edgecolors="white", linewidths=1.5, label="Phase 4 Final (Blue Dot)", zorder=8)

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("East (m)", fontsize=9)
    ax.set_ylabel("North (m)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=8)


def main():
    print("Loading Unseen Trip S-M.csv and setting up models...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = GenericDataLoader()
    trip_sm = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv")

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    calibrator = MountCalibrator(window_size=100)
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

    road_net_sm, all_road_pts = build_road_network(trip_sm, "sm_road")

    t0_ns = trip_sm.imu_samples[0].timestamp_ns
    candidate_gnss = [
        g for g in trip_sm.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps > 4.0 and g.bearing_deg is not None
        and (600.0 * 1e9) <= (g.timestamp_ns - t0_ns) <= (9500.0 * 1e9)
    ]
    step = len(candidate_gnss) // 35
    selected_entries = candidate_gnss[::step][:35]
    test_durs = [30.0, 45.0, 60.0, 75.0]

    all_results = []
    for i, g_ent in enumerate(selected_entries):
        dur = test_durs[i % len(test_durs)]
        res = run_single_scenario(trip_sm, calib_samples, v_preds, road_net_sm, g_ent, dur)
        if res is not None:
            res["scenario_id"] = i + 1
            all_results.append(res)

    print(f"Total valid scenarios extracted: {len(all_results)}", flush=True)

    tier1 = [r for r in all_results if r["map_drift_pct"] < 10.0]
    tier2 = [r for r in all_results if 10.0 <= r["map_drift_pct"] <= 30.0]
    tier3 = [r for r in all_results if r["map_drift_pct"] > 30.0]

    print(f"  Tier 1 (<10% Drift, PASSED): {len(tier1)} scenarios")
    print(f"  Tier 2 (10-30% Drift):        {len(tier2)} scenarios")
    print(f"  Tier 3 (>30% Drift):         {len(tier3)} scenarios")

    # Generate 9-Panel Comprehensive Gallery
    # Pick 3 representative from each tier
    t1_samples = [tier1[0], tier1[len(tier1)//2], tier1[-1]]
    t2_samples = [tier2[0], tier2[len(tier2)//2], tier2[-1]]
    t3_samples = [tier3[0], tier3[len(tier3)//2], tier3[-1]]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=250)
    fig.suptitle("Unseen Dataset (S-M.csv): Trajectory Map Gallery Across All Error Tiers\nComparing Pure 6-Axis (Red Dot) vs Phase 4 Map-Matched (Blue Dot)", fontsize=16, fontweight="bold", y=0.99)

    # Row 1: Tier 1 (< 10% Drift)
    for col, res in enumerate(t1_samples):
        draw_panel(axes[0, col], res, all_road_pts,
                   f"Tier 1: <10% Drift (Scenario #{res['scenario_id']})",
                   f"Dist: {res['dist_m']:.0f}m ({res['duration_s']:.0f}s) | Phase 4 Drift: {res['map_drift_pct']:.1f}% (Pure: {res['pure_drift_pct']:.1f}%)")

    # Row 2: Tier 2 (10% - 30% Drift)
    for col, res in enumerate(t2_samples):
        draw_panel(axes[1, col], res, all_road_pts,
                   f"Tier 2: 10%-30% Drift (Scenario #{res['scenario_id']})",
                   f"Dist: {res['dist_m']:.0f}m ({res['duration_s']:.0f}s) | Phase 4 Drift: {res['map_drift_pct']:.1f}% (Pure: {res['pure_drift_pct']:.1f}%)")

    # Row 3: Tier 3 (> 30% Drift)
    for col, res in enumerate(t3_samples):
        draw_panel(axes[2, col], res, all_road_pts,
                   f"Tier 3: >30% Drift (Scenario #{res['scenario_id']})",
                   f"Dist: {res['dist_m']:.0f}m ({res['duration_s']:.0f}s) | Phase 4 Drift: {res['map_drift_pct']:.1f}% (Pure: {res['pure_drift_pct']:.1f}%)")

    # Put a shared clean legend on the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    gallery_path = os.path.join(ARTIFACT_DIR, "unseen_sm_all_tiers_gallery.png")
    plt.savefig(gallery_path, bbox_inches="tight")
    plt.close()
    print(f"Master 9-Panel Gallery saved to: {gallery_path}", flush=True)

    # Also save dedicated high-resolution single plots for Tier 1, Tier 2, and Tier 3
    for name, r in [("unseen_sm_tier1_best_run.png", tier1[0]),
                    ("unseen_sm_tier2_median_run.png", tier2[len(tier2)//2]),
                    ("unseen_sm_tier3_challenging_run.png", tier3[0])]:
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        draw_panel(ax, r, all_road_pts,
                   f"Unseen S-M: {name.replace('.png', '').replace('_', ' ').title()}",
                   f"Dist: {r['dist_m']:.0f}m | Phase 4 Drift: {r['map_drift_pct']:.2f}% | Pure 6-Axis Drift: {r['pure_drift_pct']:.2f}%")
        ax.legend(loc="best", fontsize=9, frameon=True)
        plt.tight_layout()
        out_single = os.path.join(ARTIFACT_DIR, name)
        plt.savefig(out_single)
        plt.close()
        print(f"Saved single plot: {out_single}", flush=True)


if __name__ == "__main__":
    main()
