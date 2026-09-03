"""
Diagnostic Metric: Post-Turn Cross-Track Error (CTE) Growth Rate
Isolates heading/yaw estimation errors from forward velocity scaling by evaluating
cross-track error growth in the N seconds immediately following detected turn events (|omega_z| > threshold).
"""

import os
import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
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

def load_scenario_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = GenericDataLoader()
    trip_sm = loader.load_file(DATA_PATH)
    print(f"Loaded Trip S-M.csv: {len(trip_sm.imu_samples)} IMU samples, {len(trip_sm.gnss_samples)} GNSS samples.", flush=True)

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    calibrator = MountCalibrator(window_size=100)
    if "mount_pitch_deg" in trip_sm.metadata and "mount_roll_deg" in trip_sm.metadata:
        calibrator.calibrate_from_mount_angles(trip_sm.metadata["mount_pitch_deg"], trip_sm.metadata["mount_roll_deg"])
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

    road_net, all_road_pts = build_road_network(trip_sm, prefix="sm_road", min_step_m=20.0)
    return trip_sm, calib_samples, v_preds, road_net

def compute_cte_growth_for_scenario(
    trip, calib_samples, v_preds, road_net,
    t_start_s: float, duration_s: float,
    turn_thresh_dps: float = 1.5,
    eval_window_s: float = 5.0,
    use_refined: bool = False
) -> Dict[str, float]:
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(t_start_s * 1e9)
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    g_entry = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
    g_exit  = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

    gt_start_enu = geodetic_to_enu(g_entry.latitude_deg, g_entry.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_end_enu   = geodetic_to_enu(g_exit.latitude_deg, g_exit.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_disp = gt_end_enu - gt_start_enu
    gt_dist = float(np.linalg.norm(gt_disp))
    if gt_dist < 20.0:
        return {}

    ekf = ErrorStateEKF(
        turn_threshold_rad_s=np.radians(turn_thresh_dps),
        cooldown_duration_s=0.5,
        max_gyro_bias_rad_s=np.radians(0.5),
        initial_speed_scale=1.00,
    )
    warmup_start_ns = bo_start_ns - int(30 * 1e9)
    warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns))
    ekf.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    n_gnss = len(trip.gnss_samples)
    gnss_idx = 0
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
        gnss_idx += 1

    blackout_started = False
    est_records = []
    
    for j, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr < warmup_start_ns: continue
        if t_curr > bo_end_ns + int(1e9): break

        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            if g.timestamp_ns <= bo_start_ns:
                ekf.update_gnss(g)
            gnss_idx += 1

        cal = calib_samples[j]
        v_fwd = float(v_preds[j])
        m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)

        if not blackout_started and t_curr >= bo_start_ns:
            blackout_started = True
            ekf._p[0] = gt_start_enu[0]
            ekf._p[1] = gt_start_enu[1]

            if use_refined:
                pre_gnss = [g for g in valid_gnss if bo_start_ns - int(5.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
                init_road_bearing = None
                init_cands = road_net.find_candidates(gt_start_enu, radius_m=35.0)
                for s in init_cands:
                    proj, d_p, _ = s.project_point(gt_start_enu)
                    if d_p < 20.0:
                        init_road_bearing = s.bearing_deg
                        break
                ekf.seed_pre_blackout_heading(pre_gnss, road_bearing_deg=init_road_bearing)
            else:
                init_hdg_deg = g_entry.bearing_deg if g_entry.bearing_deg is not None else 0.0
                init_cands = road_net.find_candidates(gt_start_enu, radius_m=35.0)
                for s in init_cands:
                    proj, d_p, _ = s.project_point(gt_start_enu)
                    h_diff = abs((init_hdg_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)
                    if d_p < 20.0 and h_diff < 25.0:
                        init_hdg_deg = s.bearing_deg
                        break
                ekf.align_heading_to_road(init_hdg_deg)

        fused = ekf.predict(cal, vel)

        if bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 1.0:
            turn_rate_dps = abs(np.degrees(cal.gyro_vehicle[2]))
            is_turning = turn_rate_dps > 2.0
            curr_p = ekf._p[:2]
            curr_head_deg = float(np.degrees(ekf._heading_rad)) % 360.0

            if use_refined:
                var_yaw = float(ekf._P[8, 8])
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

                        ekf._p[0] = best_proj[0]
                        ekf._p[1] = best_proj[1]

                        if not is_turning and not is_fork:
                            ekf.reanchor_heading(best_s.bearing_deg, confidence=0.8)
            else:
                # Baseline Phase 4 map matching constraint
                cands = road_net.find_candidates(curr_p, radius_m=50.0)
                if len(cands) > 0:
                    valid_cands = []
                    for s in cands:
                        proj, d_perp, frac = s.project_point(curr_p)
                        h_diff = abs((curr_head_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)
                        max_h = 110.0 if is_turning else 45.0
                        if d_perp < 35.0 and h_diff < max_h:
                            end_factor = 0.05 if frac >= 0.95 else 1.0
                            score = np.exp(-0.5 * (d_perp / 8.0)**2) * np.exp(-0.5 * (h_diff / 40.0)**2) * end_factor
                            valid_cands.append((s, proj, d_perp, h_diff, score))
                    if len(valid_cands) > 0:
                        best_s, best_proj, _, _, _ = max(valid_cands, key=lambda x: x[4])
                        ekf._p[0] = best_proj[0]
                        ekf._p[1] = best_proj[1]
                        if not is_turning:
                            ekf._heading_rad = float(np.radians(best_s.bearing_deg))

        if bo_start_ns <= t_curr <= bo_end_ns:
            w_z = float(cal.gyro_vehicle[2])
            pos = ekf._p[:2].copy()
            head_rad = ekf._heading_rad
            est_records.append({
                "ts_ns": t_curr,
                "pos": pos,
                "heading_rad": head_rad,
                "w_z_dps": np.degrees(abs(w_z)),
                "v_fwd": v_fwd,
            })

    # Ground truth trajectory interpolation
    gt_times = []
    gt_enu_pts = []
    for g in valid_gnss:
        if bo_start_ns <= g.timestamp_ns <= bo_end_ns:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_times.append(g.timestamp_ns)
            gt_enu_pts.append(enu)

    gt_times = np.array(gt_times)
    gt_enu_pts = np.array(gt_enu_pts)
    if len(gt_times) < 3 or len(est_records) < 10:
        return {}

    est_ts = np.array([r["ts_ns"] for r in est_records])
    gt_interp_e = np.interp(est_ts, gt_times, gt_enu_pts[:, 0])
    gt_interp_n = np.interp(est_ts, gt_times, gt_enu_pts[:, 1])
    gt_pos_interp = np.stack([gt_interp_e, gt_interp_n], axis=1)

    gt_diffs = np.diff(gt_pos_interp, axis=0, prepend=[gt_pos_interp[0]])
    gt_headings = np.arctan2(gt_diffs[:, 0], gt_diffs[:, 1])

    cte_list = []
    for i in range(len(est_records)):
        p_est = est_records[i]["pos"]
        p_gt  = gt_pos_interp[i]
        err_vec = p_est - p_gt
        psi_gt = gt_headings[i]
        n_gt = np.array([-np.cos(psi_gt), np.sin(psi_gt)])
        cte = abs(float(np.dot(err_vec, n_gt)))
        cte_list.append(cte)
    cte_arr = np.array(cte_list)

    w_z_arr = np.array([r["w_z_dps"] for r in est_records])
    turn_events = []
    i = 0
    while i < len(w_z_arr):
        if w_z_arr[i] > turn_thresh_dps:
            start_i = i
            while i < len(w_z_arr) and w_z_arr[i] > turn_thresh_dps:
                i += 1
            end_i = i
            peak_i = start_i + np.argmax(w_z_arr[start_i:end_i])
            turn_events.append(peak_i)
        else:
            i += 1

    post_turn_cte_growths = []
    post_turn_norm_rates = []

    win_samples = int(eval_window_s * 10.0)
    for t_idx in turn_events:
        if t_idx + win_samples < len(cte_arr):
            cte_start = cte_arr[t_idx]
            cte_end   = cte_arr[t_idx + win_samples]
            d_cte = cte_end - cte_start
            v_mean = np.mean([est_records[k]["v_fwd"] for k in range(t_idx, t_idx + win_samples)])
            norm_rate = (d_cte / max(v_mean * eval_window_s, 1.0)) * 100.0
            post_turn_cte_growths.append(d_cte)
            post_turn_norm_rates.append(norm_rate)

    total_err = float(np.linalg.norm(est_records[-1]["pos"] - gt_pos_interp[-1]))
    total_drift = (total_err / gt_dist) * 100.0

    return {
        "num_turns": len(turn_events),
        "mean_cte_m": float(np.mean(cte_arr)),
        "max_cte_m": float(np.max(cte_arr)),
        "post_turn_cte_growth_m": float(np.mean(post_turn_cte_growths)) if post_turn_cte_growths else 0.0,
        "post_turn_norm_rate_pct": float(np.mean(post_turn_norm_rates)) if post_turn_norm_rates else 0.0,
        "total_drift_pct": total_drift,
        "gt_dist_m": gt_dist,
    }

def main():
    print("==========================================================================")
    print("      DIAGNOSTIC BENCHMARK: HEADING-ISOLATED POST-TURN CTE GROWTH         ")
    print("==========================================================================")
    trip, calib_samples, v_preds, road_net = load_scenario_data()

    results_csv = r"C:\Users\carpe\SIH\artifacts\phase4_unseen_sm_benchmark_results.csv"
    if not os.path.exists(results_csv):
        print("Results CSV not found!")
        return
    df = pd.read_csv(results_csv)

    target_scenarios = [2, 12, 14, 20, 30, 31]
    
    # 1. Baseline Run
    print(f"\n--- 1. EVALUATING BASELINE ON TARGET SCENARIOS: {target_scenarios} ---")
    rows_base = []
    for sc_id in target_scenarios:
        if sc_id - 1 < len(df):
            row = df.iloc[sc_id - 1]
            t_start = float(row["start_time_s"])
            dur = float(row["duration_s"])
            res = compute_cte_growth_for_scenario(trip, calib_samples, v_preds, road_net, t_start, dur, use_refined=False)
            if res:
                res["scenario_id"] = sc_id
                rows_base.append(res)
                print(f"Scenario #{sc_id:02d} ({res['gt_dist_m']:.0f}m): "
                      f"Mean CTE={res['mean_cte_m']:5.1f}m | "
                      f"Post-Turn Growth={res['post_turn_cte_growth_m']:+6.2f}m ({res['post_turn_norm_rate_pct']:+6.1f}%) | "
                      f"Drift={res['total_drift_pct']:5.1f}%")

    df_base = pd.DataFrame(rows_base)

    # 2. Refined Run
    print(f"\n--- 2. EVALUATING REFINED (COVARIANCE-DRIVEN + MULTI-EPOCH + FORK) ON SAME SCENARIOS ---")
    rows_ref = []
    for sc_id in target_scenarios:
        if sc_id - 1 < len(df):
            row = df.iloc[sc_id - 1]
            t_start = float(row["start_time_s"])
            dur = float(row["duration_s"])
            res = compute_cte_growth_for_scenario(trip, calib_samples, v_preds, road_net, t_start, dur, use_refined=True)
            if res:
                res["scenario_id"] = sc_id
                rows_ref.append(res)
                print(f"Scenario #{sc_id:02d} ({res['gt_dist_m']:.0f}m): "
                      f"Mean CTE={res['mean_cte_m']:5.1f}m | "
                      f"Post-Turn Growth={res['post_turn_cte_growth_m']:+6.2f}m ({res['post_turn_norm_rate_pct']:+6.1f}%) | "
                      f"Drift={res['total_drift_pct']:5.1f}%")

    df_ref = pd.DataFrame(rows_ref)

    # 3. Comparative Summary
    print("\n==========================================================================")
    print("           HEAD-TO-HEAD COMPARISON ON TURN/FORK FAILURE SCENARIOS         ")
    print("==========================================================================")
    print("Metric                           | Baseline Phase 4 | Refined Phase 4   | Relative Gain")
    print("--------------------------------------------------------------------------")
    b_cte = df_base['mean_cte_m'].mean()
    r_cte = df_ref['mean_cte_m'].mean()
    gain_cte = ((b_cte - r_cte) / b_cte) * 100.0
    print(f"Average Cross-Track Error (CTE)  | {b_cte:14.2f} m | {r_cte:15.2f} m | {gain_cte:+6.1f}%")

    b_gw = df_base['post_turn_cte_growth_m'].mean()
    r_gw = df_ref['post_turn_cte_growth_m'].mean()
    gain_gw = ((b_gw - r_gw) / abs(b_gw)) * 100.0 if abs(b_gw) > 1e-3 else 0.0
    print(f"Average Post-Turn CTE Growth     | {b_gw:+14.2f} m | {r_gw:+15.2f} m | {gain_gw:+6.1f}%")

    b_dr = df_base['total_drift_pct'].mean()
    r_dr = df_ref['total_drift_pct'].mean()
    gain_dr = ((b_dr - r_dr) / b_dr) * 100.0
    print(f"Average Overall Drift            | {b_dr:14.1f} % | {r_dr:15.1f} % | {gain_dr:+6.1f}%")
    print("==========================================================================")

if __name__ == "__main__":
    main()
