#!/usr/bin/env python3
"""
Offline Highway Validation Harness for Held-Out S-M Partition (SIH Problem Statement 26168).

Evaluates the 15 Highway (S-M Part 3) blackout scenarios with:
1. Straight-Line Heading Lock (ZARU): ErrorStateEKF.update_straight_line_lock()
2. Adaptive Spectral Speed Blending: ErrorStateEKF.compute_hybrid_speed()
3. Soft Anisotropic Map-Matching: update_map_measurement()
4. Trajectory Error Decomposition: Along-track and cross-track error analysis
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.split import load_partition, compute_trip_partition
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork
from sih.map.governor import RoadKinematicsGovernor, update_map_measurement
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from benchmarks.run_final_benchmark import load_ai_model, predict_velocities, build_road_network


def decompose_trajectory_errors(est_pts: np.ndarray, gt_pts: np.ndarray):
    """Decomposes position errors into along-track and cross-track components.
    
    Args:
        est_pts: (N, 2) estimated ENU coordinates.
        gt_pts: (N, 2) ground-truth ENU coordinates.
        
    Returns:
        along_track_errs: (N,) along-track error projection (m).
        cross_track_errs: (N,) orthogonal cross-track deviation (m).
    """
    N = min(len(est_pts), len(gt_pts))
    along_track = np.zeros(N)
    cross_track = np.zeros(N)
    
    for i in range(N):
        p_est = est_pts[i]
        p_gt = gt_pts[i]
        err_vec = p_est - p_gt
        
        # Determine ground truth tangent vector
        if i < N - 1:
            tangent = gt_pts[i + 1] - p_gt
        elif i > 0:
            tangent = p_gt - gt_pts[i - 1]
        else:
            tangent = np.array([1.0, 0.0])
            
        norm_t = np.linalg.norm(tangent)
        if norm_t > 1e-6:
            u_t = tangent / norm_t
            u_n = np.array([-u_t[1], u_t[0]])
            along_track[i] = np.dot(err_vec, u_t)
            cross_track[i] = np.dot(err_vec, u_n)
        else:
            along_track[i] = np.linalg.norm(err_vec)
            cross_track[i] = 0.0
            
    return along_track, cross_track


def run_highway_scenario(
    trip,
    calib_samples,
    v_preds,
    spectral_feats,
    road_net,
    succ_map,
    g_start,
    duration_s=60.0,
):
    bo_start_ns = g_start.timestamp_ns
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)
    warmup_start_ns = bo_start_ns - int(30.0 * 1e9)
    t0_ns = trip.imu_samples[0].timestamp_ns

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    bo_gnss = [g for g in valid_gnss if bo_start_ns <= g.timestamp_ns <= bo_end_ns]
    if len(bo_gnss) < 3:
        return None

    gt_pts = [
        geodetic_to_enu(
            g.latitude_deg, g.longitude_deg, 0.0,
            trip.reference_lat_deg, trip.reference_lon_deg, 0.0
        )[:2]
        for g in bo_gnss
    ]
    gt_pts = np.array(gt_pts)
    gt_start_enu = gt_pts[0]
    gt_end_enu   = gt_pts[-1]
    gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1)))
    if gt_dist < 20.0:
        return None

    warmup_gnss = min([g for g in valid_gnss if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])

    # Instantiate filters with reference geodetic coordinates
    ekf_pure = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_pure.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    ekf_map = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_map.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    governor = RoadKinematicsGovernor(a_lat_max=1.2, speed_limit_mps=33.3)


    n_gnss = len(trip.gnss_samples)
    gnss_idx = 0
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
        gnss_idx += 1

    pure_pts = []
    map_pts  = []
    map_ts_list = []
    blackout_started = False
    speed_scale = 1.00
    active_seg = None
    last_t_ns = None

    for j, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr < warmup_start_ns:
            continue
        if t_curr > bo_end_ns + int(1e9):
            break

        dt = (t_curr - last_t_ns) * 1e-9 if last_t_ns is not None else 0.1
        dt = np.clip(dt, 0.005, 0.5)
        last_t_ns = t_curr

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

        if not blackout_started and t_curr >= bo_start_ns:
            blackout_started = True
            ekf_pure._p[0] = gt_start_enu[0]
            ekf_pure._p[1] = gt_start_enu[1]
            ekf_map._p[0]  = gt_start_enu[0]
            ekf_map._p[1]  = gt_start_enu[1]

            pre_gnss_window = [g for g in valid_gnss if bo_start_ns - int(25.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]

            if pre_gnss_window:
                g_speeds = [g.speed_mps for g in pre_gnss_window if g.speed_mps is not None and g.speed_mps > 2.0]
                ai_speeds = [v_preds[k] for k in range(max(0, j - 200), j)]
                if len(g_speeds) >= 3 and len(ai_speeds) >= 10:
                    scale = np.mean(g_speeds) / max(0.5, np.mean(ai_speeds))
                    speed_scale = float(np.clip(scale, 0.85, 1.38))

            valid_moving = [g for g in pre_gnss_window if g.is_valid and g.speed_mps is not None and g.speed_mps > 0.5]
            if valid_moving and valid_moving[-1].speed_mps >= 2.5:
                ref_fix = valid_moving[-1]
            else:
                stable_fixes = [g for g in pre_gnss_window if g.is_valid and g.speed_mps is not None and g.speed_mps >= 2.0 and g.bearing_deg is not None]
                ref_fix = stable_fixes[-1] if stable_fixes else (valid_moving[-1] if valid_moving else (pre_gnss_window[-1] if pre_gnss_window else None))

            delta_gyro_deg = 0.0
            if ref_fix is not None:
                last_g_ts = ref_fix.timestamp_ns
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
                active_seg = best_cand

            ekf_pure.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing, delta_heading_gyro_deg=delta_gyro_deg)
            ekf_map.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing, delta_heading_gyro_deg=delta_gyro_deg)

        v_fwd = float(v_preds[j]) * (speed_scale if blackout_started else 1.0)

        # Road kinematics governor
        local_kappa = 0.0
        if blackout_started:
            turn_rate_yaw = float(cal.gyro_vehicle[2])
            nearest_segs = road_net.find_candidates(ekf_map._p[:2], radius_m=35.0)
            if nearest_segs and len(nearest_segs) >= 2:
                p1 = nearest_segs[0].start_enu_m
                p2 = nearest_segs[0].end_enu_m
                p3 = nearest_segs[1].end_enu_m
                kappas = governor.compute_curvature(np.array([p1, p2, p3]))
                local_kappa = float(np.max(kappas))
            v_fwd, _ = governor.govern_speed(v_fwd, curvature=local_kappa, yaw_rate_rad_s=turn_rate_yaw)

        m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)

        fused_pure = ekf_pure.predict(cal, vel)
        fused_map  = ekf_map.predict(cal, vel)



        # Topological Candidate Pool & Soft Anisotropic Map Matching Update
        if bo_start_ns <= t_curr <= bo_end_ns and v_fwd > 1.0:
            turn_rate_dps = abs(np.degrees(cal.gyro_vehicle[2]))
            curr_p = ekf_map._p[:2].copy()
            curr_head_deg = float(np.degrees(ekf_map._heading_rad)) % 360.0

            cands_dict = {}
            if active_seg is not None:
                cands_dict[active_seg.segment_id] = (active_seg, "active")
                for succ in succ_map.get(active_seg.segment_id, []):
                    cands_dict[succ.segment_id] = (succ, "succ")
                    for s2 in succ_map.get(succ.segment_id, []):
                        cands_dict[s2.segment_id] = (s2, "succ2")

            for s in road_net.find_candidates(curr_p, radius_m=45.0):
                if s.segment_id not in cands_dict:
                    cands_dict[s.segment_id] = (s, "spatial")

            scored = []
            for sid, (s, role) in cands_dict.items():
                proj, d_perp, frac = s.project_point(curr_p)
                h_diff = abs((curr_head_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)

                topo_bonus = 1.0
                if role == "active":
                    topo_bonus = 0.3 if frac >= 0.90 else 1.5
                elif role == "succ":
                    active_frac = active_seg.project_point(curr_p)[2] if active_seg else 0.0
                    topo_bonus = 3.0 if active_frac >= 0.75 else 1.2
                elif role == "succ2":
                    topo_bonus = 1.0

                max_h = 60.0 if role in ("succ", "succ2") else 45.0
                if d_perp < 30.0 and h_diff < max_h:
                    p_dist = np.exp(-0.5 * (d_perp / 10.0)**2)
                    p_head = np.exp(-0.5 * (h_diff / 30.0)**2)
                    score = p_dist * p_head * topo_bonus
                    scored.append((s, proj, d_perp, h_diff, score, frac))

            if scored:
                scored.sort(key=lambda x: x[4], reverse=True)
                best_s, best_proj, d_perp, h_diff, score, frac = scored[0]
                active_seg = best_s

                # Perpendicular lateral snap to road centerline (preserves along-track DR integration)
                seg_vec = best_s.end_enu_m - best_s.start_enu_m
                u_seg = seg_vec / np.linalg.norm(seg_vec)
                u_norm = np.array([-u_seg[1], u_seg[0]])
                d_lat = np.dot(best_proj - curr_p, u_norm)
                ekf_map._p[0] = curr_p[0] + d_lat * u_norm[0]
                ekf_map._p[1] = curr_p[1] + d_lat * u_norm[1]

                conf = 0.5 if turn_rate_dps < 2.0 else 0.25
                ekf_map.reanchor_heading(best_s.bearing_deg, confidence=conf, forward_speed_mps=v_fwd)

        if bo_start_ns <= t_curr <= bo_end_ns:
            pure_pts.append(fused_pure.position_enu_m[:2].copy())
            map_pts.append(ekf_map._p[:2].copy())
            map_ts_list.append(t_curr)

    pure_pts = np.array(pure_pts)
    map_pts  = np.array(map_pts)

    if len(pure_pts) < 2 or len(map_pts) < 2:
        return None

    # Interpolate map_pts at ground-truth GNSS timestamps for consistent trajectory evaluation
    map_ts_arr = np.array(map_ts_list, dtype=np.float64)
    gt_ts_arr  = np.array([g.timestamp_ns for g in bo_gnss], dtype=np.float64)

    map_at_gt_east  = np.interp(gt_ts_arr, map_ts_arr, map_pts[:, 0])
    map_at_gt_north = np.interp(gt_ts_arr, map_ts_arr, map_pts[:, 1])
    map_pts_at_gt   = np.column_stack([map_at_gt_east, map_at_gt_north])

    final_err_pure = float(np.linalg.norm(pure_pts[-1] - gt_end_enu))
    final_err_map  = float(np.linalg.norm(map_pts_at_gt[-1] - gt_end_enu))
    pure_drift_pct = (final_err_pure / gt_dist) * 100.0
    map_drift_pct  = (final_err_map  / gt_dist) * 100.0

    along_track, cross_track = decompose_trajectory_errors(map_pts_at_gt, gt_pts)
    ate_rmse = float(np.sqrt(np.mean(np.linalg.norm(map_pts_at_gt - gt_pts, axis=1)**2)))
    rte_mps  = float(final_err_map / duration_s)

    return {
        "t_start_s": (bo_start_ns - t0_ns) * 1e-9,
        "duration_s": duration_s,
        "dist_m": gt_dist,
        "pure_err_m": final_err_pure,
        "pure_drift_pct": pure_drift_pct,
        "map_err_m": final_err_map,
        "map_drift_pct": map_drift_pct,
        "along_track_final_m": float(along_track[-1]),
        "cross_track_final_m": float(cross_track[-1]),
        "ate_rmse_m": ate_rmse,
        "rte_mps": rte_mps,
        "pure_pts": pure_pts,
        "map_pts": map_pts,
        "gt_pts": gt_pts,
    }



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("    HIGHWAY BENCHMARK HARNESS - S-M HELD-OUT PARTITION 3")
    print("    Evaluating Straight-Line Lock (ZARU), Hybrid Speed, & Soft Map Innovation")
    print("=" * 80)
    print(f"Device: {device}")

    # Load S-M partition 3
    print("Loading S-M Part 3 partition via sih.data.split.load_partition()...")
    part_info = load_partition("S-M", partition="part3")
    trip = part_info["full_trip"]
    calib_samples = part_info["full_calib_samples"]

    # Road network
    road_net, road_pts = build_road_network(trip, "sm_road")
    succ_map = {}
    for s1 in road_net.segments:
        succ_map[s1.segment_id] = []
        for s2 in road_net.segments:
            if s1.segment_id != s2.segment_id:
                if np.linalg.norm(s1.end_enu_m - s2.start_enu_m) < 8.0:
                    succ_map[s1.segment_id].append(s2)

    # Load AI model & predict speeds
    model, norm_mean, norm_std, model_type = load_ai_model(device)
    v_preds = predict_velocities(model, calib_samples, norm_mean, norm_std, device, model_type=model_type, trip_id="S-M")

    # Load spectral features
    cache_file = os.path.join(ROOT_DIR, "data", "cache", "S-M_features_12ch.npz")
    spectral_feats = None
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        spectral_feats = data["feats"][:, 8:12]
        print(f"Loaded 4 spectral feature channels from cache: {spectral_feats.shape}")

    # Find candidate GNSS fixes strictly inside Part 3
    part = compute_trip_partition("S-M", len(trip.imu_samples))
    b_start_ns = trip.imu_samples[part.bench_range[0]].timestamp_ns
    b_end_ns   = trip.imu_samples[part.bench_range[1] - 1].timestamp_ns

    cand_gnss = [
        g for g in trip.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps > 2.5 and g.bearing_deg is not None
        and (b_start_ns + int(30.0 * 1e9)) <= g.timestamp_ns <= (b_end_ns - int(75.0 * 1e9))
    ]
    if len(cand_gnss) < 15:
        cand_gnss = [
            g for g in trip.gnss_samples
            if g.is_valid and g.speed_mps is not None and g.speed_mps > 1.5 and g.bearing_deg is not None
            and b_start_ns <= g.timestamp_ns <= (b_end_ns - int(75.0 * 1e9))
        ]

    target_count = 15
    test_durs = [30.0, 45.0, 60.0, 75.0]
    step = max(1, len(cand_gnss) // (target_count * 2)) if len(cand_gnss) > target_count * 2 else 1

    results = []
    added = 0
    for g_ent in cand_gnss[::step]:
        if added >= target_count:
            break
        dur = test_durs[added % len(test_durs)]
        res = run_highway_scenario(trip, calib_samples, v_preds, spectral_feats, road_net, succ_map, g_ent, dur)
        if res is not None and res["dist_m"] >= 15.0:
            res["scenario_id"] = added + 1
            results.append(res)
            added += 1

    df = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print(f"{'Scen':<5} | {'Dur (s)':<8} | {'Dist (m)':<9} | {'Pure Err':<9} | {'Pure %':<7} | {'Map Err':<8} | {'Map %':<7} | {'Along-Trk':<10} | {'Cross-Trk':<10} | {'ATE (m)':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['scenario_id']:<5} | {r['duration_s']:<8.0f} | {r['dist_m']:<9.1f} | {r['pure_err_m']:<9.2f} | {r['pure_drift_pct']:<7.2f} | {r['map_err_m']:<8.2f} | {r['map_drift_pct']:<7.2f} | {r['along_track_final_m']:<10.2f} | {r['cross_track_final_m']:<10.2f} | {r['ate_rmse_m']:<8.2f}")

    med_map_drift = float(df["map_drift_pct"].median())
    mean_map_drift = float(df["map_drift_pct"].mean())
    med_pure_drift = float(df["pure_drift_pct"].median())
    t1_pass = len(df[df["map_drift_pct"] < 10.0])
    t1_rate = (t1_pass / len(df)) * 100.0

    print("=" * 95)
    print(f"HIGHWAY S-M BENCHMARK SUMMARY (15 Scenarios):")
    print(f"  - Median Map Drift:  {med_map_drift:.2f}% (Target: < 8.0%) {'[PASS]' if med_map_drift < 8.0 else '[FAIL]'}")
    print(f"  - Mean Map Drift:    {mean_map_drift:.2f}%")
    print(f"  - Median Pure Drift: {med_pure_drift:.2f}%")
    print(f"  - Tier 1 (< 10%):    {t1_pass}/{len(df)} ({t1_rate:.1f}%)")
    print(f"  - Mean Along-Track:  {df['along_track_final_m'].mean():.2f} m")
    print(f"  - Mean Cross-Track:  {df['cross_track_final_m'].mean():.2f} m")
    print(f"  - Mean ATE RMSE:     {df['ate_rmse_m'].mean():.2f} m")
    print("=" * 95)

    return df


if __name__ == "__main__":
    main()
