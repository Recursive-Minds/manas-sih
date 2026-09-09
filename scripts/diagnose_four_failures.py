"""
diagnose_four_failures.py
-------------------------
Deep diagnostic tool to root-cause scenarios #15, #27, #33, and #12.
Extracts:
1. Matched road ID vs ground-truth road ID over time (wrong fork/branch divergence).
2. AI speed model mean prediction error in the 20s pre-blackout and during blackout.
3. Accelerometer variance, gyro rate, and ZUPT trigger status during stopped periods (phantom distance).
4. Error decomposition (along-track vs cross-track).
"""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.run_final_benchmark import (
    load_ai_model,
    build_road_network,
    compute_trip_partition,
    GenericDataLoader,
    MountCalibrator,
    ErrorStateEKF,
    RoadKinematicsGovernor,
    HMMMapMatcher,
    geodetic_to_enu,
    VelocityEstimate,
    DATA_DIR
)

def diagnose_scenarios():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, norm_mean, norm_std, model_type = load_ai_model(device)
    loader = GenericDataLoader()

    target_sc_ids = {12, 15, 27, 33}
    target_results = {}

    trip_configs = [
        ("S-M", 15, "Highway"),
        ("S-S2", 10, "Arterial"),
        ("S-S1", 10, "Urban"),
    ]

    test_durs = [30.0, 45.0, 60.0, 75.0]
    curr_sc_id = 0

    from benchmarks.run_final_benchmark import predict_velocities

    for tid, target_count, domain in trip_configs:
        csv_name = f"{tid}.csv"
        csv_path = os.path.join(DATA_DIR, csv_name)
        trip = loader.load_file(csv_path)

        # Calibrate exactly as production benchmark does
        calibrator = MountCalibrator(min_samples=30)
        gnss_idx = 0
        n_g = len(trip.gnss_samples)
        calib_samples = []
        for imu in trip.imu_samples:
            while gnss_idx < n_g and trip.gnss_samples[gnss_idx].timestamp_ns <= imu.timestamp_ns:
                calibrator.observe_gnss(trip.gnss_samples[gnss_idx])
                gnss_idx += 1
            calib_samples.append(calibrator.update(imu))

        # Road net
        road_net, road_pts = build_road_network(trip, prefix=f"{tid.lower()}_road")
        succ_map = {}
        for s1 in road_net.segments:
            succ_map[s1.segment_id] = []
            for s2 in road_net.segments:
                if s1.segment_id != s2.segment_id:
                    if np.linalg.norm(s1.end_enu_m - s2.start_enu_m) < 8.0:
                        succ_map[s1.segment_id].append(s2)

        # Predict AI speed
        v_preds = predict_velocities(model, calib_samples, norm_mean, norm_std, device, model_type=model_type, trip_id=tid)

        part = compute_trip_partition(tid, len(trip.imu_samples))
        b_start_ns = trip.imu_samples[part.bench_range[0]].timestamp_ns
        b_end_ns = trip.imu_samples[part.bench_range[1] - 1].timestamp_ns

        cand_gnss = [
            g for g in trip.gnss_samples
            if g.is_valid and g.speed_mps is not None and g.speed_mps > 2.5 and g.bearing_deg is not None
            and (b_start_ns + int(30.0 * 1e9)) <= g.timestamp_ns <= (b_end_ns - int(75.0 * 1e9))
        ]
        if len(cand_gnss) < target_count:
            cand_gnss = [
                g for g in trip.gnss_samples
                if g.is_valid and g.speed_mps is not None and g.speed_mps > 1.5 and g.bearing_deg is not None
                and b_start_ns <= g.timestamp_ns <= (b_end_ns - int(75.0 * 1e9))
            ]

        added_for_trip = 0
        step = max(1, len(cand_gnss) // (target_count * 2)) if len(cand_gnss) > target_count * 2 else 1
        for g_ent in cand_gnss[::step]:
            if added_for_trip >= target_count:
                break
            dur = test_durs[curr_sc_id % len(test_durs)]
            curr_sc_id += 1

            if curr_sc_id in target_sc_ids:
                print(f"\n=======================================================")
                print(f"  DIAGNOSING SCENARIO #{curr_sc_id}: {tid} ({domain}), dur={dur}s")
                print(f"=======================================================")
                res = run_diagnostic_scenario(curr_sc_id, tid, domain, trip, calib_samples, v_preds, road_net, succ_map, g_ent, dur)
                target_results[curr_sc_id] = res

            added_for_trip += 1

    return target_results


def run_diagnostic_scenario(sc_id, tid, domain, trip, calib_samples, v_preds, road_net, succ_map, g_entry, duration_s):
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = g_entry.timestamp_ns
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    gt_end_sample = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

    gt_start_enu = geodetic_to_enu(g_entry.latitude_deg, g_entry.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
    gt_end_enu   = geodetic_to_enu(gt_end_sample.latitude_deg, gt_end_sample.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]

    # Pre-blackout window: 20s immediately before entry
    pre_20s_gnss = [g for g in valid_gnss if (bo_start_ns - int(20.0 * 1e9)) <= g.timestamp_ns <= bo_start_ns]
    pre_20s_gps_speeds = [g.speed_mps for g in pre_20s_gnss if g.speed_mps is not None]

    # Find IMU index of bo_start_ns
    j_entry = min(range(len(trip.imu_samples)), key=lambda k: abs(trip.imu_samples[k].timestamp_ns - bo_start_ns))
    j_pre_20s = min(range(len(trip.imu_samples)), key=lambda k: abs(trip.imu_samples[k].timestamp_ns - (bo_start_ns - int(20.0 * 1e9))))
    pre_20s_ai_speeds = v_preds[j_pre_20s : j_entry]

    mean_pre_gps = float(np.mean(pre_20s_gps_speeds)) if pre_20s_gps_speeds else 0.0
    mean_pre_ai  = float(np.mean(pre_20s_ai_speeds)) if len(pre_20s_ai_speeds) > 0 else 0.0
    pre_speed_err = mean_pre_ai - mean_pre_gps

    # Ground truth trajectory & speeds during blackout
    bo_gnss = [g for g in valid_gnss if bo_start_ns <= g.timestamp_ns <= bo_end_ns]
    bo_gps_speeds = [g.speed_mps for g in bo_gnss if g.speed_mps is not None]

    gt_pts = [gt_start_enu]
    gt_times = [0.0]
    for g in valid_gnss:
        if bo_start_ns < g.timestamp_ns <= bo_end_ns:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_pts.append(enu)
            gt_times.append((g.timestamp_ns - bo_start_ns) * 1e-9)
    gt_pts = np.array(gt_pts)

    if len(gt_pts) > 1:
        gt_dist = float(sum(np.linalg.norm(gt_pts[k+1] - gt_pts[k]) for k in range(len(gt_pts)-1)))
    else:
        gt_dist = float(np.linalg.norm(gt_end_enu - gt_start_enu))

    # Initialize EKFs
    warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
    warmup_gnss = min([g for g in valid_gnss if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])

    ekf_pure = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_pure.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    ekf_map = ErrorStateEKF(turn_threshold_rad_s=np.radians(1.5), cooldown_duration_s=0.5, max_gyro_bias_rad_s=np.radians(0.1), initial_speed_scale=1.00)
    ekf_map.init_from_gnss(warmup_gnss, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg, reference_alt_m=0.0)

    governor = RoadKinematicsGovernor(a_lat_max=3.5, speed_limit_mps=33.3)

    n_gnss = len(trip.gnss_samples)
    gnss_idx = 0
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
        gnss_idx += 1

    blackout_started = False
    speed_scale = 1.00
    active_seg = None

    # Step-by-step diagnostic logging
    time_series = []

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
                    speed_scale = float(np.clip(scale, 0.85, 1.25))

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

        v_raw_ai = float(v_preds[j])
        v_fwd = v_raw_ai * (speed_scale if blackout_started else 1.0)

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

        # Intercept stationary detector state from EKF before prediction
        accel_buf = list(ekf_map._accel_buf)
        g_norm = float(np.linalg.norm(cal.accel_vehicle))
        temp_buf = (accel_buf + [g_norm])[-20:]
        a_var = float(np.var(temp_buf)) if len(temp_buf) >= 10 else 1.0
        gyro_norm = float(np.linalg.norm(cal.gyro_vehicle))
        g_norm_err = abs(g_norm - 9.80665)
        is_phys_rest = (a_var < 0.04 and g_norm_err < 0.6 and gyro_norm < 0.04)

        fused_pure = ekf_pure.predict(cal, vel)
        fused_map  = ekf_map.predict(cal, vel)

        # Map matching
        matched_seg_id = None
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
                matched_seg_id = best_s.segment_id
                ekf_map._p[0] = best_proj[0]
                ekf_map._p[1] = best_proj[1]
                conf = 0.5 if turn_rate_dps < 2.0 else 0.25
                ekf_map.reanchor_heading(best_s.bearing_deg, confidence=conf, forward_speed_mps=v_fwd)

        if bo_start_ns <= t_curr <= bo_end_ns:
            # Interpolate true ground truth position and speed at t_curr
            t_rel_s = (t_curr - bo_start_ns) * 1e-9
            # Closest GT GPS fix
            closest_gt = min(bo_gnss, key=lambda g: abs(g.timestamp_ns - t_curr)) if bo_gnss else None
            gt_speed_val = float(closest_gt.speed_mps) if (closest_gt and closest_gt.speed_mps is not None) else 0.0
            gt_pos_enu = geodetic_to_enu(closest_gt.latitude_deg, closest_gt.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2] if closest_gt else gt_start_enu

            # Find closest ground truth road segment to gt_pos_enu
            gt_cands = road_net.find_candidates(gt_pos_enu, radius_m=35.0)
            true_road_seg = None
            if gt_cands:
                gt_cands.sort(key=lambda s: s.project_point(gt_pos_enu)[1])
                true_road_seg = gt_cands[0].segment_id

            time_series.append({
                "t_rel_s": t_rel_s,
                "v_raw_ai": v_raw_ai,
                "v_fwd": v_fwd,
                "gt_speed": gt_speed_val,
                "a_var": a_var,
                "gyro_norm": gyro_norm,
                "is_phys_rest": is_phys_rest,
                "is_stationary": ekf_map._stat_count > 0,
                "pos_pure": ekf_pure._p[:2].copy(),
                "pos_map": ekf_map._p[:2].copy(),
                "gt_pos": gt_pos_enu,
                "matched_seg": matched_seg_id,
                "true_road_seg": true_road_seg,
            })

    # Summary analysis
    final_err_pure = float(np.linalg.norm(time_series[-1]["pos_pure"] - gt_end_enu))
    final_err_map  = float(np.linalg.norm(time_series[-1]["pos_map"] - gt_end_enu))
    pure_drift_pct = (final_err_pure / gt_dist) * 100.0
    map_drift_pct  = (final_err_map  / gt_dist) * 100.0

    # Speed metrics during blackout
    fwd_speeds = [x["v_fwd"] for x in time_series]
    gt_speeds  = [x["gt_speed"] for x in time_series]
    mean_bo_fwd = float(np.mean(fwd_speeds))
    mean_bo_gt  = float(np.mean(gt_speeds))
    bo_speed_err = mean_bo_fwd - mean_bo_gt

    # Total integrated distance during blackout
    dt_arr = [(time_series[k+1]["t_rel_s"] - time_series[k]["t_rel_s"]) for k in range(len(time_series)-1)]
    integrated_ai_dist = sum(time_series[k]["v_fwd"] * dt_arr[k] for k in range(len(dt_arr)))
    integrated_gt_dist = sum(time_series[k]["gt_speed"] * dt_arr[k] for k in range(len(dt_arr)))

    # Track divergence / fork analysis
    diverged_steps = 0
    total_active_steps = 0
    matched_segs = []
    true_segs = []
    for x in time_series:
        if x["matched_seg"] is not None and x["true_road_seg"] is not None:
            total_active_steps += 1
            matched_segs.append(x["matched_seg"])
            true_segs.append(x["true_road_seg"])
            if x["matched_seg"] != x["true_road_seg"]:
                diverged_steps += 1

    fork_mismatch_rate = (diverged_steps / max(1, total_active_steps)) * 100.0

    # Along-track vs cross-track error decomposition at the end
    end_map_pos = time_series[-1]["pos_map"]
    # Final road vector or GT bearing
    if len(gt_pts) > 1:
        tangent = gt_pts[-1] - gt_pts[-2]
    else:
        tangent = gt_end_enu - gt_start_enu
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-4:
        u_along = tangent / tangent_norm
        u_cross = np.array([-u_along[1], u_along[0]])
    else:
        u_along = np.array([1.0, 0.0])
        u_cross = np.array([0.0, 1.0])

    pos_err_vec = end_map_pos - gt_end_enu
    along_err = float(np.dot(pos_err_vec, u_along))
    cross_err = float(np.dot(pos_err_vec, u_cross))

    # Stationary analysis (stopped periods)
    stopped_steps = [x for x in time_series if x["gt_speed"] < 0.5]
    stopped_seconds = len(stopped_steps) * 0.01  # approx 100Hz
    stopped_zupt_triggered = sum(1 for x in stopped_steps if x["is_stationary"])
    stopped_zupt_rate = (stopped_zupt_triggered / max(1, len(stopped_steps))) * 100.0
    stopped_mean_ai_speed = float(np.mean([x["v_fwd"] for x in stopped_steps])) if stopped_steps else 0.0
    stopped_mean_var = float(np.mean([x["a_var"] for x in stopped_steps])) if stopped_steps else 0.0

    print(f"Distance: {gt_dist:.1f}m | Duration: {duration_s}s")
    print(f"Drift Error: Pure 6-Axis = {pure_drift_pct:.2f}% ({final_err_pure:.1f}m) | Map = {map_drift_pct:.2f}% ({final_err_map:.1f}m)")
    print(f"Error Decomposition: Along-track = {along_err:+.1f}m | Cross-track = {cross_err:+.1f}m")
    print(f"Pre-Blackout 20s: Mean GPS = {mean_pre_gps:.2f} m/s | Mean AI = {mean_pre_ai:.2f} m/s | Error = {pre_speed_err:+.2f} m/s | Scale = {speed_scale:.3f}")
    print(f"Blackout Speeds: Mean GT = {mean_bo_gt:.2f} m/s | Mean AI = {mean_bo_fwd:.2f} m/s | Error = {bo_speed_err:+.2f} m/s")
    print(f"Integrated Distance: Pred = {integrated_ai_dist:.1f}m vs GT = {integrated_gt_dist:.1f}m (diff = {integrated_ai_dist - integrated_gt_dist:+.1f}m)")
    print(f"Map Matching Fork/Segment Divergence: {fork_mismatch_rate:.1f}% ({diverged_steps}/{total_active_steps} steps)")
    # Print timeline every 10s
    print("Timeline breakdown (every 10s):")
    sample_steps = range(0, len(time_series), max(1, len(time_series) // 6))
    for idx in sample_steps:
        x = time_series[idx]
        print(f"  t={x['t_rel_s']:4.1f}s | Matched: {x['matched_seg']} | True: {x['true_road_seg']} | Match?: {x['matched_seg'] == x['true_road_seg']} | v_fwd={x['v_fwd']:.1f} m/s | v_gt={x['gt_speed']:.1f} m/s")

    if stopped_seconds > 0:
        print(f"Vehicle Stationary Analysis: Stopped = {stopped_seconds:.1f}s | ZUPT triggered: {stopped_zupt_rate:.1f}% ({stopped_zupt_triggered}/{len(stopped_steps)}) | Mean Accel Var = {stopped_mean_var:.4f} | Phantom Speed = {stopped_mean_ai_speed:.2f} m/s")
    else:
        print(f"Vehicle Stationary Analysis: Vehicle never stopped during blackout (always moving).")

    return {
        "sc_id": sc_id,
        "tid": tid,
        "domain": domain,
        "dur": duration_s,
        "dist_m": gt_dist,
        "map_drift_pct": map_drift_pct,
        "map_err_m": final_err_map,
        "along_err": along_err,
        "cross_err": cross_err,
        "mean_pre_gps": mean_pre_gps,
        "mean_pre_ai": mean_pre_ai,
        "pre_speed_err": pre_speed_err,
        "speed_scale": speed_scale,
        "mean_bo_gt": mean_bo_gt,
        "mean_bo_fwd": mean_bo_fwd,
        "bo_speed_err": bo_speed_err,
        "integrated_ai_dist": integrated_ai_dist,
        "integrated_gt_dist": integrated_gt_dist,
        "fork_mismatch_rate": fork_mismatch_rate,
        "stopped_seconds": stopped_seconds,
        "stopped_zupt_rate": stopped_zupt_rate,
        "stopped_mean_var": stopped_mean_var,
        "stopped_mean_ai_speed": stopped_mean_ai_speed,
    }

if __name__ == "__main__":
    diagnose_scenarios()
