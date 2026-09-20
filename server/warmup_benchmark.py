"""
server/warmup_benchmark.py
--------------------------
Step 1 Follow-up: Pre-Blackout Warm-Up Length Sensitivity & Failure Analysis

Evaluates 4 configurations on Seed 541098 (40 canonical scenarios):
1. Full History (= Canonical benchmark direct call, reproducing 14.32% / 32.87% / 17 Tier-1)
2. 30.0 s App-Realistic Cold Start (cutting IMU, GNSS, calibration, features to bo_start - 30s)
3. 10.0 s App-Realistic Cold Start (cutting to bo_start - 10s)
4. 5.0 s App-Realistic Cold Start (cutting to bo_start - 5s)
"""

from __future__ import annotations
import os
import sys
import copy
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator, calibrate_stream
from sih.features.streaming import StreamingFeatureExtractor
from sih.map.network import load_trip_road_network, RoadNetwork
from sih.models.inference import load_ai_model, predict_velocities
from sih.engine.dead_reckoning_engine import DeadReckoningEngine, run_dead_reckoning_scenario
from sih.core.contracts import GNSSSample, VelocityEstimate, IMUSample, CalibratedSample
from sih.fusion.es_ekf import ErrorStateEKF
from sih.map.governor import RoadKinematicsGovernor
from sih.map.matcher import HMMMapMatcher
from sih.engine.speed_observer import KinematicSpeedObserver
from sih.data.geo import geodetic_to_enu
from sih.data.split import compute_trip_partition
import benchmarks.run_final_benchmark as bench


def run_app_cold_start_scenario(
    trip,
    calib_samples_baseline,
    v_preds_trip,
    road_net,
    g_entry,
    duration_s,
    domain="Highway",
    can_speeds=None,
    warmup_cap_s=None,
):
    """
    Simulates an app-realistic cold start where the app is launched at
    t_start = bo_start - warmup_cap_s.
    All data before t_start is absent.
    """
    bo_start_ns = g_entry.timestamp_ns
    bo_end_ns = bo_start_ns + int(duration_s * 1e9)

    if warmup_cap_s is None:
        # Full history mode: use canonical DeadReckoningEngine directly
        res = run_dead_reckoning_scenario(
            trip=trip,
            calib_samples=calib_samples_baseline,
            v_preds=v_preds_trip,
            road_net=road_net,
            g_entry=g_entry,
            duration_s=duration_s,
            domain=domain,
            can_speeds=can_speeds,
            enable_speed_scale=True,
        )

        # Check turn events and yaw lock status at bo_start in full history
        calib_check = MountCalibrator(min_samples=30)
        g_i = 0
        n_g = len(trip.gnss_samples)
        for imu in trip.imu_samples:
            if imu.timestamp_ns > bo_start_ns:
                break
            while g_i < n_g and trip.gnss_samples[g_i].timestamp_ns <= imu.timestamp_ns:
                if trip.gnss_samples[g_i].timestamp_ns <= bo_start_ns:
                    calib_check.observe_gnss(trip.gnss_samples[g_i])
                g_i += 1
            calib_check.update(imu)

        turn_cnt = len(calib_check._turn_events)
        diag = {
            "warmup_cap_s": "Full History",
            "yaw_locked": (turn_cnt >= 8),
            "turn_events_count": turn_cnt,
            "gravity_converged": calib_check.is_calibrated,
            "alpha_learned": True,
            "macro_buffer_warm": True,
            "map_drift_pct": res["map_drift_pct"] if res else None,
            "pure_drift_pct": res["pure_drift_pct"] if res else None,
            "dist_m": res["dist_m"] if res else None,
        }
        return res, diag

    # App-realistic cold start:
    t_start_ns = bo_start_ns - int(warmup_cap_s * 1e9)

    # 1. Chronologically interleave cold IMU and GNSS samples into fresh MountCalibrator
    calibrator = MountCalibrator(min_samples=30)
    cold_gnss = [g for g in trip.gnss_samples if t_start_ns <= g.timestamp_ns <= bo_start_ns]
    cold_imu = [im for im in trip.imu_samples if t_start_ns <= im.timestamp_ns <= bo_start_ns]

    g_idx = 0
    n_g = len(cold_gnss)
    calib_cold = []
    for imu in cold_imu:
        while g_idx < n_g and cold_gnss[g_idx].timestamp_ns <= imu.timestamp_ns:
            calibrator.observe_gnss(cold_gnss[g_idx])
            g_idx += 1
        calib_cold.append(calibrator.update(imu))

    # Diagnostics at bo_start:
    gravity_converged = calibrator.is_calibrated
    turn_events_count = len(calibrator._turn_events)
    yaw_locked = (turn_events_count >= 8)

    # 2. Cold-start StreamingFeatureExtractor:
    feature_extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
    for c in calib_cold:
        feature_extractor.push(c)
    macro_buffer_warm = feature_extractor.is_warm

    # 3. For blackout interval, freeze mount orientation from pre-START data:
    blackout_imu = [im for im in trip.imu_samples if bo_start_ns < im.timestamp_ns <= bo_end_ns]
    calib_bo = [calibrator.update(im) for im in blackout_imu]
    sim_calib = calib_cold + calib_bo

    # Overlay cold-calibrated samples into a copy of calib_samples_baseline for this scenario
    calib_samples_scenario = list(calib_samples_baseline)
    j_start = next((i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= t_start_ns), 0)
    for idx_rel, c_sample in enumerate(sim_calib):
        idx_target = j_start + idx_rel
        if idx_target < len(calib_samples_scenario):
            calib_samples_scenario[idx_target] = c_sample

    # 4. Execute DeadReckoningEngine
    engine = DeadReckoningEngine(enable_speed_scale=True)
    res = engine.run_scenario(
        trip=trip,
        calib_samples=calib_samples_scenario,
        v_preds=v_preds_trip,
        road_net=road_net,
        g_entry=g_entry,
        duration_s=duration_s,
        domain=domain,
        can_speeds=can_speeds,
    )

    valid_moving_gnss = [g for g in cold_gnss if g.is_valid and (g.speed_mps or 0.0) > 2.0]
    alpha_learned = (len(valid_moving_gnss) >= 3 and len(cold_imu) >= 10)

    diag = {
        "warmup_cap_s": f"{warmup_cap_s:.1f} s",
        "yaw_locked": yaw_locked,
        "turn_events_count": turn_events_count,
        "gravity_converged": gravity_converged,
        "alpha_learned": alpha_learned,
        "macro_buffer_warm": macro_buffer_warm,
        "map_drift_pct": res["map_drift_pct"] if res else None,
        "pure_drift_pct": res["pure_drift_pct"] if res else None,
        "dist_m": res["dist_m"] if res else None,
    }
    return res, diag


def run_full_evaluation(pre, seed=541098):
    rng = np.random.RandomState(seed)
    trip_configs = pre["trip_configs"]
    trips = pre["trips"]
    calibs = pre["calibs"]
    road_nets = pre["road_nets"]
    v_preds_dict = pre["v_preds_dict"]
    can_speeds_dict = pre["can_speeds_dict"]

    dur_cycle = [30.0, 45.0, 60.0, 75.0]

    scenarios = []
    for tid, target_count, domain in trip_configs:
        trip = trips[tid]
        if tid in ("S-S3a", "S-S4"):
            min_start_ns = trip.imu_samples[0].timestamp_ns + int(30.0 * 1e9)
            max_end_ns = trip.imu_samples[-1].timestamp_ns
        else:
            part = compute_trip_partition(tid, len(trip.imu_samples))
            b_start_ns = trip.imu_samples[part.bench_range[0]].timestamp_ns
            b_end_ns = trip.imu_samples[part.bench_range[1] - 1].timestamp_ns
            min_start_ns = b_start_ns + int(25.0 * 1e9)
            max_end_ns = b_end_ns

        trip_durs = [dur_cycle[i % len(dur_cycle)] for i in range(target_count)]
        rng.shuffle(trip_durs)

        min_spd = 2.0 if domain not in ("Urban", "Mixed") else 1.2
        cand_gnss = [
            g for g in trip.gnss_samples
            if g.is_valid and g.speed_mps is not None and g.speed_mps >= min_spd and g.bearing_deg is not None
            and min_start_ns <= g.timestamp_ns <= (max_end_ns - int(30.0 * 1e9))
        ]
        if len(cand_gnss) < target_count * 2:
            cand_gnss = [
                g for g in trip.gnss_samples
                if g.is_valid and g.speed_mps is not None and g.speed_mps >= 1.0 and g.bearing_deg is not None
                and min_start_ns <= g.timestamp_ns <= (max_end_ns - int(30.0 * 1e9))
            ]

        cand_indices = list(range(len(cand_gnss)))
        rng.shuffle(cand_indices)

        selected_candidates = []
        for sep_s in (15.0, 10.0, 5.0):
            sep_ns = int(sep_s * 1e9)
            for idx in cand_indices:
                g_cand = cand_gnss[idx]
                dur = trip_durs[len(selected_candidates) % len(trip_durs)]
                t_start = g_cand.timestamp_ns
                t_end = t_start + int(dur * 1e9)

                overlap = False
                for s_start, s_end, _, _ in selected_candidates:
                    if not (t_end + sep_ns <= s_start or t_start >= s_end + sep_ns):
                        overlap = True
                        break
                if overlap:
                    continue

                bo_gnss = [g for g in trip.gnss_samples if t_start <= g.timestamp_ns <= t_end and g.is_valid]
                if len(bo_gnss) < 2:
                    continue
                gt_pts_check = [
                    geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
                    for g in bo_gnss
                ]
                gt_dist_check = float(np.sum(np.linalg.norm(np.diff(np.array(gt_pts_check), axis=0), axis=1)))
                if gt_dist_check >= 20.0:
                    selected_candidates.append((t_start, t_end, dur, g_cand))

                if len(selected_candidates) >= target_count:
                    break
            if len(selected_candidates) >= target_count:
                break

        selected_candidates.sort(key=lambda x: x[0])
        for t_start, t_end, dur, g_cand in selected_candidates:
            scenarios.append({
                "tid": tid,
                "domain": domain,
                "dur": dur,
                "g_cand": g_cand,
            })

    configs = [None, 30.0, 10.0, 5.0]
    all_results = {}

    for cfg in configs:
        label = "Full History" if cfg is None else f"{cfg:.1f} s"
        print(f"\nRunning configuration: {label}...")
        cfg_rows = []
        for sc in scenarios:
            tid = sc["tid"]
            res, diag = run_app_cold_start_scenario(
                trip=trips[tid],
                calib_samples_baseline=calibs[tid],
                v_preds_trip=v_preds_dict[tid],
                road_net=road_nets[tid],
                g_entry=sc["g_cand"],
                duration_s=sc["dur"],
                domain=sc["domain"],
                can_speeds=can_speeds_dict.get(tid),
                warmup_cap_s=cfg,
            )
            if res is not None:
                diag["map_drift_pct"] = res["map_drift_pct"]
                diag["pure_drift_pct"] = res["pure_drift_pct"]
                diag["dist_m"] = res["dist_m"]
                cfg_rows.append(diag)

        df = pd.DataFrame(cfg_rows)
        all_results[label] = df

    return all_results


def main():
    device = torch.device("cpu")
    print("Loading benchmark data and models...")
    pre = bench.load_precomputed_benchmark_data(device=device, model_path=None, map_source="osm")

    results_dict = run_full_evaluation(pre, seed=541098)

    print("\n" + "=" * 105)
    print(f"{'Warm-Up Setting':<14} | {'Med Drift':<10} | {'P90 Drift':<10} | {'Tier-1 (<10%)':<14} | {'Yaw Locked (>=8)':<18} | {'Grav Conv':<10} | {'Alpha Ok':<10} | {'Buf Warm':<10}")
    print("-" * 105)

    for label, df in results_dict.items():
        med = df["map_drift_pct"].median()
        p90 = df["map_drift_pct"].quantile(0.90)
        t1 = (df["map_drift_pct"] < 10.0).sum()
        total = len(df)
        yaw_locked_pct = (df["yaw_locked"].sum() / total) * 100.0
        grav_pct = (df["gravity_converged"].sum() / total) * 100.0
        alpha_pct = (df["alpha_learned"].sum() / total) * 100.0
        buf_pct = (df["macro_buffer_warm"].sum() / total) * 100.0

        print(f"{label:<14} | {med:>8.2f} % | {p90:>8.2f} % | {t1:>4}/{total:<8} | {df['yaw_locked'].sum():>2}/{total} ({yaw_locked_pct:>4.1f}%) | {grav_pct:>8.1f}% | {alpha_pct:>8.1f}% | {buf_pct:>8.1f}%")

    print("\n" + "=" * 105)
    print("DRIFT PERFORMANCE SPLIT BY YAW-LOCKED (>=8 events) vs NOT-LOCKED (<8 events)")
    print("=" * 105)
    print(f"{'Warm-Up Setting':<14} | {'Yaw-Locked Count':<18} | {'Locked Med Drift':<18} | {'Not-Locked Count':<18} | {'Not-Locked Med Drift':<20}")
    print("-" * 105)
    for label, df in results_dict.items():
        locked = df[df["yaw_locked"] == True]
        not_locked = df[df["yaw_locked"] == False]
        l_med = f"{locked['map_drift_pct'].median():.2f} %" if len(locked) > 0 else "N/A"
        nl_med = f"{not_locked['map_drift_pct'].median():.2f} %" if len(not_locked) > 0 else "N/A"
        print(f"{label:<14} | {len(locked):>4}/{len(df):<12} | {l_med:>16} | {len(not_locked):>4}/{len(df):<12} | {nl_med:>18}")


if __name__ == "__main__":
    main()
