"""
A/B Checkpoint Comparison Benchmark.
Evaluates Champion vs CAN-Supervised Candidate across identical 40 scenarios.

Enforces:
1. Exact same random seed (541098).
2. Exact same unseen / held-out test scenarios.
3. Zero training data exposure.
4. Clean plain-text reporting.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from benchmarks.run_final_benchmark import (
    load_ai_model,
    predict_velocities,
    build_road_network,
    run_scenario,
    compute_trip_partition,
    DATA_DIR,
)
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator


def run_ab_comparison(
    champ_path: str = "models/checkpoints/best_moe_velocity_model.pt",
    cand_path: str = "models/checkpoints/moe_can_supervised.pt",
    seed: int = 541098,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("   A/B CHECKPOINT BENCHMARK: CHAMPION vs CAN-SUPERVISED CANDIDATE")
    print("   Strict Zero-Leakage: Part 3 Held-Out Benchmark Partition + Unseen Trips")
    print(f"   Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"   Random Benchmark Seed: {seed}")
    print("=" * 85)

    if not os.path.exists(champ_path):
        raise FileNotFoundError(f"Champion checkpoint not found: {champ_path}")
    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Candidate checkpoint not found: {cand_path}")

    loader = GenericDataLoader()
    trip_configs = [
        ("S-M", 8, "Highway"),
        ("S-S2", 6, "Arterial"),
        ("S-S1", 6, "Urban"),
        ("S-S3a", 10, "Mixed"),
        ("S-S4", 10, "Arterial"),
    ]

    trips = {}
    calibs = {}
    road_nets = {}
    road_pts_dict = {}

    for tid, count, domain in trip_configs:
        trip_path = os.path.join(DATA_DIR, f"{tid}.csv")
        trip = loader.load_file(trip_path)
        trips[tid] = trip

        calibrator = MountCalibrator(min_samples=30)
        gnss_idx = 0
        n_g = len(trip.gnss_samples)
        calib_samples = []
        for imu in trip.imu_samples:
            while gnss_idx < n_g and trip.gnss_samples[gnss_idx].timestamp_ns <= imu.timestamp_ns:
                calibrator.observe_gnss(trip.gnss_samples[gnss_idx])
                gnss_idx += 1
            calib_samples.append(calibrator.update(imu))
        calibs[tid] = calib_samples

        rnet, rpts = build_road_network(trip, f"{tid.lower()}_road")
        road_nets[tid] = rnet
        road_pts_dict[tid] = rpts

    # 1. Deterministically sample the exact 40 scenarios
    rng = np.random.RandomState(seed)
    dur_cycle = [30.0, 45.0, 60.0, 75.0]
    sampled_scenarios_by_trip = {}

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

        selected_for_trip = []
        for sep_s in (15.0, 10.0, 5.0):
            sep_ns = int(sep_s * 1e9)
            for idx in cand_indices:
                if len(selected_for_trip) >= target_count:
                    break
                g_cand = cand_gnss[idx]
                dur = trip_durs[len(selected_for_trip)]
                t_start = g_cand.timestamp_ns
                t_end = t_start + int(dur * 1e9)
                if t_end > max_end_ns:
                    continue
                overlap = False
                for s_start, s_end, _, _ in selected_for_trip:
                    if not (t_end + sep_ns <= s_start or t_start >= s_end + sep_ns):
                        overlap = True
                        break
                if overlap:
                    continue

                selected_for_trip.append((t_start, t_end, dur, g_cand))

            if len(selected_for_trip) >= target_count:
                break

        selected_for_trip.sort(key=lambda x: x[0])
        sampled_scenarios_by_trip[tid] = selected_for_trip

    total_scenarios = sum(len(v) for v in sampled_scenarios_by_trip.values())
    print(f"Sampled {total_scenarios} scenarios across 5 trips.")

    # 2. Evaluate Models
    models_to_test = [
        ("Champion (GPS-Supervised)", champ_path),
        ("Candidate (CAN-Supervised)", cand_path),
    ]

    all_results = {}

    for model_name, model_file in models_to_test:
        print(f"\nEvaluating: {model_name}...")
        model, norm_mean, norm_std, mtype = load_ai_model(device, model_path=model_file)

        v_preds_dict = {}
        for tid, _, _ in trip_configs:
            v_preds_dict[tid] = predict_velocities(
                model, calibs[tid], norm_mean, norm_std, device, model_type=mtype, trip_id=tid
            )

        sc_results = []
        sc_id = 1
        for tid, count, domain in trip_configs:
            trip = trips[tid]
            calib_samples = calibs[tid]
            v_preds = v_preds_dict[tid]
            road_net = road_nets[tid]

            for t_start, t_end, dur, g_cand in sampled_scenarios_by_trip[tid]:
                res = run_scenario(trip, calib_samples, v_preds, road_net, g_cand, dur, domain=domain)
                if res is not None:
                    sc_results.append({
                        "scenario_id": sc_id,
                        "trip_id": tid,
                        "domain": domain,
                        "duration_s": dur,
                        "dist_m": res["dist_m"],
                        "pure_drift_pct": res["pure_drift_pct"],
                        "map_drift_pct": res["map_drift_pct"],
                        "map_err_m": res["map_err_m"],
                    })
                    sc_id += 1

        df = pd.DataFrame(sc_results)
        all_results[model_name] = df

    # 3. Print Head-to-Head Comparison Table
    print("\n" + "=" * 85)
    print("                   HEAD-TO-HEAD BENCHMARK COMPARISON TABLE")
    print("=" * 85)
    headers = f"{'Metric':<40} | {'Champion (GPS)':<18} | {'Candidate (CAN)':<18} | {'Delta':<10}"
    print(headers)
    print("-" * 85)

    c_df = all_results["Champion (GPS-Supervised)"]
    k_df = all_results["Candidate (CAN-Supervised)"]

    c_med = c_df["map_drift_pct"].median()
    k_med = k_df["map_drift_pct"].median()
    diff_med = k_med - c_med
    print(f"{'Overall Median Drift (%)':<40} | {c_med:6.2f}%            | {k_med:6.2f}%            | {diff_med:+6.2f}%")

    c_pure = c_df["pure_drift_pct"].median()
    k_pure = k_df["pure_drift_pct"].median()
    print(f"{'Pure IMU EKF Median Drift (%)':<40} | {c_pure:6.2f}%            | {k_pure:6.2f}%            | {k_pure-c_pure:+6.2f}%")

    c_sub10 = (c_df["map_drift_pct"] < 10.0).sum() / len(c_df) * 100
    k_sub10 = (k_df["map_drift_pct"] < 10.0).sum() / len(k_df) * 100
    print(f"{'Benchmark Target Pass Rate (<10%)':<40} | {c_sub10:5.1f}%             | {k_sub10:5.1f}%             | {k_sub10-c_sub10:+5.1f}%")

    c_sub30 = (c_df["map_drift_pct"] < 30.0).sum() / len(c_df) * 100
    k_sub30 = (k_df["map_drift_pct"] < 30.0).sum() / len(k_df) * 100
    print(f"{'High Reliability Rate (<30%)':<40} | {c_sub30:5.1f}%             | {k_sub30:5.1f}%             | {k_sub30-c_sub30:+5.1f}%")

    print("-" * 85)
    print("PER-TRIP BREAKDOWN (Median Map Drift %):")
    for tid, _, dom in trip_configs:
        c_t = c_df[c_df["trip_id"] == tid]["map_drift_pct"].median()
        k_t = k_df[k_df["trip_id"] == tid]["map_drift_pct"].median()
        lbl = f"  - {tid} ({dom})"
        print(f"{lbl:<40} | {c_t:6.2f}%            | {k_t:6.2f}%            | {k_t-c_t:+6.2f}%")

    print("-" * 85)
    print("TIER BREAKDOWN (Median Map Drift %):")
    for dur in [30.0, 45.0, 60.0, 75.0]:
        c_dur = c_df[c_df["duration_s"] == dur]["map_drift_pct"].median()
        k_dur = k_df[k_df["duration_s"] == dur]["map_drift_pct"].median()
        lbl = f"  - Duration {int(dur)}s"
        print(f"{lbl:<40} | {c_dur:6.2f}%            | {k_dur:6.2f}%            | {k_dur-c_dur:+6.2f}%")

    print("=" * 85)
    return all_results


if __name__ == "__main__":
    run_ab_comparison()
