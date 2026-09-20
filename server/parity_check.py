"""
server/parity_check.py
----------------------
Parity verification between production batch DeadReckoningEngine and
real-time streaming EngineAdapterStageB.

Usage:
  python -m server.parity_check           # Quick parity: 5 scenarios on S-S3a
  python -m server.parity_check --quick   # Quick parity: 5 scenarios on S-S3a
  python -m server.parity_check --full    # Full 40-scenario benchmark, writes PARITY_REPORT.md
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import calibrate_stream, MountCalibrator
from sih.models.inference import load_ai_model, predict_velocities
from sih.map.network import load_trip_road_network
from sih.engine.dead_reckoning_engine import run_dead_reckoning_scenario
from sih.data.geo import geodetic_to_enu
from sih.data.can_sync import load_synchronized_can_speed, is_can_supervised_allowed
from sih.data.trip_partition import compute_trip_partition
from server.engine_adapter import EngineAdapterStageB


def run_single_streaming_scenario(
    trip,
    calib_samples,
    v_preds,
    road_net,
    entry_g,
    dur_s: float,
    domain: str = "Mixed",
) -> Dict[str, Any]:
    """Runs a single scenario through EngineAdapterStageB causally sample-by-sample."""
    bo_start_ns = entry_g.timestamp_ns
    bo_end_ns = bo_start_ns + int(dur_s * 1e9)
    warmup_start_ns = max(trip.imu_samples[0].timestamp_ns, bo_start_ns - int(30.0 * 1e9))

    n_g = len(trip.gnss_samples)

    # 1. Warm up mount calibration up to blackout start
    calib_mount = MountCalibrator(min_samples=30)
    idx_g = 0
    for im in trip.imu_samples:
        if im.timestamp_ns > bo_start_ns:
            break
        while idx_g < n_g and trip.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
            calib_mount.observe_gnss(trip.gnss_samples[idx_g])
            idx_g += 1
        calib_mount.update(im)
    saved_align = calib_mount.alignment

    # 2. Instantiate streaming adapter
    adapter = EngineAdapterStageB(
        reference_lat_deg=trip.reference_lat_deg,
        reference_lon_deg=trip.reference_lon_deg,
        reference_alt_m=0.0,
        road_network=road_net,
        domain=domain,
        saved_alignment=saved_align,
        model=None,
    )

    # 3. Stream through warmup and blackout
    g_stream_idx = 0
    stream_pts = []
    stream_ts_list = []

    for j, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr < warmup_start_ns:
            continue
        if t_curr > bo_end_ns:
            break

        while g_stream_idx < n_g and trip.gnss_samples[g_stream_idx].timestamp_ns <= t_curr:
            g_fix = trip.gnss_samples[g_stream_idx]
            if g_fix.timestamp_ns <= bo_start_ns:
                adapter.on_gnss(g_fix)
            g_stream_idx += 1

        if not adapter.blackout_started and t_curr >= bo_start_ns:
            adapter.set_blackout(True, entry_gnss=entry_g)

        fused = adapter.on_imu(imu, override_speed=v_preds[j], pre_calibrated=calib_samples[j])

        if adapter.blackout_started and fused is not None:
            stream_pts.append(adapter.ekf._p[:2].copy())
            stream_ts_list.append(t_curr)

    return {
        "stream_pts": np.array(stream_pts),
        "stream_ts_arr": np.array(stream_ts_list, dtype=np.float64),
        "bo_start_ns": bo_start_ns,
        "bo_end_ns": bo_end_ns,
    }


def evaluate_parity_metrics(
    trip,
    batch_res: Dict[str, Any],
    stream_res: Dict[str, Any],
    dur_s: float,
) -> Dict[str, Any]:
    """Compares batch and streaming outputs against ground truth."""
    bo_start_ns = stream_res["bo_start_ns"]
    bo_end_ns = stream_res["bo_end_ns"]
    bo_gnss = [g for g in trip.gnss_samples if bo_start_ns <= g.timestamp_ns <= bo_end_ns and g.is_valid]
    eval_t_ns = float(bo_gnss[-1].timestamp_ns)

    gt_end_enu = geodetic_to_enu(
        bo_gnss[-1].latitude_deg, bo_gnss[-1].longitude_deg, 0.0,
        trip.reference_lat_deg, trip.reference_lon_deg, 0.0
    )[:2]

    # Batch evaluation
    batch_pts = batch_res["map_pts"]
    batch_ts_arr = batch_res["time_rel_s"] * 1e9 + bo_start_ns
    b_east = float(np.interp(eval_t_ns, batch_ts_arr, batch_pts[:, 0]))
    b_north = float(np.interp(eval_t_ns, batch_ts_arr, batch_pts[:, 1]))
    b_eval_pt = np.array([b_east, b_north])
    batch_err = float(np.linalg.norm(b_eval_pt - gt_end_enu))

    # Stage B streaming evaluation
    stream_pts = stream_res["stream_pts"]
    stream_ts_arr = stream_res["stream_ts_arr"]
    s_east = float(np.interp(eval_t_ns, stream_ts_arr, stream_pts[:, 0]))
    s_north = float(np.interp(eval_t_ns, stream_ts_arr, stream_pts[:, 1]))
    s_eval_pt = np.array([s_east, s_north])
    stage_b_err = float(np.linalg.norm(s_eval_pt - gt_end_enu))

    dist_m = float(batch_res["dist_m"])
    batch_drift_pct = (batch_err / dist_m) * 100.0 if dist_m > 0 else 0.0
    stage_b_drift_pct = (stage_b_err / dist_m) * 100.0 if dist_m > 0 else 0.0

    endpoint_diff = float(np.linalg.norm(s_eval_pt - b_eval_pt))

    # Max trajectory difference
    n_pts = min(len(stream_pts), len(batch_pts))
    common_t = np.linspace(stream_ts_arr[0], stream_ts_arr[-1], n_pts)
    b_e = np.interp(common_t, batch_ts_arr, batch_pts[:, 0])
    b_n = np.interp(common_t, batch_ts_arr, batch_pts[:, 1])
    s_e = np.interp(common_t, stream_ts_arr, stream_pts[:, 0])
    s_n = np.interp(common_t, stream_ts_arr, stream_pts[:, 1])
    traj_diffs = np.hypot(s_e - b_e, s_n - b_n)
    max_traj_diff = float(np.max(traj_diffs))

    return {
        "dist_m": dist_m,
        "batch_err_m": batch_err,
        "stage_b_err_m": stage_b_err,
        "batch_drift_pct": batch_drift_pct,
        "stage_b_drift_pct": stage_b_drift_pct,
        "drift_delta_pct": stage_b_drift_pct - batch_drift_pct,
        "endpoint_diff_m": endpoint_diff,
        "max_traj_diff_m": max_traj_diff,
    }


def run_quick_parity():
    """Runs 5 scenarios on S-S3a."""
    from scripts.quick_parity import run_quick_parity as _quick
    return _quick()


def run_full_parity(seed: int = 541098) -> None:
    """Runs the full 40 canonical scenarios and writes PARITY_REPORT.md."""
    print("=" * 80)
    print("FULL 40-SCENARIO PRODUCTION PARITY BENCHMARK")
    print("Batch DeadReckoningEngine vs Streaming EngineAdapterStageB")
    print(f"Random Seed: {seed}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips")
    loader = GenericDataLoader()
    trip_configs = [
        ("S-M", 8, "Highway"),
        ("S-S2", 6, "Arterial"),
        ("S-S1", 6, "Urban"),
        ("S-S3a", 10, "Mixed"),
        ("S-S4", 10, "Arterial"),
    ]

    model, norm_mean, norm_std, model_type = load_ai_model(device)

    trips = {}
    calibs = {}
    road_nets = {}
    v_preds_dict = {}
    can_speeds_dict = {}

    for tid, count, domain in trip_configs:
        trip_path = os.path.join(data_dir, f"{tid}.csv")
        trip = loader.load_file(trip_path)
        trips[tid] = trip
        print(f"Loaded Trip {tid} ({domain}): {len(trip.imu_samples):,} IMU, {len(trip.gnss_samples):,} GNSS")

        if is_can_supervised_allowed(tid):
            c_spd = load_synchronized_can_speed(tid, data_dir)
            if c_spd is not None:
                can_speeds_dict[tid] = c_spd

        calib_samples = calibrate_stream(trip, min_samples=30)
        calibs[tid] = calib_samples

        rnet, _ = load_trip_road_network(trip, map_source="osm", cache_dir="data/maps/cache")
        road_nets[tid] = rnet

        v_preds = predict_velocities(model, calib_samples, norm_mean, norm_std, device, model_type=model_type)
        v_preds_dict[tid] = v_preds

    # Select candidates per trip exactly as canonical benchmark
    rng = np.random.RandomState(seed)
    dur_cycle = [30.0, 45.0, 60.0, 75.0]
    all_rows = []

    print("\nExecuting 40 Scenarios...")
    print("-" * 110)
    print(f"{'Scen':<5} | {'Trip':<15} | {'Dur':<4} | {'Dist':<7} | {'Batch Err':<10} | {'Stage B Err':<11} | {'Endpt Diff':<11} | {'Max Traj':<10} | {'Status'}")
    print("-" * 110)

    scenario_id = 1
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

        calib_samples = calibs[tid]
        v_preds = v_preds_dict[tid]
        road_net = road_nets[tid]

        selected_candidates = []
        for sep_s in (15.0, 10.0, 5.0):
            sep_ns = int(sep_s * 1e9)
            for idx in cand_indices:
                if len(selected_candidates) >= target_count:
                    break
                g_cand = cand_gnss[idx]
                dur = trip_durs[len(selected_candidates)]
                t_start = g_cand.timestamp_ns
                t_end = t_start + int(dur * 1e9)
                if t_end > max_end_ns:
                    continue
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

        selected_candidates.sort(key=lambda x: x[0])

        for t_start, t_end, dur, g_cand in selected_candidates:
            # 1. Batch execution
            batch_res = run_dead_reckoning_scenario(
                trip, calib_samples, v_preds, road_net, g_cand, dur,
                domain=domain, can_speeds=can_speeds_dict.get(tid),
                enable_speed_scale=True,
            )
            if batch_res is None or batch_res["dist_m"] < 20.0:
                continue

            # 2. Stage B streaming execution
            stream_res = run_single_streaming_scenario(
                trip, calib_samples, v_preds, road_net, g_cand, dur, domain=domain
            )

            # 3. Parity metrics
            metrics = evaluate_parity_metrics(trip, batch_res, stream_res, dur)
            status = "PASS" if metrics["endpoint_diff_m"] < 5.0 else ("CLOSE" if metrics["endpoint_diff_m"] < 12.0 else "DIVERGED")

            row = {
                "scenario_id": scenario_id,
                "trip": tid,
                "domain": domain,
                "duration_s": dur,
                "distance_m": metrics["dist_m"],
                "batch_err_m": metrics["batch_err_m"],
                "stage_b_err_m": metrics["stage_b_err_m"],
                "batch_drift_pct": metrics["batch_drift_pct"],
                "stage_b_drift_pct": metrics["stage_b_drift_pct"],
                "drift_delta_pct": metrics["drift_delta_pct"],
                "endpoint_diff_m": metrics["endpoint_diff_m"],
                "max_traj_diff_m": metrics["max_traj_diff_m"],
                "status": status,
            }
            all_rows.append(row)

            print(f"#{scenario_id:<4} | {tid:<15} | {dur:2.0f}s  | {metrics['dist_m']:5.1f}m | {metrics['batch_err_m']:6.2f} m    | {metrics['stage_b_err_m']:6.2f} m     | {metrics['endpoint_diff_m']:7.2f} m    | {metrics['max_traj_diff_m']:7.2f} m  | {status}")
            scenario_id += 1

    # Generate PARITY_REPORT.md
    df = pd.DataFrame(all_rows)
    report_path = os.path.join(ROOT_DIR, "PARITY_REPORT.md")
    generate_markdown_report(df, report_path)
    print("=" * 80)
    print(f"Full Parity Report written to: {report_path}")
    print("=" * 80)


def generate_markdown_report(df: pd.DataFrame, report_path: str) -> None:
    """Writes PARITY_REPORT.md with full markdown tables and summary."""
    median_batch_drift = df["batch_drift_pct"].median()
    median_stream_drift = df["stage_b_drift_pct"].median()
    p90_batch_drift = df["batch_drift_pct"].quantile(0.90)
    p90_stream_drift = df["stage_b_drift_pct"].quantile(0.90)
    median_diff = df["endpoint_diff_m"].median()
    p90_diff = df["endpoint_diff_m"].quantile(0.90)
    max_diff = df["endpoint_diff_m"].max()
    pass_count = len(df[df["endpoint_diff_m"] < 5.0])
    close_count = len(df[(df["endpoint_diff_m"] >= 5.0) & (df["endpoint_diff_m"] < 12.0)])
    diverged_count = len(df[df["endpoint_diff_m"] >= 12.0])

    md = []
    md.append("# Production vs Streaming Parity Report")
    md.append("")
    md.append("Empirical comparison between the canonical batch DeadReckoningEngine and the real-time sample-by-sample EngineAdapterStageB streaming adapter across all 40 benchmark scenarios (Seed 541098).")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append("| Metric | Batch DeadReckoningEngine | Streaming EngineAdapterStageB | Parity Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| **Median Drift %** | {median_batch_drift:.2f}% | {median_stream_drift:.2f}% | {median_stream_drift - median_batch_drift:+.2f}% |")
    md.append(f"| **P90 Drift %** | {p90_batch_drift:.2f}% | {p90_stream_drift:.2f}% | {p90_stream_drift - p90_batch_drift:+.2f}% |")
    md.append(f"| **Target Met (<10% Drift)** | {len(df[df['batch_drift_pct'] < 10.0])} / {len(df)} | {len(df[df['stage_b_drift_pct'] < 10.0])} / {len(df)} | - |")
    md.append(f"| **Median Endpoint Diff** | - | {median_diff:.2f} m | - |")
    md.append(f"| **P90 Endpoint Diff** | - | {p90_diff:.2f} m | - |")
    md.append(f"| **Max Endpoint Diff** | - | {max_diff:.2f} m | - |")
    md.append(f"| **Exact Match (<5.0 m)** | - | {pass_count} / {len(df)} ({pass_count/len(df)*100:.1f}%) | - |")
    md.append(f"| **Close (<12.0 m)** | - | {close_count} / {len(df)} ({close_count/len(df)*100:.1f}%) | - |")
    md.append(f"| **Diverged (>=12.0 m)** | - | {diverged_count} / {len(df)} ({diverged_count/len(df)*100:.1f}%) | - |")
    md.append("")
    md.append("## Per-Scenario Parity Scorecard")
    md.append("")
    md.append("| # | Trip (Domain) | Dur | Dist (m) | Batch Err (m) | Stream Err (m) | Batch Drift % | Stream Drift % | Endpt Diff (m) | Max Traj Diff (m) | Status |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(
            f"| #{int(r['scenario_id'])} | {r['trip']} ({r['domain']}) | {int(r['duration_s'])}s | {r['distance_m']:.1f} | "
            f"{r['batch_err_m']:.2f} | {r['stage_b_err_m']:.2f} | {r['batch_drift_pct']:.2f}% | {r['stage_b_drift_pct']:.2f}% | "
            f"{r['endpoint_diff_m']:.2f} | {r['max_traj_diff_m']:.2f} | {r['status']} |"
        )
    md.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser(description="Parity check between batch engine and streaming adapter.")
    parser.add_argument("--full", action="store_true", help="Run full 40-scenario benchmark and write PARITY_REPORT.md.")
    parser.add_argument("--quick", action="store_true", help="Run quick 5-scenario check on trip S-S3a.")
    args = parser.parse_args()

    if args.full:
        run_full_parity()
    else:
        run_quick_parity()


if __name__ == "__main__":
    main()
