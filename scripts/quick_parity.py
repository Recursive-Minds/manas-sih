"""
scripts/quick_parity.py
-----------------------
Quick foreground parity check on trip S-S3a:
Runs 5 canonical benchmark scenarios (including #26).
Compares batch-engine (DeadReckoningEngine) vs streaming EngineAdapterStageB.
Reports:
- Batch-engine endpoint error
- Stage B endpoint error
- Endpoint difference (m)
- Max trajectory difference (m)
"""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import calibrate_stream, MountCalibrator
from sih.models.inference import load_ai_model, predict_velocities
from sih.map.network import load_trip_road_network
from sih.engine.dead_reckoning_engine import run_dead_reckoning_scenario
from sih.data.geo import geodetic_to_enu
from server.engine_adapter import EngineAdapterStageB
from sih.round1.config import get_active_config


def run_quick_parity(raw_mode: bool = False):
    mode_str = "RAW INPUT MODE (adapter computes mount + features + AI speed)" if raw_mode else "ENGINE PARITY MODE (exact component comparison)"
    print("=" * 80)
    print("QUICK PARITY CHECK: S-S3a (Mixed Domain) - 5 Canonical Scenarios")
    print(f"Batch Engine vs Streaming EngineAdapterStageB [{mode_str}]")
    print("=" * 80)

    # 1. Load Trip S-S3a
    trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-S3a.csv")
    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)
    print(f"Loaded S-S3a: {len(trip.imu_samples):,} IMU, {len(trip.gnss_samples):,} GNSS")

    # 2. Calibrations, Road Network, AI Velocities
    calibs = calibrate_stream(trip, min_samples=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_mean, norm_std, model_type = load_ai_model(device)
    v_preds = predict_velocities(model, calibs, norm_mean, norm_std, device, model_type=model_type)
    rnet, rpts = load_trip_road_network(trip, map_source="osm", cache_dir="data/maps/cache")

    # 3. Canonical target scenarios on S-S3a from Seed 541098:
    # #22: 45s, ~475.2m
    # #23: 75s, ~1128.4m
    # #25: 45s, ~614.3m
    # #26: 75s, ~892.8m (key scenario)
    # #30: 60s, ~244.2m
    targets = [
        {"id": 22, "dur": 45.0, "dist": 475.2},
        {"id": 23, "dur": 75.0, "dist": 1128.4},
        {"id": 25, "dur": 45.0, "dist": 614.3},
        {"id": 26, "dur": 75.0, "dist": 892.8},
        {"id": 30, "dur": 60.0, "dist": 244.2},
    ]

    # Pre-filter candidate GNSS fixes
    cand_gnss = [
        g for g in trip.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps >= 1.0 and g.bearing_deg is not None
    ]

    matched_scenarios = []
    print("\nLocating canonical scenarios in S-S3a...")
    for tgt in targets:
        best_g = None
        best_diff = float("inf")

        for g in cand_gnss:
            bo_start_ns = g.timestamp_ns
            bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
            bo_gnss = [x for x in trip.gnss_samples if bo_start_ns <= x.timestamp_ns <= bo_end_ns and x.is_valid]
            if len(bo_gnss) < 3:
                continue

            pts = [
                geodetic_to_enu(x.latitude_deg, x.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
                for x in bo_gnss
            ]
            pts = np.array(pts)
            dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

            diff = abs(dist - tgt["dist"])
            if diff < best_diff:
                best_diff = diff
                best_g = g
                if diff < 1.0:
                    break

        if best_diff < 5.0 and best_g is not None:
            res = run_dead_reckoning_scenario(trip, calibs, v_preds, rnet, best_g, tgt["dur"], domain="Mixed")
            matched_scenarios.append({
                "target": tgt,
                "entry_gnss": best_g,
                "batch_res": res,
            })
            print(f"  Matched Scenario #{tgt['id']}: Target Dist {tgt['dist']:.1f}m -> Found {res['dist_m']:.1f}m (start={best_g.timestamp_ns})")
        else:
            print(f"  FAILED to match Scenario #{tgt['id']} (best diff = {best_diff:.1f}m)")

    print(f"\nRunning Streaming Parity for {len(matched_scenarios)} scenarios...")
    print("-" * 95)
    print(f"{'Scenario':<12} | {'Batch Err':<11} | {'Stage B Err':<12} | {'Endpoint Diff':<15} | {'Max Traj Diff':<15} | {'Status':<10}")
    print("-" * 95)

    n_g = len(trip.gnss_samples)
    sc26_passed = False
    all_passed = True

    for item in matched_scenarios:
        tgt = item["target"]
        entry_g = item["entry_gnss"]
        batch_res = item["batch_res"]

        bo_start_ns = entry_g.timestamp_ns
        bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
        cfg = get_active_config()
        warmup_dur_s = max(60.0, cfg.history_s + 10.0) if cfg.needs_history() else 60.0
        warmup_start_ns = max(trip.imu_samples[0].timestamp_ns, bo_start_ns - int(warmup_dur_s * 1e9))

        # 1. Warm-up MountCalibrator alignment up to bo_start_ns
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

        # 2. Instantiate EngineAdapterStageB
        adapter = EngineAdapterStageB(
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0,
            road_network=rnet,
            domain="Mixed",
            saved_alignment=saved_align,
            model=model if raw_mode else None,
            norm_mean=norm_mean if raw_mode else None,
            norm_std=norm_std if raw_mode else None,
            device=device if raw_mode else None,
            decimate_gnss_for_seeding=False,
            lock_saved_alignment=True,
        )

        valid_gnss = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
        warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])
        adapter.session.init_from_gnss(warmup_gnss)

        # 3. Stream through warmup then blackout
        g_stream_idx = 0
        while g_stream_idx < n_g and trip.gnss_samples[g_stream_idx].timestamp_ns < warmup_start_ns:
            g_stream_idx += 1

        stream_pts = []
        stream_ts_list = []

        for j, imu in enumerate(trip.imu_samples):
            t_curr = imu.timestamp_ns
            if t_curr < warmup_start_ns:
                continue
            if t_curr > bo_end_ns:
                break

            # Feed GNSS that occurred before or at t_curr
            while g_stream_idx < n_g and trip.gnss_samples[g_stream_idx].timestamp_ns <= t_curr:
                g_fix = trip.gnss_samples[g_stream_idx]
                if g_fix.timestamp_ns <= bo_start_ns:
                    adapter.on_gnss(g_fix)
                g_stream_idx += 1

            # Trigger blackout exactly at bo_start_ns
            if not adapter.blackout_started and t_curr >= bo_start_ns:
                adapter.set_blackout(True, entry_gnss=entry_g)

            # Feed IMU sample
            if raw_mode:
                fused = adapter.on_imu(imu)
            else:
                fused = adapter.on_imu(imu, override_speed=v_preds[j], pre_calibrated=calibs[j])

            if adapter.blackout_started and fused is not None:
                stream_pts.append(adapter.ekf._p[:2].copy())
                stream_ts_list.append(t_curr)

        stream_pts = np.array(stream_pts)
        stream_ts_arr = np.array(stream_ts_list, dtype=np.float64)

        # 4. Compare vs Ground Truth & Batch Engine
        bo_gnss = [g for g in trip.gnss_samples if bo_start_ns <= g.timestamp_ns <= bo_end_ns and g.is_valid]
        gt_end_enu = geodetic_to_enu(
            bo_gnss[-1].latitude_deg, bo_gnss[-1].longitude_deg, 0.0,
            trip.reference_lat_deg, trip.reference_lon_deg, 0.0
        )[:2]
        eval_t_ns = float(bo_gnss[-1].timestamp_ns)

        # Batch evaluation
        batch_pts = batch_res["map_pts"]
        batch_ts_arr = batch_res["time_rel_s"] * 1e9 + bo_start_ns
        b_east = float(np.interp(eval_t_ns, batch_ts_arr, batch_pts[:, 0]))
        b_north = float(np.interp(eval_t_ns, batch_ts_arr, batch_pts[:, 1]))
        b_eval_pt = np.array([b_east, b_north])
        batch_err = float(np.linalg.norm(b_eval_pt - gt_end_enu))

        # Stage B streaming evaluation
        s_east = float(np.interp(eval_t_ns, stream_ts_arr, stream_pts[:, 0]))
        s_north = float(np.interp(eval_t_ns, stream_ts_arr, stream_pts[:, 1]))
        s_eval_pt = np.array([s_east, s_north])
        stage_b_err = float(np.linalg.norm(s_eval_pt - gt_end_enu))

        endpoint_diff = float(np.linalg.norm(s_eval_pt - b_eval_pt))

        # Trajectory difference
        common_t = np.linspace(stream_ts_arr[0], stream_ts_arr[-1], min(len(stream_pts), len(batch_pts)))
        b_interp_e = np.interp(common_t, batch_ts_arr, batch_pts[:, 0])
        b_interp_n = np.interp(common_t, batch_ts_arr, batch_pts[:, 1])
        s_interp_e = np.interp(common_t, stream_ts_arr, stream_pts[:, 0])
        s_interp_n = np.interp(common_t, stream_ts_arr, stream_pts[:, 1])
        traj_diffs = np.hypot(s_interp_e - b_interp_e, s_interp_n - b_interp_n)
        max_traj_diff = float(np.max(traj_diffs))

        thresh = 5.0 if raw_mode else 0.01
        passed = endpoint_diff < thresh
        status = "PASS" if passed else "FAIL"
        if tgt["id"] == 26:
            sc26_passed = passed
        if not passed:
            all_passed = False

        print(f"Scenario #{tgt['id']:<4} | {batch_err:6.2f} m    | {stage_b_err:6.2f} m     | {endpoint_diff:8.4f} m      | {max_traj_diff:8.4f} m      | {status}")

    thresh_str = "<5m" if raw_mode else "<0.01m"
    print("-" * 95)
    print(f"Scenario #26 Passed ({thresh_str}): {sc26_passed}")
    print(f"All Scenarios Passed ({thresh_str}): {all_passed}")
    print("=" * 80)
    return sc26_passed and all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick Parity Check: Batch vs Streaming EngineAdapterStageB")
    parser.add_argument("--raw", action="store_true", help="Run in raw-input mode (adapter computes mount + features + AI speeds)")
    args = parser.parse_args()

    success = run_quick_parity(raw_mode=args.raw)
    sys.exit(0 if success else 1)
