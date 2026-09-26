"""
run_ondevice_parity.py
-----------------------
Runs on-device parity replay of the 5 canonical S-S3a scenarios and
executes on-device zero-future-leak verification on the physical phone.
"""

import os
import sys
import pickle
import json
import time
import numpy as np

from sih.data.geo import geodetic_to_enu
from sih.calibration.mount import MountCalibrator
from sih.round1.config import get_active_config
from server.engine_adapter import EngineAdapterStageB


def run_all_tests(predictor_bridge=None, bundle_path=None, report_path=None):
    print("=" * 80)
    print("=== ON-DEVICE PARITY REPLAY & NO-FUTURE-LEAK VERIFICATION ===")
    print("=" * 80)

    # 1. Load exported parity bundle
    if bundle_path is None or not os.path.exists(bundle_path):
        candidate_paths = [
            bundle_path,
            "/sdcard/Android/data/com.recursiveminds.idr.ondevice/files/s_s3a_parity_bundle.pkl",
            "/sdcard/s_s3a_parity_bundle.pkl",
            os.path.join(os.path.dirname(__file__), "s_s3a_parity_bundle.pkl"),
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                bundle_path = p
                break

    if report_path is None:
        report_path = "/sdcard/Android/data/com.recursiveminds.idr.ondevice/files/step4_parity_report.json"

    print(f"Loading bundle from {bundle_path}...")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    trip = bundle["trip"]
    calibs = bundle["calibs"]
    rnet = bundle["rnet"]
    norm_mean = bundle["norm_mean"]
    norm_std = bundle["norm_std"]
    scenarios = bundle["scenarios"]

    print(f"Loaded Trip S-S3a: {len(trip.imu_samples):,} IMU, {len(trip.gnss_samples):,} GNSS")
    print(f"Loaded Road Network: {len(rnet.segments):,} segments")
    print(f"Scenarios to replay: {len(scenarios)}")

    # 2. Setup predictor
    # If predictor_bridge is passed (JavaBridge), wrap it with JavaBridgeVelocityPredictor
    if predictor_bridge is not None:
        from sih.models.predictor import JavaBridgeVelocityPredictor
        predictor = JavaBridgeVelocityPredictor(predictor_bridge, use_bytes=True)
        print("Using on-device TFLitePredictorBridge (fast float32 bytes transfer)")
    else:
        from sih.models.predictor import create_predictor
        predictor = create_predictor("tflite", device="cpu")
        print("Using local TFLite predictor")

    # Diagnostic check: verify TFLite predictor on 3 specific windows
    from sih.features.streaming import StreamingFeatureExtractor
    fe_test = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
    feats_test = [fe_test.push(c) for c in calibs[:2563]]
    feats_arr = np.array(feats_test, dtype=np.float32)
    norm_feats = (feats_arr.T - norm_mean) / (norm_std + 1e-6)
    pad_l = np.repeat(norm_feats[:, 0:1], 59, axis=1)
    padded = np.hstack([pad_l, norm_feats]).astype(np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    w_l = np.ascontiguousarray(sliding_window_view(padded, window_shape=60, axis=1).transpose(1, 0, 2)).astype(np.float32)
    w_s = np.ascontiguousarray(w_l[:, :, -20:]).astype(np.float32)
    v0, _ = predictor.predict_window(w_s[0], w_l[0])
    v100, _ = predictor.predict_window(w_s[100], w_l[100])
    v1000, _ = predictor.predict_window(w_s[1000], w_l[1000])
    print(f"PHONE TFLite check: v0={v0:.6f}, v100={v100:.6f}, v1000={v1000:.6f}")

    results = []
    n_g = len(trip.gnss_samples)
    cfg = get_active_config()
    print(f"Active Config: online_calib={cfg.online_calib.enabled}, junction={cfg.junction.enabled}, scale_level={cfg.scale_level.enabled} (src={cfg.scale_level.source})")
    history_s = cfg.history_s if hasattr(cfg, "history_s") else 180.0
    warmup_dur_s = 30.0

    print("\nStarting Replay of 5 Canonical Scenarios on Phone Hardware...")
    print("-" * 105)
    print(f"{'Scenario':<12} | {'Laptop Batch':<13} | {'On-Device Err':<14} | {'Endpoint Diff':<14} | {'Max Traj Diff':<14} | {'Fork Flips':<10} | {'Status':<8}")
    print("-" * 105)

    all_scenarios_passed = True

    for item in scenarios:
        tgt = item["target"]
        entry_g = item["entry_gnss"]
        sc_id = tgt["id"]
        dur_s = tgt["dur"]
        laptop_batch_err = item["batch_err_m"]
        batch_pts = item["batch_pts"]
        batch_ts = item["batch_ts"]
        eval_t = item["eval_t"]
        b_e = item["b_e"]
        b_n = item["b_n"]
        gt_end_enu = item["gt_end_enu"]

        bo_start_ns = entry_g.timestamp_ns
        bo_end_ns = bo_start_ns + int(dur_s * 1e9)
        t_hist_ns = bo_start_ns - int((history_s + 10.0) * 1e9)
        warmup_start_ns = max(trip.imu_samples[0].timestamp_ns, bo_start_ns - int(warmup_dur_s * 1e9))

        # Mount calibration up to bo_start_ns
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

        adapter = EngineAdapterStageB(
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0,
            road_network=rnet,
            domain="Mixed",
            saved_alignment=saved_align,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device="cpu",
            decimate_gnss_for_seeding=False,
            lock_saved_alignment=True,
            use_speed_smoother=True,
            predictor=predictor,
        )

        valid_gnss = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
        warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])
        adapter.session.init_from_gnss(warmup_gnss)

        j_warm = next(i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= warmup_start_ns)

        # Pre-roll features from trip start (same as batch path)
        adapter.prime_features(trip.imu_samples[:j_warm], calib_samples=calibs[:j_warm])
        adapter.recent_gnss_window = [g for g in valid_gnss if t_hist_ns <= g.timestamp_ns < warmup_start_ns]

        g_idx = 0
        while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns < warmup_start_ns:
            g_idx += 1

        stream_pts = []
        stream_ts = []
        fork_flips = 0
        last_seg_id = None

        t0_replay = time.perf_counter()
        for j in range(j_warm, len(trip.imu_samples)):
            imu = trip.imu_samples[j]
            t_curr = imu.timestamp_ns
            if t_curr > bo_end_ns:
                break
            while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns <= t_curr:
                if trip.gnss_samples[g_idx].timestamp_ns <= bo_start_ns:
                    adapter.on_gnss(trip.gnss_samples[g_idx])
                g_idx += 1

            if not adapter.blackout_started and t_curr >= bo_start_ns:
                adapter.set_blackout(True, entry_gnss=entry_g)

            fused = adapter.on_imu(imu)

            if adapter.blackout_started and fused is not None:
                stream_pts.append(adapter.ekf._p[:2].copy())
                stream_ts.append(t_curr)

                # Track map matcher segment flips
                if adapter.matcher is not None and getattr(adapter.matcher, "_active_segment", None) is not None:
                    curr_seg = adapter.matcher._active_segment.segment_id
                    if curr_seg != last_seg_id and last_seg_id is not None:
                        fork_flips += 1
                    last_seg_id = curr_seg

        replay_elapsed = time.perf_counter() - t0_replay
        stream_pts = np.array(stream_pts)
        stream_ts = np.array(stream_ts, dtype=np.float64)

        # Evaluate at eval_t
        s_e = float(np.interp(eval_t, stream_ts, stream_pts[:, 0]))
        s_n = float(np.interp(eval_t, stream_ts, stream_pts[:, 1]))
        s_eval_pt = np.array([s_e, s_n])

        ondevice_err = float(np.linalg.norm(s_eval_pt - gt_end_enu))
        endpoint_diff = float(np.hypot(s_e - b_e, s_n - b_n))
        print(f"  [DIAG #{sc_id}] scale={adapter.speed_scale:.4f}, v_entry={adapter.v_entry:.2f}, s_eval=[{s_e:.2f}, {s_n:.2f}], b_eval=[{b_e:.2f}, {b_n:.2f}], pts_len={len(stream_pts)}")
        print(f"  [DIAG #{sc_id}] first_pt={stream_pts[0].tolist()}, last_pt={stream_pts[-1].tolist()}")

        # Trajectory difference
        common_t = np.linspace(stream_ts[0], stream_ts[-1], min(len(stream_pts), len(batch_pts)))
        b_interp_e = np.interp(common_t, batch_ts, batch_pts[:, 0])
        b_interp_n = np.interp(common_t, batch_ts, batch_pts[:, 1])
        s_interp_e = np.interp(common_t, stream_ts, stream_pts[:, 0])
        s_interp_n = np.interp(common_t, stream_ts, stream_pts[:, 1])
        traj_diffs = np.hypot(s_interp_e - b_interp_e, s_interp_n - b_interp_n)
        max_traj_diff = float(np.max(traj_diffs))

        passed = endpoint_diff < 1.0  # SIH Target: < 1 m diff vs laptop
        if not passed:
            all_scenarios_passed = False

        status_str = "PASS" if passed else "FAIL"
        print(f"Scenario #{sc_id:<5} | {laptop_batch_err:9.2f} m   | {ondevice_err:10.2f} m   | {endpoint_diff:10.4f} m   | {max_traj_diff:10.4f} m   | {fork_flips:8d}   | {status_str:<8}")

        results.append({
            "scenario_id": sc_id,
            "duration_s": dur_s,
            "laptop_batch_err_m": laptop_batch_err,
            "ondevice_err_m": ondevice_err,
            "endpoint_diff_m": endpoint_diff,
            "max_traj_diff_m": max_traj_diff,
            "fork_flips": fork_flips,
            "passed": passed,
            "replay_time_s": replay_elapsed,
        })

    # =========================================================================
    # On-Device No-Future-Leak Test: Post-Blackout NaN Corruption Bit-Identity
    # =========================================================================
    print("\n--------------------------------------------------------------------------------")
    print("=== ON-DEVICE ZERO-FUTURE-LEAK VERIFICATION ===")
    leak_item = scenarios[0] # Scenario #22
    entry_g = leak_item["entry_gnss"]
    bo_start_ns = entry_g.timestamp_ns
    bo_end_ns = bo_start_ns + int(leak_item["target"]["dur"] * 1e9)
    warmup_start_ns = max(trip.imu_samples[0].timestamp_ns, bo_start_ns - int(30.0 * 1e9))

    def run_leak_pass(corrupt_post_blackout_gnss: bool):
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

        adapter = EngineAdapterStageB(
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0,
            road_network=rnet,
            domain="Mixed",
            saved_alignment=saved_align,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device="cpu",
            decimate_gnss_for_seeding=False,
            lock_saved_alignment=True,
            use_speed_smoother=True,
            predictor=predictor,
        )

        valid_gnss = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
        warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])
        adapter.session.init_from_gnss(warmup_gnss)
        j_warm = next(i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= warmup_start_ns)
        adapter.prime_features(trip.imu_samples[:j_warm], calib_samples=calibs[:j_warm])
        adapter.recent_gnss_window = [g for g in valid_gnss if bo_start_ns - int(190.0 * 1e9) <= g.timestamp_ns < warmup_start_ns]

        g_idx = 0
        while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns < warmup_start_ns:
            g_idx += 1

        pts = []
        for j in range(j_warm, len(trip.imu_samples)):
            imu = trip.imu_samples[j]
            t_curr = imu.timestamp_ns
            if t_curr > bo_end_ns:
                break
            while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns <= t_curr:
                g_fix = trip.gnss_samples[g_idx]
                if corrupt_post_blackout_gnss and g_fix.timestamp_ns > bo_start_ns:
                    import dataclasses
                    corrupted = dataclasses.replace(
                        g_fix,
                        latitude_deg=float("nan"),
                        longitude_deg=float("nan"),
                        speed_mps=float("nan"),
                    )
                    adapter.on_gnss(corrupted)
                else:
                    adapter.on_gnss(g_fix)
                g_idx += 1

            if not adapter.blackout_started and t_curr >= bo_start_ns:
                adapter.set_blackout(True, entry_gnss=entry_g)

            fused = adapter.on_imu(imu)
            if adapter.blackout_started and fused is not None:
                pts.append(adapter.ekf._p[:2].copy())

        return np.array(pts)

    pts_clean = run_leak_pass(corrupt_post_blackout_gnss=False)
    pts_nan = run_leak_pass(corrupt_post_blackout_gnss=True)

    max_leak_diff = float(np.max(np.abs(pts_clean - pts_nan)))
    leak_passed = (max_leak_diff == 0.0)
    print(f"Post-Blackout NaN Corruption Test: Max Diff = {max_leak_diff:.8f} m | Status: {'PASS (Bit-Identical)' if leak_passed else 'FAIL'}")

    report = {
        "timestamp": time.time(),
        "scenarios": results,
        "all_scenarios_passed": all_scenarios_passed,
        "leak_test": {
            "max_abs_diff_m": max_leak_diff,
            "passed": leak_passed,
            "status": "PASS (Bit-Identical)" if leak_passed else "FAIL",
        },
        "overall_status": "PASS" if (all_scenarios_passed and leak_passed) else "FAIL",
    }

    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {report_path}")
    except Exception as e:
        print(f"Could not write report to {report_path}: {e}")

    return report


if __name__ == "__main__":
    run_all_tests()
