"""
tests/test_app_raw_parity.py
----------------------------
Test raw streaming parity against batch dead reckoning engine.
Verifies that EngineAdapterStageB in raw mode (with prime_features pre-roll from trip start
and live model inference, use_speed_smoother=False):
1. Matches batch speed predictions (< 1e-4 m/s) from warmup start through blackout end.
2. Achieves endpoint difference < 0.05 m across all 5 canonical S-S3a scenarios.
"""

import os
import sys
import pytest
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


@pytest.fixture(scope="module")
def trip_context():
    trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-S3a.csv")
    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)
    calibs = calibrate_stream(trip, min_samples=30)
    device = torch.device("cpu")
    model, norm_mean, norm_std, model_type = load_ai_model(device)
    v_preds = predict_velocities(model, calibs, norm_mean, norm_std, device, model_type=model_type, apply_smoothing=True)
    rnet, _ = load_trip_road_network(trip, map_source="osm", cache_dir="data/maps/cache")

    targets = [
        {"id": 22, "dur": 45.0, "dist": 475.2},
        {"id": 23, "dur": 75.0, "dist": 1128.4},
        {"id": 25, "dur": 45.0, "dist": 614.3},
        {"id": 26, "dur": 75.0, "dist": 892.8},
        {"id": 30, "dur": 60.0, "dist": 244.2},
    ]

    cand_gnss = [
        g for g in trip.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps >= 1.0 and g.bearing_deg is not None
    ]

    scenarios = []
    for tgt in targets:
        best_g = None
        best_diff = float("inf")
        for g in cand_gnss:
            bo_start_ns = g.timestamp_ns
            bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
            bo_gnss = [x for x in trip.gnss_samples if bo_start_ns <= x.timestamp_ns <= bo_end_ns and x.is_valid]
            if len(bo_gnss) < 3:
                continue
            pts = np.array([
                geodetic_to_enu(x.latitude_deg, x.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
                for x in bo_gnss
            ])
            dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            diff = abs(dist - tgt["dist"])
            if diff < best_diff:
                best_diff = diff
                best_g = g
                if diff < 1.0:
                    break

        batch_res = run_dead_reckoning_scenario(trip, calibs, v_preds, rnet, best_g, tgt["dur"], domain="Mixed")
        scenarios.append({
            "target": tgt,
            "entry_gnss": best_g,
            "batch_res": batch_res,
        })

    return {
        "trip": trip,
        "calibs": calibs,
        "device": device,
        "model": model,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "rnet": rnet,
        "v_preds": v_preds,
        "scenarios": scenarios,
    }


@pytest.mark.parametrize("scenario_idx", [0, 1, 2, 3, 4], ids=["sc22", "sc23", "sc25", "sc26", "sc30"])
def test_raw_pre_rolled_parity(trip_context, scenario_idx):
    trip = trip_context["trip"]
    calibs = trip_context["calibs"]
    device = trip_context["device"]
    model = trip_context["model"]
    norm_mean = trip_context["norm_mean"]
    norm_std = trip_context["norm_std"]
    rnet = trip_context["rnet"]
    v_preds = trip_context["v_preds"]

    sc_item = trip_context["scenarios"][scenario_idx]
    tgt = sc_item["target"]
    entry_g = sc_item["entry_gnss"]
    batch_res = sc_item["batch_res"]

    bo_start_ns = entry_g.timestamp_ns
    bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
    warmup_start_ns = max(trip.imu_samples[0].timestamp_ns, bo_start_ns - int(30.0 * 1e9))

    n_g = len(trip.gnss_samples)
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
        model=model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        decimate_gnss_for_seeding=False,
        lock_saved_alignment=True,
        use_speed_smoother=True,
    )

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
    warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])
    adapter.session.init_from_gnss(warmup_gnss)

    cfg = get_active_config()
    history_s = cfg.history_s if hasattr(cfg, "history_s") else 180.0
    t_hist_ns = bo_start_ns - int((history_s + 10.0) * 1e9)
    j_warm = next(i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= warmup_start_ns)

    # Pre-roll features from trip start
    adapter.prime_features(trip.imu_samples[:j_warm], calib_samples=calibs[:j_warm])
    adapter.recent_gnss_window = [g for g in valid_gnss if t_hist_ns <= g.timestamp_ns < warmup_start_ns]

    g_idx = 0
    while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns < warmup_start_ns:
        g_idx += 1

    stream_pts = []
    stream_ts = []
    live_speeds = []
    batch_speeds = []

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
        live_speeds.append(adapter.recent_ai_speeds[-1])
        batch_speeds.append(v_preds[j])
        if adapter.blackout_started and fused is not None:
            stream_pts.append(adapter.ekf._p[:2].copy())
            stream_ts.append(t_curr)

    stream_pts = np.array(stream_pts)
    stream_ts = np.array(stream_ts, dtype=np.float64)
    batch_pts = batch_res["map_pts"]
    batch_ts = batch_res["time_rel_s"] * 1e9 + bo_start_ns
    bo_gnss = [g for g in trip.gnss_samples if bo_start_ns <= g.timestamp_ns <= bo_end_ns and g.is_valid]
    eval_t = float(bo_gnss[-1].timestamp_ns)

    b_e = float(np.interp(eval_t, batch_ts, batch_pts[:, 0]))
    b_n = float(np.interp(eval_t, batch_ts, batch_pts[:, 1]))
    s_e = float(np.interp(eval_t, stream_ts, stream_pts[:, 0]))
    s_n = float(np.interp(eval_t, stream_ts, stream_pts[:, 1]))

    endpoint_diff = float(np.hypot(s_e - b_e, s_n - b_n))
    max_speed_diff = float(np.max(np.abs(np.array(live_speeds) - np.array(batch_speeds))))

    assert max_speed_diff < 1e-4, f"Scenario #{tgt['id']} max speed diff {max_speed_diff} >= 1e-4 m/s"
    assert endpoint_diff < 0.05, f"Scenario #{tgt['id']} endpoint diff {endpoint_diff} >= 0.05 m"


def test_raw_pre_rolled_parity_no_smoother_extra(trip_context):
    """
    Extra test: verifies that raw streaming also achieves exact parity
    when smoothing is disabled on both sides.
    """
    trip = trip_context["trip"]
    calibs = trip_context["calibs"]
    device = trip_context["device"]
    model = trip_context["model"]
    norm_mean = trip_context["norm_mean"]
    norm_std = trip_context["norm_std"]
    rnet = trip_context["rnet"]

    # Compute raw speeds without smoothing
    v_preds_raw = predict_velocities(model, calibs, norm_mean, norm_std, device, model_type="moe", apply_smoothing=False)

    # Test Scenario #26
    sc_item = trip_context["scenarios"][3]  # sc26
    tgt = sc_item["target"]
    entry_g = sc_item["entry_gnss"]

    batch_res = run_dead_reckoning_scenario(trip, calibs, v_preds_raw, rnet, entry_g, tgt["dur"], domain="Mixed")

    bo_start_ns = entry_g.timestamp_ns
    bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
    warmup_start_ns = max(trip.imu_samples[0].timestamp_ns, bo_start_ns - int(30.0 * 1e9))

    n_g = len(trip.gnss_samples)
    calib_mount = MountCalibrator(min_samples=30)
    idx_g = 0
    for im in trip.imu_samples:
        if im.timestamp_ns > bo_start_ns:
            break
        while idx_g < n_g and trip.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
            calib_mount.observe_gnss(trip.gnss_samples[idx_g])
            idx_g += 1
        calib_mount.update(im)

    adapter = EngineAdapterStageB(
        reference_lat_deg=trip.reference_lat_deg,
        reference_lon_deg=trip.reference_lon_deg,
        reference_alt_m=0.0,
        road_network=rnet,
        domain="Mixed",
        saved_alignment=calib_mount.alignment,
        model=model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        decimate_gnss_for_seeding=False,
        lock_saved_alignment=True,
        use_speed_smoother=False,
    )

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
    warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])
    adapter.session.init_from_gnss(warmup_gnss)

    cfg = get_active_config()
    history_s = cfg.history_s if hasattr(cfg, "history_s") else 180.0
    t_hist_ns = bo_start_ns - int((history_s + 10.0) * 1e9)
    j_warm = next(i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= warmup_start_ns)

    adapter.prime_features(trip.imu_samples[:j_warm], calib_samples=calibs[:j_warm])
    adapter.recent_gnss_window = [g for g in valid_gnss if t_hist_ns <= g.timestamp_ns < warmup_start_ns]

    g_idx = 0
    while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns < warmup_start_ns:
        g_idx += 1

    stream_pts = []
    stream_ts = []
    live_speeds = []
    batch_speeds = []

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
        live_speeds.append(adapter.recent_ai_speeds[-1])
        batch_speeds.append(v_preds_raw[j])
        if adapter.blackout_started and fused is not None:
            stream_pts.append(adapter.ekf._p[:2].copy())
            stream_ts.append(t_curr)

    stream_pts = np.array(stream_pts)
    stream_ts = np.array(stream_ts, dtype=np.float64)
    batch_pts = batch_res["map_pts"]
    batch_ts = batch_res["time_rel_s"] * 1e9 + bo_start_ns
    bo_gnss = [g for g in trip.gnss_samples if bo_start_ns <= g.timestamp_ns <= bo_end_ns and g.is_valid]
    eval_t = float(bo_gnss[-1].timestamp_ns)

    b_e = float(np.interp(eval_t, batch_ts, batch_pts[:, 0]))
    b_n = float(np.interp(eval_t, batch_ts, batch_pts[:, 1]))
    s_e = float(np.interp(eval_t, stream_ts, stream_pts[:, 0]))
    s_n = float(np.interp(eval_t, stream_ts, stream_pts[:, 1]))

    endpoint_diff = float(np.hypot(s_e - b_e, s_n - b_n))
    max_speed_diff = float(np.max(np.abs(np.array(live_speeds) - np.array(batch_speeds))))

    assert max_speed_diff < 1e-4, f"Extra test max speed diff {max_speed_diff} >= 1e-4 m/s"
    assert endpoint_diff < 0.05, f"Extra test endpoint diff {endpoint_diff} >= 0.05 m"

