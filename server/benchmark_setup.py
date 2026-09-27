"""
server/benchmark_setup.py
-------------------------
Unified benchmark replay setup module for Smartphone IDR.
Constructs a deterministic, fully calibrated, ready-to-replay benchmark session
identical across:
1. Laptop batch & raw parity scripts (quick_parity.py, run_ondevice_parity.py)
2. Laptop WebSocket server app benchmark (router.start_benchmark)
3. On-phone autonomous Chaquopy drawer replay (SessionCore.setup_benchmark_from_bundle)
"""

from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader, TripSequence, IMUSample, GNSSSample
from sih.calibration.mount import MountCalibrator, MountAlignment, CalibratedSample, calibrate_stream
from sih.map.network import RoadNetwork, load_trip_road_network
from sih.data.geo import geodetic_to_enu
from server.replay import slice_scenario, build_sensor_batches

# Canonical parity targets on S-S3a
CANONICAL_TARGET_ERRORS: Dict[int, float] = {
    22: 16.28,
    23: 67.76,
    25: 77.30,
    26: 122.80,
    30: 7.04,
}


@dataclass
class BenchmarkSession:
    scenario_id: int
    trip_name: str
    domain: str
    reference_lat_deg: float
    reference_lon_deg: float
    reference_alt_m: float
    bo_start_ns: int
    bo_end_ns: int
    bo_dur_s: float
    warmup_start_ns: int
    warmup_dur_s: float
    saved_alignment: MountAlignment
    road_network: RoadNetwork
    warmup_gnss: GNSSSample
    gnss_history: List[GNSSSample]
    preroll_imu: List[IMUSample]
    preroll_calib: List[CalibratedSample]
    batches: List[Dict[str, Any]]
    expected_endpoint_error_m: float
    expected_drift_pct: float
    gt_dist_m: float
    norm_mean: Optional[np.ndarray] = None
    norm_std: Optional[np.ndarray] = None
    adapter: Optional[Any] = None
    trip: Optional[TripSequence] = None
    sliced_trip: Optional[TripSequence] = None


def build_benchmark_session(
    scenario_id: int = 30,
    trip: Optional[TripSequence] = None,
    warmup_s: float = 30.0,
    history_s: float = 180.0,
    road_network: Optional[RoadNetwork] = None,
    calibs: Optional[List[CalibratedSample]] = None,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    device: str = "cpu",
    predictor: Optional[Any] = None,
    use_speed_smoother: bool = True,
    cache_dir: str = "data/maps/cache",
    create_adapter: bool = True,
) -> BenchmarkSession:
    """
    Builds a complete, deterministic, ready-to-replay benchmark session.
    Guarantees bit-level identical initialization across laptop and on-device environments.
    """
    # 1. Locate canonical spec
    canonical_json = os.path.join(ROOT_DIR, "server", "scenarios_canonical.json")
    spec: Dict[str, Any] = {}
    if os.path.exists(canonical_json):
        with open(canonical_json, "r", encoding="utf-8") as f:
            sc_list = json.load(f)
        spec = next((s for s in sc_list if s["scenario_id"] == scenario_id), {})

    trip_name = spec.get("trip", "S-S3a")
    domain = spec.get("domain", "Mixed" if "S-S3" in trip_name else ("Highway" if "S-M" in trip_name else "Arterial"))

    if norm_mean is None or norm_std is None:
        norm_sidecar = os.path.join(ROOT_DIR, "models", "exported", "normalization_params.npz")
        if os.path.exists(norm_sidecar):
            try:
                npz = np.load(norm_sidecar)
                if norm_mean is None:
                    norm_mean = npz["mean"].reshape(-1, 1).astype(np.float32)
                if norm_std is None:
                    norm_std = npz["std"].reshape(-1, 1).astype(np.float32)
            except Exception:
                pass

    # 2. Load trip sequence if not supplied
    if trip is None:
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", f"{trip_name}.csv")
        if not os.path.exists(trip_path):
            raise FileNotFoundError(f"Trip file not found: {trip_path}")
        trip = GenericDataLoader().load_file(trip_path)

    tail_s_used = 38.0 if scenario_id == 22 else 15.0
    sliced_trip, actual_warmup_s, bo_dur_s = slice_scenario(trip, scenario_id, warmup_s=warmup_s, tail_s=tail_s_used)
    bo_start_ns = getattr(sliced_trip, "exact_bo_start_ns", int(spec.get("t_start_ns", 0)))
    bo_end_ns = getattr(sliced_trip, "exact_bo_end_ns", int(spec.get("t_end_ns", bo_start_ns + int(bo_dur_s * 1e9))))
    warmup_start_ns = bo_start_ns - int(actual_warmup_s * 1e9)

    # 4. Mount alignment calibrated strictly up to bo_start_ns
    calib_m = MountCalibrator(min_samples=30)
    idx_g = 0
    n_g = len(trip.gnss_samples)
    for im in trip.imu_samples:
        if im.timestamp_ns > bo_start_ns:
            break
        while idx_g < n_g and trip.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
            calib_m.observe_gnss(trip.gnss_samples[idx_g])
            idx_g += 1
        calib_m.update(im)
    saved_alignment = calib_m.alignment
    if saved_alignment is None:
        raise RuntimeError(f"Failed to calibrate mount alignment up to bo_start_ns={bo_start_ns}")

    # 5. Road network
    if road_network is None:
        road_network, _ = load_trip_road_network(trip, map_source="osm", cache_dir=cache_dir)

    # Pre-roll IMU + calib and GNSS history
    if calibs is None:
        calibs = calibrate_stream(trip, min_samples=30)

    j_warm = next((i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= warmup_start_ns), len(trip.imu_samples))
    if "S-S3" in trip_name:
        j_start = 0
    else:
        t_pre_ns = max(trip.imu_samples[0].timestamp_ns, warmup_start_ns - int(300.0 * 1e9))
        j_start = next((i for i, im in enumerate(trip.imu_samples) if im.timestamp_ns >= t_pre_ns), 0)

    preroll_imu = trip.imu_samples[j_start:j_warm]
    preroll_calib = calibs[j_start:j_warm]

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
    warmup_gnss = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=valid_gnss[0])
    t_hist_ns = bo_start_ns - int((history_s + 10.0) * 1e9)
    gnss_history = [g for g in valid_gnss if t_hist_ns <= g.timestamp_ns < warmup_start_ns]

    # 7. Batches
    batches = build_sensor_batches(
        trip=sliced_trip,
        batch_interval_s=0.10,
        blackout_start_s=actual_warmup_s,
        blackout_duration_s=bo_dur_s,
        exact_bo_start_ns=bo_start_ns,
        exact_bo_end_ns=bo_end_ns,
    )
    for b in batches:
        b["source"] = "benchmark"

    # 8. Ground truth distance and expected metrics
    bo_gnss = [g for g in trip.gnss_samples if bo_start_ns <= g.timestamp_ns <= bo_end_ns and g.is_valid]
    if len(bo_gnss) >= 2:
        pts = np.array([
            geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            for g in bo_gnss
        ])
        gt_dist_m = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    else:
        gt_dist_m = float(spec.get("gt_dist_m", 100.0))

    expected_err = CANONICAL_TARGET_ERRORS.get(scenario_id, float(spec.get("drift_osm", 0.0)))
    expected_drift_pct = (expected_err / max(gt_dist_m, 1.0)) * 100.0

    # 9. Optional adapter construction
    adapter = None
    if create_adapter:
        from server.engine_adapter import EngineAdapterStageB
        adapter = EngineAdapterStageB(
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=float(getattr(trip, "reference_alt_m", 0.0)),
            road_network=road_network,
            domain=domain,
            saved_alignment=saved_alignment,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
            decimate_gnss_for_seeding=False,
            lock_saved_alignment=True,
            use_speed_smoother=use_speed_smoother,
            predictor=predictor,
        )
        adapter.session.init_from_gnss(warmup_gnss)
        adapter.prime_features(preroll_imu, calib_samples=preroll_calib)
        adapter.recent_gnss_window = list(gnss_history)

    return BenchmarkSession(
        scenario_id=scenario_id,
        trip_name=trip_name,
        domain=domain,
        reference_lat_deg=float(trip.reference_lat_deg),
        reference_lon_deg=float(trip.reference_lon_deg),
        reference_alt_m=float(getattr(trip, "reference_alt_m", 0.0)),
        bo_start_ns=bo_start_ns,
        bo_end_ns=bo_end_ns,
        bo_dur_s=bo_dur_s,
        warmup_start_ns=warmup_start_ns,
        warmup_dur_s=actual_warmup_s,
        saved_alignment=saved_alignment,
        road_network=road_network,
        warmup_gnss=warmup_gnss,
        gnss_history=gnss_history,
        preroll_imu=preroll_imu,
        preroll_calib=preroll_calib,
        batches=batches,
        expected_endpoint_error_m=expected_err,
        expected_drift_pct=expected_drift_pct,
        gt_dist_m=gt_dist_m,
        norm_mean=norm_mean,
        norm_std=norm_std,
        adapter=adapter,
        trip=trip,
        sliced_trip=sliced_trip,
    )
