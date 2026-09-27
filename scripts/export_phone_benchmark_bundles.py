"""
scripts/export_phone_benchmark_bundles.py
-----------------------------------------
Exports compact, self-contained, fully-calibrated benchmark bundles (bench_<id>.bin)
for canonical scenarios (1, 22, 23, 25, 26, 30) into Android assets.

Bundle format: gzip-compressed JSON containing:
- Geodetic reference coordinates (lat/lon/alt)
- Serialized MountAlignment
- Road network geometry (corridor cropped or trip network, <= 3 MB)
- Pre-roll IMU + calibrated samples from trip start
- 180s pre-blackout GNSS history
- Warmup GNSS seed fix
- 100 ms replay batches (IMU + GNSS)
- Expected parity metrics (endpoint error m, drift %, gt_dist_m)
"""

from __future__ import annotations
import os
import sys
import json
import gzip
import time
from typing import Dict, Any, List
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader, TripSequence, IMUSample, GNSSSample
from sih.calibration.mount import MountAlignment, CalibratedSample, MountCalibrator, calibrate_stream
from sih.map.network import RoadNetwork, RoadSegment, load_trip_road_network
from sih.data.geo import geodetic_to_enu
from server.benchmark_setup import build_benchmark_session, CANONICAL_TARGET_ERRORS
from server.engine_adapter import EngineAdapterStageB

ASSETS_ONDEVICE_DIR = os.path.join(ROOT_DIR, "android", "app", "src", "ondevice", "assets")
ASSETS_MAIN_DIR = os.path.join(ROOT_DIR, "android", "app", "src", "main", "assets")


def crop_road_network(full_rnet: RoadNetwork, gnss_samples: List[GNSSSample], ref_lat: float, ref_lon: float, margin_m: float = 1000.0) -> RoadNetwork:
    """Crops road network to a corridor bounding box around route points + margin_m."""
    valid_gnss = [g for g in gnss_samples if g.is_valid]
    if not valid_gnss:
        return full_rnet
    pts = np.array([
        geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, ref_lat, ref_lon, 0.0)[:2]
        for g in valid_gnss
    ])
    min_e, max_e = float(np.min(pts[:, 0])) - margin_m, float(np.max(pts[:, 0])) + margin_m
    min_n, max_n = float(np.min(pts[:, 1])) - margin_m, float(np.max(pts[:, 1])) + margin_m

    cropped = RoadNetwork(cell_size_m=full_rnet.cell_size_m)
    for seg in full_rnet.segments:
        s_e, s_n = seg.start_enu_m
        e_e, e_n = seg.end_enu_m
        if (min_e <= s_e <= max_e and min_n <= s_n <= max_n) or (min_e <= e_e <= max_e and min_n <= e_n <= max_n):
            cropped.add_segment(seg)
    return cropped


def serialize_road_network(rnet: RoadNetwork) -> Dict[str, Any]:
    return {
        "cell_size_m": float(rnet.cell_size_m),
        "segments": [
            {
                "id": s.segment_id,
                "s_enu": [float(s.start_enu_m[0]), float(s.start_enu_m[1])],
                "e_enu": [float(s.end_enu_m[0]), float(s.end_enu_m[1])],
                "s_ll": [float(s.start_lat_lon[0]), float(s.start_lat_lon[1])],
                "e_ll": [float(s.end_lat_lon[0]), float(s.end_lat_lon[1])],
                "brg": float(s.bearing_deg),
                "len": float(s.length_m),
                "type": str(s.road_type),
                "spd": float(s.speed_limit_mps),
                "ow": bool(s.is_oneway),
                "sn": s.start_node_id,
                "en": s.end_node_id,
            }
            for s in rnet.segments
        ],
    }


def serialize_mount_alignment(align: MountAlignment) -> Dict[str, Any]:
    return {
        "is_calibrated": bool(align.is_calibrated),
        "quat": [float(x) for x in align.R_phone_to_vehicle.as_quat()],
        "matrix": [[float(val) for val in row] for row in align.R_phone_to_vehicle.as_matrix()],
        "forward_axis_phone": [float(x) for x in align.forward_axis_phone],
        "lateral_axis_phone": [float(x) for x in align.lateral_axis_phone],
        "vertical_axis_phone": [float(x) for x in align.vertical_axis_phone],
        "yaw_axis_index": int(align.yaw_axis_index),
        "yaw_axis_sign": float(align.yaw_axis_sign),
        "mount_yaw_offset_rad": float(align.mount_yaw_offset_rad),
        "pitch_deg": float(align.pitch_deg),
        "roll_deg": float(align.roll_deg),
        "state": "REUSED",
    }


def evaluate_bundle_error(sess: Any, rnet: RoadNetwork) -> Tuple[float, float]:
    """Runs a replay of the session through EngineAdapterStageB on CPU to compute exact expected error."""
    from sih.models.predictor import create_predictor
    predictor = create_predictor("torch", device="cpu")

    adapter = EngineAdapterStageB(
        reference_lat_deg=sess.reference_lat_deg,
        reference_lon_deg=sess.reference_lon_deg,
        reference_alt_m=sess.reference_alt_m,
        road_network=rnet,
        domain=sess.domain,
        saved_alignment=sess.saved_alignment,
        device="cpu",
        decimate_gnss_for_seeding=False,
        lock_saved_alignment=True,
        use_speed_smoother=True,
        predictor=predictor,
    )
    adapter.session.init_from_gnss(sess.warmup_gnss)
    adapter.prime_features(sess.preroll_imu, calib_samples=sess.preroll_calib)
    adapter.recent_gnss_window = list(sess.gnss_history)

    last_valid_gnss = sess.warmup_gnss
    stream_pts = []
    stream_ts = []
    for b in sess.batches:
        for g_d in b.get("gnss", []):
            g = GNSSSample(
                timestamp_ns=int(g_d["timestamp_ns"]),
                latitude_deg=float(g_d["latitude_deg"]),
                longitude_deg=float(g_d["longitude_deg"]),
                altitude_m=float(g_d.get("altitude_m", 0.0)),
                speed_mps=float(g_d["speed_mps"]) if g_d.get("speed_mps") is not None else None,
                bearing_deg=float(g_d["bearing_deg"]) if g_d.get("bearing_deg") is not None else None,
                accuracy_h_m=float(g_d.get("accuracy_h_m", 5.0)),
                is_valid=bool(g_d.get("is_valid", True)),
            )
            if not adapter.blackout_started:
                adapter.on_gnss(g)
                if g.is_valid:
                    last_valid_gnss = g

        req_state = b.get("state")
        if req_state == "BLACKOUT" and not adapter.blackout_started:
            adapter.set_blackout(True, entry_gnss=last_valid_gnss)

        for im_d in b.get("imu", []):
            im = IMUSample(
                timestamp_ns=int(im_d["timestamp_ns"]),
                accel=np.array(im_d["accel"], dtype=np.float64),
                gyro=np.array(im_d["gyro"], dtype=np.float64),
            )
            fused = adapter.on_imu(im)
            if adapter.blackout_started and fused is not None:
                stream_pts.append(adapter.ekf._p[:2].copy())
                stream_ts.append(im.timestamp_ns)

    bo_gnss = [g for g in sess.trip.gnss_samples if sess.bo_start_ns <= g.timestamp_ns <= sess.bo_end_ns and g.is_valid]
    gt_end = geodetic_to_enu(bo_gnss[-1].latitude_deg, bo_gnss[-1].longitude_deg, 0.0, sess.reference_lat_deg, sess.reference_lon_deg, 0.0)[:2]
    eval_t = float(bo_gnss[-1].timestamp_ns)
    stream_pts = np.array(stream_pts)
    stream_ts = np.array(stream_ts, dtype=np.float64)
    e = float(np.interp(eval_t, stream_ts, stream_pts[:, 0]))
    n = float(np.interp(eval_t, stream_ts, stream_pts[:, 1]))
    err = float(np.linalg.norm(np.array([e, n]) - gt_end))
    drift = (err / max(sess.gt_dist_m, 1.0)) * 100.0
    return err, drift


def export_benchmark_bundles(scenario_ids: List[int] = [1, 22, 23, 25, 26, 30]) -> None:
    os.makedirs(ASSETS_ONDEVICE_DIR, exist_ok=True)
    os.makedirs(ASSETS_MAIN_DIR, exist_ok=True)

    print("=" * 80)
    print("EXPORTING ON-DEVICE BENCHMARK BUNDLES (Gzip JSON <= 3 MB)")
    print("=" * 80)

    cached_trips: Dict[str, Any] = {}
    cached_networks: Dict[str, Any] = {}
    results = []

    for sc_id in scenario_ids:
        print(f"\nProcessing Scenario #{sc_id}...")
        sess = build_benchmark_session(scenario_id=sc_id, create_adapter=False)

        # Scenarios 22, 25 and 26 preserve trip road network to guarantee exact candidate ordering and parity (< 0.05m).
        # Scenarios 1, 23, 30 use corridor cropped road network (+1 km margin).
        if sc_id in (22, 25, 26):
            rnet = sess.road_network
            crop_mode = "full_trip"
        else:
            rnet = crop_road_network(sess.road_network, sess.sliced_trip.gnss_samples, sess.reference_lat_deg, sess.reference_lon_deg, margin_m=1000.0)
            crop_mode = "corridor_1km"

        # Evaluate expected metrics on laptop with this exact network
        err_m, drift_pct = evaluate_bundle_error(sess, rnet)
        expected_parity_target = CANONICAL_TARGET_ERRORS.get(sc_id, err_m)
        print(f"  Network mode: {crop_mode} ({len(rnet.segments)} segments)")
        print(f"  Laptop Replay Error: {err_m:.2f} m ({drift_pct:.2f}% drift) | Target: {expected_parity_target:.2f} m")

        # Build bundle dictionary
        bundle_data = {
            "scenario_id": sess.scenario_id,
            "trip": sess.trip_name,
            "domain": sess.domain,
            "reference_lat_deg": sess.reference_lat_deg,
            "reference_lon_deg": sess.reference_lon_deg,
            "reference_alt_m": sess.reference_alt_m,
            "bo_start_ns": sess.bo_start_ns,
            "bo_end_ns": sess.bo_end_ns,
            "bo_dur_s": sess.bo_dur_s,
            "warmup_start_ns": sess.warmup_start_ns,
            "warmup_dur_s": sess.warmup_dur_s,
            "saved_alignment": serialize_mount_alignment(sess.saved_alignment),
            "road_network": serialize_road_network(rnet),
            "warmup_gnss": {
                "t": int(sess.warmup_gnss.timestamp_ns),
                "lat": float(sess.warmup_gnss.latitude_deg),
                "lon": float(sess.warmup_gnss.longitude_deg),
                "alt": float(sess.warmup_gnss.altitude_m or 0.0),
                "spd": float(sess.warmup_gnss.speed_mps) if sess.warmup_gnss.speed_mps is not None else None,
                "brg": float(sess.warmup_gnss.bearing_deg) if sess.warmup_gnss.bearing_deg is not None else None,
                "acc": float(sess.warmup_gnss.accuracy_h_m or 5.0),
                "valid": bool(sess.warmup_gnss.is_valid),
            },
            "gnss_history": [
                {
                    "t": int(g.timestamp_ns),
                    "lat": float(g.latitude_deg),
                    "lon": float(g.longitude_deg),
                    "alt": float(g.altitude_m or 0.0),
                    "spd": float(g.speed_mps) if g.speed_mps is not None else None,
                    "brg": float(g.bearing_deg) if g.bearing_deg is not None else None,
                    "acc": float(g.accuracy_h_m or 5.0),
                    "valid": bool(g.is_valid),
                }
                for g in sess.gnss_history
            ],
            "preroll_imu": [
                {
                    "t": int(im.timestamp_ns),
                    "a": [float(v) for v in im.accel],
                    "g": [float(v) for v in im.gyro],
                }
                for im in sess.preroll_imu
            ],
            "preroll_calib": [
                {
                    "t": int(c.timestamp_ns),
                    "av": [float(v) for v in c.accel_vehicle],
                    "gv": [float(v) for v in c.gyro_vehicle],
                }
                for c in sess.preroll_calib
            ],
            "batches": sess.batches,
            "norm_mean": [float(x) for x in sess.norm_mean.flatten()] if sess.norm_mean is not None else [0.0] * 12,
            "norm_std": [float(x) for x in sess.norm_std.flatten()] if sess.norm_std is not None else [1.0] * 12,
            "expected": {
                "endpoint_error_m": round(float(err_m), 2),
                "drift_pct": round(float(drift_pct), 2),
                "gt_dist_m": round(float(sess.gt_dist_m), 1),
            },
        }

        # Compress to gzip JSON
        raw_json = json.dumps(bundle_data, separators=(",", ":")).encode("utf-8")
        gz_bytes = gzip.compress(raw_json, compresslevel=9)
        size_mb = len(gz_bytes) / 1024 / 1024

        filename = f"bench_{sc_id}.bin"
        out_ondevice = os.path.join(ASSETS_ONDEVICE_DIR, filename)
        out_main = os.path.join(ASSETS_MAIN_DIR, filename)

        with open(out_ondevice, "wb") as f:
            f.write(gz_bytes)
        with open(out_main, "wb") as f:
            f.write(gz_bytes)

        print(f"  -> Saved {out_ondevice}: {len(rnet.segments)} segments, {len(sess.batches)} batches, {size_mb:.2f} MB ({len(gz_bytes) / 1024:.1f} KB)")
        if size_mb > 3.0:
            raise RuntimeError(f"Bundle {filename} exceeded 3 MB limit: {size_mb:.2f} MB")

        results.append({
            "scenario_id": sc_id,
            "trip": sess.trip_name,
            "segments": len(rnet.segments),
            "size_kb": len(gz_bytes) / 1024,
            "size_mb": size_mb,
            "expected_err_m": err_m,
            "expected_drift_pct": drift_pct,
            "gt_dist_m": sess.gt_dist_m,
        })

    print("\n" + "=" * 80)
    print("BUNDLE EXPORT SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Scenario':<12} | {'Trip':<8} | {'Segments':<10} | {'Size':<12} | {'Expected Err':<14} | {'Expected Drift':<14}")
    print("-" * 80)
    for r in results:
        print(f"Scenario #{r['scenario_id']:<3} | {r['trip']:<8} | {r['segments']:<10} | {r['size_mb']:.2f} MB ({r['size_kb']:.0f}K) | {r['expected_err_m']:<6.2f} m       | {r['expected_drift_pct']:<5.2f} %")
    print("=" * 80)


if __name__ == "__main__":
    export_benchmark_bundles()
