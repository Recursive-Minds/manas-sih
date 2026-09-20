"""
server/replay.py
----------------
Trip Replay CLI and Test Runner for Smartphone IDR.

Replays IO-VNBD datasets or phone sensor logs through the server router,
enabling automated benchmark verification, scenario re-simulation, and
visual inspection on the web dashboard.

Features:
- Configurable playback rate (1.0x real-time, 5.0x, or 0 = maximum CPU throughput).
- Supports both live network streaming (WebSocket to server/router.py) and
  direct in-memory pipeline injection for unit tests.
- Precise GNSS blackout injection with exact timestamps.
"""

from __future__ import annotations
import os
import sys
import time
import json
import asyncio
import argparse
from typing import Optional, Dict, Any, List
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader, TripSequence
from sih.core.contracts import IMUSample, GNSSSample


def build_sensor_batches(
    trip: TripSequence,
    batch_interval_s: float = 0.10,
    blackout_start_s: float = 30.0,
    blackout_duration_s: float = 45.0,
) -> List[Dict[str, Any]]:
    """
    Chunks a TripSequence into chronological 100 ms JSON-serializable sensor batches.
    Labels each batch as 'WARMING_UP' or 'BLACKOUT'.
    """
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(blackout_start_s * 1e9)
    bo_end_ns = bo_start_ns + int(blackout_duration_s * 1e9)

    batch_interval_ns = int(batch_interval_s * 1e9)
    batches: List[Dict[str, Any]] = []

    g_idx = 0
    n_g = len(trip.gnss_samples)
    i_idx = 0
    n_i = len(trip.imu_samples)

    curr_window_start_ns = t0_ns

    while i_idx < n_i:
        curr_window_end_ns = curr_window_start_ns + batch_interval_ns
        imu_batch = []
        while i_idx < n_i and trip.imu_samples[i_idx].timestamp_ns < curr_window_end_ns:
            im = trip.imu_samples[i_idx]
            imu_batch.append({
                "timestamp_ns": int(im.timestamp_ns),
                "accel": [float(v) for v in im.accel],
                "gyro": [float(v) for v in im.gyro],
            })
            i_idx += 1

        gnss_batch = []
        while g_idx < n_g and trip.gnss_samples[g_idx].timestamp_ns < curr_window_end_ns:
            g = trip.gnss_samples[g_idx]
            gnss_batch.append({
                "timestamp_ns": int(g.timestamp_ns),
                "latitude_deg": float(g.latitude_deg),
                "longitude_deg": float(g.longitude_deg),
                "altitude_m": float(g.altitude_m or 0.0),
                "speed_mps": float(g.speed_mps) if g.speed_mps is not None else None,
                "bearing_deg": float(g.bearing_deg) if g.bearing_deg is not None else None,
                "accuracy_h_m": float(g.accuracy_h_m or 5.0),
                "is_valid": bool(g.is_valid),
            })
            g_idx += 1

        is_bo = (curr_window_start_ns >= bo_start_ns and curr_window_start_ns <= bo_end_ns)
        state_str = "BLACKOUT" if is_bo else "WARMING_UP"

        if imu_batch or gnss_batch:
            batches.append({
                "type": "sensor_batch",
                "state": state_str,
                "timestamp_ns": curr_window_start_ns,
                "imu": imu_batch,
                "gnss": gnss_batch,
            })

        curr_window_start_ns = curr_window_end_ns

    return batches


async def replay_to_server(
    batches: List[Dict[str, Any]],
    ws_url: str = "ws://localhost:8765/ws/stream",
    speed: float = 1.0,
    verbose: bool = True,
) -> None:
    """Streams sensor batches over WebSocket to router."""
    import aiohttp

    if verbose:
        print(f"Connecting to replay server at {ws_url} ...")
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            if verbose:
                print(f"Connected. Streaming {len(batches)} batches at {speed}x speed ...")

            t_prev = time.time()
            for b_idx, batch in enumerate(batches):
                await ws.send_str(json.dumps(batch))

                if speed > 0.0:
                    dt_target = 0.10 / speed
                    elapsed = time.time() - t_prev
                    sleep_time = max(0.0, dt_target - elapsed)
                    if sleep_time > 0.001:
                        await asyncio.sleep(sleep_time)
                    t_prev = time.time()

                if verbose and (b_idx % 50 == 0 or b_idx == len(batches) - 1):
                    print(f"  [Replay] Progress: {b_idx + 1}/{len(batches)} batches ({batch['state']})")

    if verbose:
        print("Replay completed successfully.")


def replay_direct_in_memory(
    trip: TripSequence,
    router: Any,
    blackout_start_s: float = 30.0,
    blackout_duration_s: float = 45.0,
) -> Dict[str, Any]:
    """
    Synchronously feeds trip batches directly into a NavigationRouter instance in-memory.
    Ideal for rapid, deterministic unit testing.
    """
    batches = build_sensor_batches(
        trip=trip,
        batch_interval_s=0.10,
        blackout_start_s=blackout_start_s,
        blackout_duration_s=blackout_duration_s,
    )
    for batch in batches:
        # Direct synchronous call on engine and evaluator
        is_bo = (batch["state"] == "BLACKOUT")
        if is_bo and router.state != "BLACKOUT":
            router.set_blackout(True)
        elif not is_bo and router.state != "WARMING_UP":
            router.set_blackout(False)

        for g_dict in batch.get("gnss", []):
            gnss = GNSSSample(
                timestamp_ns=int(g_dict["timestamp_ns"]),
                latitude_deg=float(g_dict["latitude_deg"]),
                longitude_deg=float(g_dict["longitude_deg"]),
                altitude_m=float(g_dict.get("altitude_m", 0.0)),
                speed_mps=float(g_dict["speed_mps"]) if g_dict.get("speed_mps") is not None else None,
                bearing_deg=float(g_dict["bearing_deg"]) if g_dict.get("bearing_deg") is not None else None,
                accuracy_h_m=float(g_dict.get("accuracy_h_m", 5.0)),
                is_valid=bool(g_dict.get("is_valid", True)),
            )
            router.evaluator.on_gnss(gnss)
            if not is_bo:
                router.engine.on_gnss(gnss)

        for im_dict in batch.get("imu", []):
            imu = IMUSample(
                timestamp_ns=int(im_dict["timestamp_ns"]),
                accel=np.array(im_dict["accel"], dtype=np.float64),
                gyro=np.array(im_dict["gyro"], dtype=np.float64),
            )
            fused = router.engine.on_imu(imu)
            if fused is not None:
                router.evaluator.on_dr(fused)

    return router.evaluator.get_summary_dict()


def load_phone_csv(file_path: str, trip_id: Optional[str] = None) -> TripSequence:
    """
    Parses a raw CSV recorded directly on an Android smartphone by SensorStreamService.
    Format:
      type,timestamp_ns,val1,val2,val3,val4,val5,val6
      IMU,ts,ax,ay,az,gx,gy,gz
      GNSS,ts,lat,lon,alt,spd,brg,acc
    """
    import csv
    from sih.data.geo import compute_cumulative_distance

    imu_samples: List[IMUSample] = []
    gnss_samples: List[GNSSSample] = []
    tid = trip_id or os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 7:
                continue
            row_type = row[0].strip().upper()
            try:
                ts = int(row[1].strip())
                if row_type == "IMU":
                    ax = float(row[2])
                    ay = float(row[3])
                    az = float(row[4])
                    gx = float(row[5])
                    gy = float(row[6])
                    gz = float(row[7])
                    imu_samples.append(IMUSample(
                        timestamp_ns=ts,
                        accel=np.array([ax, ay, az], dtype=np.float64),
                        gyro=np.array([gx, gy, gz], dtype=np.float64),
                    ))
                elif row_type == "GNSS":
                    lat = float(row[2])
                    lon = float(row[3])
                    alt = float(row[4])
                    spd = float(row[5]) if float(row[5]) != 0.0 or len(row) > 5 else None
                    brg = float(row[6]) if float(row[6]) != 0.0 or len(row) > 6 else None
                    acc = float(row[7]) if len(row) > 7 else 5.0
                    gnss_samples.append(GNSSSample(
                        timestamp_ns=ts,
                        latitude_deg=lat,
                        longitude_deg=lon,
                        altitude_m=alt,
                        speed_mps=spd,
                        bearing_deg=brg,
                        accuracy_h_m=acc,
                        is_valid=(lat != 0.0 or lon != 0.0),
                    ))
            except (ValueError, IndexError):
                continue

    imu_samples.sort(key=lambda s: s.timestamp_ns)
    gnss_samples.sort(key=lambda s: s.timestamp_ns)

    ref_lat = gnss_samples[0].latitude_deg if gnss_samples else 0.0
    ref_lon = gnss_samples[0].longitude_deg if gnss_samples else 0.0
    ref_alt = gnss_samples[0].altitude_m or 0.0 if gnss_samples else 0.0
    valid_gnss = [g for g in gnss_samples if g.is_valid]
    if len(valid_gnss) >= 2:
        lats = np.array([g.latitude_deg for g in valid_gnss], dtype=np.float64)
        lons = np.array([g.longitude_deg for g in valid_gnss], dtype=np.float64)
        cum_dist = compute_cumulative_distance(lats, lons)
    else:
        cum_dist = 0.0
    dur = 0.0
    if imu_samples:
        dur = (imu_samples[-1].timestamp_ns - imu_samples[0].timestamp_ns) * 1e-9

    return TripSequence(
        trip_id=tid,
        imu_samples=imu_samples,
        gnss_samples=gnss_samples,
        reference_lat_deg=ref_lat,
        reference_lon_deg=ref_lon,
        reference_alt_m=ref_alt,
        total_gnss_distance_m=cum_dist,
        duration_s=dur,
    )


def load_any_trip(file_path: str, trip_id: Optional[str] = None) -> TripSequence:
    """Loads either a phone-logged CSV or a canonical IO-VNBD dataset CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trip file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().lower()

    if first_line.startswith("type,timestamp_ns") or "val1" in first_line:
        return load_phone_csv(file_path, trip_id=trip_id)
    return GenericDataLoader().load_file(file_path, trip_id=trip_id)


def main():
    parser = argparse.ArgumentParser(description="Trip Replay Streamer for Smartphone IDR")
    parser.add_argument("--trip", default="S-M", help="Trip name (e.g. S-M, S-S1) or CSV path")
    parser.add_argument("--blackout-start", type=float, default=30.0, help="Blackout start offset (seconds)")
    parser.add_argument("--blackout-duration", type=float, default=45.0, help="Blackout duration (seconds)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (0 = max speed)")
    parser.add_argument("--server", default="ws://localhost:8765/ws/stream", help="WebSocket server URL")
    args = parser.parse_args()

    trip_path = args.trip
    if not os.path.exists(trip_path):
        cand = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", f"{args.trip}.csv")
        if os.path.exists(cand):
            trip_path = cand
        else:
            raise FileNotFoundError(f"Could not find trip CSV: {args.trip}")

    print(f"Loading trip from {trip_path} ...")
    trip = load_any_trip(trip_path)
    print(f"Loaded {len(trip.imu_samples)} IMU samples and {len(trip.gnss_samples)} GNSS fixes.")

    batches = build_sensor_batches(
        trip=trip,
        batch_interval_s=0.10,
        blackout_start_s=args.blackout_start,
        blackout_duration_s=args.blackout_duration,
    )
    print(f"Built {len(batches)} 100 ms batches.")

    asyncio.run(replay_to_server(batches, ws_url=args.server, speed=args.speed))


if __name__ == "__main__":
    main()
