"""
Synthetic drive for round-1 unit tests (no dataset needed).

Truth is integrated from a piecewise (duration, v_start, v_end, turn_deg) profile.
Headings are clockwise from north; gyro_vehicle[2] is CCW-positive (as the EKF expects).
Accel follows the EKF usage: x forward, y left, z up (+g at rest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sih.core.contracts import IMUSample, GNSSSample, CalibratedSample
from sih.data.geo import enu_to_geodetic
from sih.map.network import RoadNetwork, RoadSegment, douglas_peucker_indices

REF_LAT, REF_LON = 12.97, 77.59
G = 9.80665

DEFAULT_PROFILE = [
    # history (GNSS healthy): three junction turns for gyro-scale learning
    (40, 10, 10, 0), (4, 10, 6, 0), (8, 6, 6, 90), (4, 6, 10, 0),
    (40, 10, 10, 0), (4, 10, 6, 0), (8, 6, 6, -90), (4, 6, 10, 0),
    (40, 10, 10, 0), (4, 10, 6, 0), (8, 6, 6, 90), (4, 6, 10, 0),
    (20, 10, 10, 0),          # blackout starts ~5 s into this leg (t = 185 s)
    (4, 10, 0, 0), (20, 0, 0, 0), (5, 0, 8, 0), (10, 8, 8, 0),
    (3, 8, 6, 0), (8, 6, 6, 90), (3, 6, 10, 0), (25, 10, 10, 0),
    (30, 10, 10, 0),
]


@dataclass
class SynthDrive:
    trip: object
    calib: List[CalibratedSample]
    v_true: np.ndarray
    v_ai: np.ndarray
    road: RoadNetwork
    pos: np.ndarray
    t_s: np.ndarray


@dataclass
class _Trip:
    trip_id: str
    imu_samples: list
    gnss_samples: list
    reference_lat_deg: float
    reference_lon_deg: float
    reference_alt_m: float = 0.0


def make_drive(profile=None, rate_hz: float = 10.0, ai_gain: float = 1.0, ai_creep_at_stop: float = 0.0,
               gyro_scale_err: float = 1.0, idle_vib_std: float = 0.25, road_vib_std: float = 0.4, seed: int = 0,
               start_heading_deg: float = 0.0) -> SynthDrive:
    rng = np.random.default_rng(seed)
    profile = profile or DEFAULT_PROFILE
    dt = 1.0 / rate_hz
    v_list, w_list = [], []
    for dur, v0, v1, turn in profile:
        n = int(round(dur * rate_hz))
        v_list.append(np.linspace(v0, v1, n, endpoint=False))
        w_list.append(np.full(n, np.radians(turn) / dur))   # clockwise yaw rate
    v = np.concatenate(v_list)
    w_cw = np.concatenate(w_list)
    n = len(v)
    t = np.arange(n) * dt
    hdg = np.radians(start_heading_deg) + np.concatenate([[0.0], np.cumsum(w_cw[:-1] * dt)])
    pos = np.zeros((n, 2))
    for i in range(1, n):
        pos[i] = pos[i - 1] + v[i - 1] * dt * np.array([np.sin(hdg[i - 1]), np.cos(hdg[i - 1])])

    a_x = np.gradient(v, dt)
    a_y = -v * w_cw
    stopped = v < 0.05
    calib, imu = [], []
    for i in range(n):
        ts = int(round(t[i] * 1e9)) + 1_000_000_000
        acc = np.array([a_x[i], a_y[i], G]) + rng.normal(0, idle_vib_std if stopped[i] else road_vib_std, 3)
        gyr = np.array([0.0, 0.0, -w_cw[i] * gyro_scale_err]) + rng.normal(0, 0.002, 3)
        imu.append(IMUSample(timestamp_ns=ts, accel=acc, gyro=gyr))
        calib.append(CalibratedSample(timestamp_ns=ts, accel_vehicle=acc, gyro_vehicle=gyr,
                                      rotation_body_to_vehicle=np.eye(3),
                                      gravity_vehicle=np.array([0.0, 0.0, 1.0]), is_calibrated=True))

    gnss = []
    for i in range(0, n, int(rate_hz)):
        lat, lon, _ = enu_to_geodetic(pos[i, 0], pos[i, 1], 0.0, REF_LAT, REF_LON, 0.0)
        gnss.append(GNSSSample(timestamp_ns=calib[i].timestamp_ns, latitude_deg=float(lat), longitude_deg=float(lon),
                               altitude_m=0.0, speed_mps=float(v[i]),
                               bearing_deg=float(np.degrees(hdg[i]) % 360.0) if v[i] > 0.5 else None,
                               accuracy_h_m=3.0, is_valid=True))

    v_ai = v * ai_gain + np.where(stopped, ai_creep_at_stop, 0.0) + rng.normal(0, 0.15, n)
    v_ai = np.clip(v_ai, 0.0, None).astype(np.float32)

    road = RoadNetwork()
    keep = douglas_peucker_indices(pos, 1.0)
    for k in range(len(keep) - 1):
        a, b = pos[keep[k]], pos[keep[k + 1]]
        L = float(np.linalg.norm(b - a))
        if L < 0.5:
            continue
        la, lo, _ = enu_to_geodetic(a[0], a[1], 0.0, REF_LAT, REF_LON, 0.0)
        lb, lob, _ = enu_to_geodetic(b[0], b[1], 0.0, REF_LAT, REF_LON, 0.0)
        road.add_segment(RoadSegment(
            segment_id=f"s{k}", start_enu_m=a.copy(), end_enu_m=b.copy(),
            start_lat_lon=(float(la), float(lo)), end_lat_lon=(float(lb), float(lob)),
            bearing_deg=float(np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 360.0), length_m=L,
            road_type="primary", start_node_id=f"n{k}", end_node_id=f"n{k+1}"))

    trip = _Trip("SYNTH", imu, gnss, REF_LAT, REF_LON)
    return SynthDrive(trip, calib, v, v_ai, road, pos, t)


def gnss_at(drive: SynthDrive, t_s: float) -> GNSSSample:
    return min(drive.trip.gnss_samples, key=lambda g: abs(g.timestamp_ns * 1e-9 - 1.0 - t_s))
