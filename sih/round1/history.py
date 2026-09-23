"""
Causal pre-blackout history (strictly t < blackout start).

Real phone equivalent: a ring buffer filled while GNSS is healthy. In the batch
benchmark we slice the same information from the trip, never past bo_start_ns.
No CAN / ground-truth speed is used here, only phone GNSS + IMU + AI speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from sih.data.geo import geodetic_to_enu


@dataclass
class PreBlackoutHistory:
    imu_ts_ns: np.ndarray      # (N,) int64
    gyro_z: np.ndarray         # (N,) rad/s, vehicle yaw rate (CCW positive)
    accel: np.ndarray          # (N, 3) vehicle-frame specific force
    v_ai_raw: np.ndarray       # (N,) unscaled AI speed
    gnss_ts_ns: np.ndarray     # (M,) valid fixes only
    gnss_en: np.ndarray        # (M, 2) ENU east/north (m)
    gnss_speed: np.ndarray     # (M,) m/s (nan if missing)
    gnss_bearing: np.ndarray   # (M,) deg clockwise from north (nan if missing)

    @property
    def n_imu(self) -> int:
        return int(len(self.imu_ts_ns))


def build_pre_blackout_history(
    trip: Any,
    calib_samples: List[Any],
    v_preds: np.ndarray,
    bo_start_ns: int,
    history_s: float,
) -> PreBlackoutHistory:
    t_lo = bo_start_ns - int(history_s * 1e9)
    imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.int64)
    idx = np.where((imu_ts >= t_lo) & (imu_ts < bo_start_ns))[0]
    idx = idx[idx < min(len(calib_samples), len(v_preds))]

    gyro_z = np.array([calib_samples[i].gyro_vehicle[2] for i in idx], dtype=np.float64)
    accel = (np.array([calib_samples[i].accel_vehicle for i in idx], dtype=np.float64)
             if len(idx) else np.zeros((0, 3)))
    v_ai = np.asarray(v_preds, dtype=np.float64)[idx] if len(idx) else np.zeros(0)

    fixes = [g for g in trip.gnss_samples if g.is_valid and t_lo <= g.timestamp_ns < bo_start_ns]
    g_ts = np.array([g.timestamp_ns for g in fixes], dtype=np.int64)
    g_en = np.array(
        [geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0,
                         trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2] for g in fixes],
        dtype=np.float64,
    ).reshape(-1, 2)
    g_spd = np.array([np.nan if g.speed_mps is None else g.speed_mps for g in fixes], dtype=np.float64)
    g_brg = np.array([np.nan if g.bearing_deg is None else g.bearing_deg for g in fixes], dtype=np.float64)

    return PreBlackoutHistory(
        imu_ts_ns=imu_ts[idx], gyro_z=gyro_z, accel=accel, v_ai_raw=v_ai,
        gnss_ts_ns=g_ts, gnss_en=g_en, gnss_speed=g_spd, gnss_bearing=g_brg,
    )


def integrate_between(ts_ns: np.ndarray, values: np.ndarray, t0_ns: int, t1_ns: int) -> float:
    """Rectangle-rule integral of a 10 Hz signal over (t0, t1]."""
    if len(ts_ns) < 2:
        return 0.0
    m = (ts_ns > t0_ns) & (ts_ns <= t1_ns)
    if not np.any(m):
        return 0.0
    dt = np.diff(ts_ns, prepend=ts_ns[0]).astype(np.float64) * 1e-9
    dt[0] = 0.1
    dt = np.clip(dt, 0.0, 1.0)
    return float(np.sum(values[m] * dt[m]))
