"""
T4 - Gyro yaw scale-factor estimation from pre-blackout GNSS turns.

Phone gyros carry a 1-3 % scale error (plus residual mount tilt that shrinks the
effective yaw rate). Over a 90 deg junction turn that is 1-3 deg of heading error,
which keeps growing as cross-track error for the rest of the blackout.

While GNSS is healthy we compare, over short windows that contain a real turn:
    d_psi_gnss = wrap(bearing(t + W) - bearing(t))          (clockwise, deg)
    d_psi_gyro = - integral(gyro_z dt) over the same window  (clockwise, deg)
and fit d_psi_gnss = s * d_psi_gyro by least squares through the origin:
    s_raw = sum(d_gnss * d_gyro) / sum(d_gyro^2)
Shrink toward 1.0 with a pseudo-count, then clip:
    s = 1 + (s_raw - 1) * n / (n + n0)
During blackout the yaw rate fed to the filters is s * gyro_z.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from sih.round1.config import GyroScaleParams
from sih.round1.history import PreBlackoutHistory, integrate_between


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def estimate_gyro_scale(hist: PreBlackoutHistory, p: GyroScaleParams) -> Dict[str, float]:
    out = {"scale": 1.0, "raw": float("nan"), "n_windows": 0}
    ok = (~np.isnan(hist.gnss_bearing)) & (~np.isnan(hist.gnss_speed)) & (hist.gnss_speed >= p.min_speed_mps)
    ts = hist.gnss_ts_ns
    if np.sum(ok) < 2 or hist.n_imu < 20:
        return out

    w_ns = int(p.window_s * 1e9)
    num = den = 0.0
    n = 0
    i = 0
    idx_ok = np.where(ok)[0]
    while i < len(idx_ok):
        a = idx_ok[i]
        # first valid fix at least W later
        later = idx_ok[(ts[idx_ok] >= ts[a] + w_ns)]
        if len(later) == 0:
            break
        b = later[0]
        if ts[b] - ts[a] > 2 * w_ns:           # gap in GNSS: skip ahead
            i = int(np.searchsorted(idx_ok, b))
            continue
        d_gnss = _wrap180(hist.gnss_bearing[b] - hist.gnss_bearing[a])
        d_gyro = -np.degrees(integrate_between(hist.imu_ts_ns, hist.gyro_z, int(ts[a]), int(ts[b])))
        if abs(d_gnss) >= p.min_turn_deg and abs(d_gyro) >= 0.5 * p.min_turn_deg \
                and np.sign(d_gnss) == np.sign(d_gyro) and abs(d_gyro) < 200.0:
            num += d_gnss * d_gyro
            den += d_gyro * d_gyro
            n += 1
            i = int(np.searchsorted(idx_ok, b))  # non-overlapping windows
        else:
            i += 1

    out["n_windows"] = n
    if n < p.min_windows or den <= 0.0:
        return out
    s_raw = num / den
    s = 1.0 + (s_raw - 1.0) * n / (n + p.prior_windows)
    out["raw"] = float(s_raw)
    out["scale"] = float(np.clip(s, p.clip_lo, p.clip_hi))
    return out
