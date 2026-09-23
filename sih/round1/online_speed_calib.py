"""
T7 - Online per-speed-band AI speed self-calibration.

The existing pre-blackout scale uses one number (mean GNSS speed / mean AI speed
over ~15 s). The AI error is not a single gain: it is speed dependent (under-reads
at highway speed, over-reads when crawling). While GNSS is healthy we learn the
SHAPE of that error over the last history_s seconds:

  For consecutive windows of >= window_s seconds between GNSS fixes:
      d_gnss = path length through the fixes            (m)
      d_ai   = integral(v_ai_raw dt)                    (m)
      band   = d_ai / T  (mean AI speed of the window)
  r_band  = sum(d_gnss) / sum(d_ai)   per band
  r_all   = sum(d_gnss) / sum(d_ai)   over all windows
  f_band  = (T_band * (r_band / r_all) + T0 * 1) / (T_band + T0)     (ridge toward 1)

Blackout speed = v_ai * speed_scale * f(v_ai * speed_scale), with f linearly
interpolated between band centres. speed_scale (existing, recent 15 s) keeps the
level; f only adds the speed-dependent shape. Positions (not Doppler speed) are
used because the phone GNSS speed is stair-stepped.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from sih.round1.config import OnlineSpeedCalibParams
from sih.round1.history import PreBlackoutHistory, integrate_between


class BandSpeedCalibrator:
    def __init__(self, p: OnlineSpeedCalibParams):
        self.p = p
        edges = np.asarray(p.band_edges_mps, dtype=np.float64)
        self.edges = edges
        self.centres = 0.5 * (edges[:-1] + edges[1:])
        self.factors = np.ones(len(self.centres))
        self.band_seconds = np.zeros(len(self.centres))
        self.fitted = False
        self.info: Dict[str, object] = {}

    def fit(self, hist: PreBlackoutHistory) -> None:
        p = self.p
        ts = hist.gnss_ts_ns
        if len(ts) < 3 or hist.n_imu < 50:
            self.info = {"reason": "not enough history"}
            return
        w_ns = int(p.window_s * 1e9)
        seg = np.r_[0.0, np.linalg.norm(np.diff(hist.gnss_en, axis=0), axis=1)]
        cum = np.cumsum(seg)

        d_g: List[float] = []
        d_a: List[float] = []
        T: List[float] = []
        a = 0
        while a < len(ts) - 1:
            b = int(np.searchsorted(ts, ts[a] + w_ns))
            if b >= len(ts):
                break
            dur = (ts[b] - ts[a]) * 1e-9
            if dur > 3.0 * p.window_s:           # GNSS gap
                a = b
                continue
            dg = float(cum[b] - cum[a])
            da = integrate_between(hist.imu_ts_ns, hist.v_ai_raw, int(ts[a]), int(ts[b]))
            if dg >= p.min_window_dist_m and da > 1.0:
                d_g.append(dg)
                d_a.append(da)
                T.append(dur)
            a = b

        if len(d_g) < 3:
            self.info = {"reason": "too few windows", "n_windows": len(d_g)}
            return
        d_g_arr, d_a_arr, T_arr = map(np.asarray, (d_g, d_a, T))
        r_all = float(np.sum(d_g_arr) / np.sum(d_a_arr))
        v_band = d_a_arr / T_arr
        for k in range(len(self.centres)):
            m = (v_band >= self.edges[k]) & (v_band < self.edges[k + 1])
            if not np.any(m):
                continue
            r_b = float(np.sum(d_g_arr[m]) / np.sum(d_a_arr[m]))
            tb = float(np.sum(T_arr[m]))
            f = (tb * (r_b / r_all) + p.prior_s * 1.0) / (tb + p.prior_s)
            self.factors[k] = float(np.clip(f, p.clip_lo, p.clip_hi))
            self.band_seconds[k] = tb
        self.fitted = True
        self.info = {"r_all": r_all, "n_windows": len(d_g),
                     "factors": self.factors.round(4).tolist(),
                     "band_seconds": self.band_seconds.round(1).tolist()}

    def factor(self, v_scaled: float) -> float:
        if not self.fitted:
            return 1.0
        return float(np.interp(v_scaled, self.centres, self.factors))
