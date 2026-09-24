"""
T3 - Sticky stop detector with hysteresis.

Problem: at a red light the engine idles, the phone vibrates, the AI speed model
reports a small creeping speed (0.5-1.5 m/s) and the strict physical ZUPT does
not fire. Over a 20-40 s stop this adds 10-40 m of along-track error.

Idea: a stop is a STATE, not a per-sample decision.
  Enter STOPPED  when (1 s mean) AI speed < v_enter, yaw is quiet and vibration
                 is at idle level, sustained t_enter.
  Stay STOPPED   until there is evidence of motion:
                 (a) launch: forward accel above the at-stop baseline, or
                 (b) AI speed above v_exit sustained t_exit.
Idle vibration level is learned from the pre-blackout history (GNSS speed < 0.3)
when available; otherwise a default gate is used.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from sih.round1.config import StopDetectorParams


class StopDetector:
    def __init__(self, p: StopDetectorParams, rate_hz: float = 10.0):
        self.p = p
        self.rate_hz = rate_hz
        n1 = max(3, int(round(rate_hz * 1.0)))
        n05 = max(2, int(round(rate_hz * 0.5)))
        self._acc_norm = deque(maxlen=n1)
        self._gyro = deque(maxlen=n1)
        self._v = deque(maxlen=n1)
        self._ax = deque(maxlen=n05)
        self.accel_std_gate = p.accel_std_default
        self.stopped = False
        self._enter_t = 0.0
        self._exit_t = 0.0
        self._ax_base: Optional[float] = None
        self.n_stops = 0
        self.stopped_time_s = 0.0

    # ---- learning from causal history -------------------------------------------------
    def learn_idle_level(self, gnss_ts_ns, gnss_speed, imu_ts_ns, accel) -> None:
        """Idle vibration std from history samples whose nearest GNSS speed < 0.3 m/s."""
        if len(gnss_ts_ns) < 3 or len(imu_ts_ns) < 20:
            return
        spd = np.interp(imu_ts_ns.astype(np.float64), gnss_ts_ns.astype(np.float64),
                        np.nan_to_num(gnss_speed, nan=99.0))
        norms = np.linalg.norm(accel, axis=1)
        idle = spd < 0.3
        n = int(self.rate_hz)
        stds = [float(np.std(norms[i - n:i])) for i in range(n, len(norms)) if np.all(idle[i - n:i])]
        if len(stds) >= int(3 * self.rate_hz):
            self.accel_std_gate = float(np.clip(self.p.accel_std_mult * np.median(stds), 0.05, 0.8))

    def reset(self, v_entry: float) -> None:
        self.stopped = v_entry < 0.3
        self._enter_t = self._exit_t = 0.0
        self._ax_base = None

    # ---- per step ---------------------------------------------------------------------
    def update(self, v_ai_scaled: float, accel_vehicle, gyro_vehicle, dt: float) -> bool:
        p = self.p
        self._acc_norm.append(float(np.linalg.norm(accel_vehicle)))
        self._gyro.append(float(np.linalg.norm(gyro_vehicle)))
        self._v.append(float(v_ai_scaled))
        self._ax.append(float(accel_vehicle[0]))
        if len(self._v) < self._v.maxlen:
            return self.stopped

        v_mean = float(np.mean(self._v))
        g_mean = float(np.mean(self._gyro))
        a_std = float(np.std(self._acc_norm))
        ax_mean = float(np.mean(self._ax))

        if not self.stopped:
            quiet = v_mean < p.v_enter_mps and g_mean < p.gyro_quiet_rad_s and a_std < self.accel_std_gate
            self._enter_t = self._enter_t + dt if quiet else 0.0
            if self._enter_t >= p.t_enter_s:
                self.stopped = True
                self.n_stops += 1
                self._exit_t = 0.0
                self._ax_base = ax_mean
        else:
            if self._ax_base is None:
                self._ax_base = ax_mean
            launch = (ax_mean - self._ax_base) > p.launch_accel_mps2
            fast = v_mean > p.v_exit_mps
            turning = g_mean > 2.0 * p.gyro_quiet_rad_s
            self._exit_t = self._exit_t + dt if (launch or fast or turning) else 0.0
            if launch and self._exit_t >= min(p.t_exit_s, 0.3) or self._exit_t >= p.t_exit_s:
                self.stopped = False
                self._enter_t = 0.0
            else:
                # slow drift of the at-rest accel baseline (gravity leakage from pitch)
                self._ax_base = 0.97 * self._ax_base + 0.03 * ax_mean
        if self.stopped:
            self.stopped_time_s += dt
        return self.stopped
