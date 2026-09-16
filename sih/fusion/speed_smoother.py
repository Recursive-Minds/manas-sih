"""
Causal Kinematic Speed Smoother.

Applies physical vehicle acceleration limits and causal low-pass filtering:
1. Slew-Rate Limiting:
   - Longitudinal acceleration capped to +3.5 m/s^2 (+1.26 km/h per 0.1s tick).
   - Normal braking deceleration capped to -5.0 m/s^2 (-1.80 km/h per 0.1s tick).
2. Causal 1st-Order Exponential Moving Average (EMA):
   - Time constant tau = 0.25s (alpha = 0.286 at 10 Hz).
   - Eliminates high-frequency chassis vibration noise and Bayesian MoE gate toggling
     without introducing observable phase lag.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class CausalSpeedSmoother:
    """
    Causal Kinematic Slew-Rate Limiter and 1st-Order Low-Pass Filter.
    Zero future lookahead, strictly causal, suitable for streaming and edge deployment.
    """

    def __init__(
        self,
        a_max_mps2: float = 3.5,
        a_min_mps2: float = -5.0,
        tau_s: float = 0.25,
    ) -> None:
        self.a_max = float(a_max_mps2)
        self.a_min = float(a_min_mps2)
        self.tau = float(tau_s)
        self._v_prev: Optional[float] = None

    def reset(self) -> None:
        self._v_prev = None

    def update(self, v_raw: float, dt_s: float = 0.1) -> float:
        """
        Process a single streaming speed sample causally.
        """
        v_nonneg = max(0.0, float(v_raw))
        if self._v_prev is None:
            self._v_prev = v_nonneg
            return v_nonneg

        dt = max(1e-4, float(dt_s))
        # 1. Acceleration slew-rate limiting
        v_clamped = float(np.clip(v_nonneg, self._v_prev + self.a_min * dt, self._v_prev + self.a_max * dt))

        # 2. Causal 1st-order low-pass EMA filter
        alpha = dt / (self.tau + dt)
        v_smooth = (1.0 - alpha) * self._v_prev + alpha * v_clamped
        v_out = max(0.0, float(v_smooth))
        self._v_prev = v_out
        return v_out

    step = update

    def filter_sequence(self, speeds: np.ndarray, dt_s: float = 0.1) -> np.ndarray:
        """
        Filter a continuous 1D array of velocity estimates.
        """
        if len(speeds) == 0:
            return np.array([], dtype=np.float32)

        out = np.zeros(len(speeds), dtype=np.float32)
        v_p = max(0.0, float(speeds[0]))
        dt = max(1e-4, float(dt_s))
        alpha = dt / (self.tau + dt)

        for i in range(len(speeds)):
            v_raw = max(0.0, float(speeds[i]))
            v_clamped = float(np.clip(v_raw, v_p + self.a_min * dt, v_p + self.a_max * dt))
            v_s = (1.0 - alpha) * v_p + alpha * v_clamped
            v_out = max(0.0, float(v_s))
            out[i] = v_out
            v_p = v_out

        return out
