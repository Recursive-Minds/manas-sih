"""
Physical Kinematic Delta-v Complementary Speed Observer.

Fuses high-rate (10 Hz) longitudinal IMU specific force integration
with low-rate neural speed envelopes and physical rest ZUPT clamping.
Eliminates causal neural network phase lag, prevents open-loop accelerometer
bias runaway, and cleanly locks velocity to zero during vehicle stops.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from sih.core.contracts import CalibratedSample


class KinematicSpeedObserver:
    """
    Continuous physical speed observer combining forward accelerometer
    integration with neural velocity estimation and physical rest detection.
    """

    def __init__(
        self,
        alpha_base: float = 0.65,
        alpha_dynamic: float = 0.82,
        dynamic_accel_thresh: float = 0.35,
        bias_learning_rate: float = 0.001,
        max_accel_bias_mps2: float = 0.80,
        zupt_accel_var_thresh: float = 0.015,
        zupt_gnorm_err_thresh: float = 0.35,
        zupt_gyro_norm_thresh: float = 0.05,
    ):
        self.alpha_base = alpha_base
        self.alpha_dynamic = alpha_dynamic
        self.dynamic_accel_thresh = dynamic_accel_thresh
        self.beta = bias_learning_rate
        self.max_bias = max_accel_bias_mps2

        # Physical rest detection thresholds
        self.zupt_accel_var_thresh = zupt_accel_var_thresh
        self.zupt_gnorm_err_thresh = zupt_gnorm_err_thresh
        self.zupt_gyro_norm_thresh = zupt_gyro_norm_thresh

        # Observer state
        self._v_est: float = 0.0
        self._b_ax: float = 0.0
        self._last_ts: Optional[int] = None
        self._accel_buf: List[float] = []
        self._rest_count: int = 0
        self._initialised: bool = False

    def reset(self, initial_speed_mps: float = 0.0, initial_ts_ns: Optional[int] = None) -> None:
        """Initializes observer state at blackout entry or trip start."""
        self._v_est = max(0.0, float(initial_speed_mps))
        self._b_ax = 0.0
        self._last_ts = initial_ts_ns
        self._accel_buf.clear()
        self._rest_count = 0
        self._initialised = True

    def update(
        self,
        cal: CalibratedSample,
        v_ai_calibrated: float,
        dt_override: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """
        Observes forward velocity for the current calibrated IMU sample.

        Parameters
        ----------
        cal : CalibratedSample
            Leveled, mount-calibrated IMU sample.
        v_ai_calibrated : float
            Neural speed estimate scaled/offset by pre-blackout calibration.
        dt_override : float, optional
            Explicit delta-time in seconds if timestamps are irregular.

        Returns
        -------
        Tuple[float, bool]
            fused_speed_mps : Estimated forward velocity.
            is_stationary : True if vehicle is confirmed at physical rest.
        """
        ts = cal.timestamp_ns
        if not self._initialised:
            self.reset(initial_speed_mps=v_ai_calibrated, initial_ts_ns=ts)
            return self._v_est, False

        # Compute sampling delta-time
        if dt_override is not None and dt_override > 0.0:
            dt = float(dt_override)
        elif self._last_ts is not None:
            dt = (ts - self._last_ts) * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 0.10
        else:
            dt = 0.10
        self._last_ts = ts

        # Specific force and angular rate components
        a_x = float(cal.accel_vehicle[0])
        g_norm = float(np.linalg.norm(cal.accel_vehicle))
        w_norm = float(np.linalg.norm(cal.gyro_vehicle))

        # Sliding window physical rest detection
        self._accel_buf.append(g_norm)
        if len(self._accel_buf) > 20:
            self._accel_buf.pop(0)

        a_var = float(np.var(self._accel_buf)) if len(self._accel_buf) >= 10 else 1.0
        g_err = abs(g_norm - 9.80665)

        is_physical_rest = (
            a_var < self.zupt_accel_var_thresh
            and g_err < self.zupt_gnorm_err_thresh
            and w_norm < self.zupt_gyro_norm_thresh
        ) or (v_ai_calibrated < 0.25 and a_var < 0.035 and g_err < 0.50)

        if is_physical_rest:
            self._rest_count += 1
            self._v_est = 0.0
            # Learn resting longitudinal accelerometer bias
            self._b_ax = float(np.clip(0.95 * self._b_ax + 0.05 * a_x, -self.max_bias, self.max_bias))
            return 0.0, True

        self._rest_count = 0

        # Effective longitudinal acceleration
        a_eff = a_x - self._b_ax

        # Kinematic extrapolation
        v_kin = max(0.0, self._v_est + a_eff * dt)

        # Dynamic complementary blending:
        # High kinematic weighting during sharp throttle/braking transients
        # Blended weighting during steady cruise to maintain absolute scale anchoring
        if abs(a_eff) >= self.dynamic_accel_thresh:
            alpha = self.alpha_dynamic
        else:
            alpha = self.alpha_base

        v_fused = alpha * v_kin + (1.0 - alpha) * max(0.0, float(v_ai_calibrated))

        # Leaky bias update towards steady neural anchor
        diff_v = v_kin - max(0.0, float(v_ai_calibrated))
        self._b_ax += self.beta * diff_v
        self._b_ax = float(np.clip(self._b_ax, -self.max_bias, self.max_bias))

        self._v_est = max(0.0, v_fused)
        return self._v_est, False
