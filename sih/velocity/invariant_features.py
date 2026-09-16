"""Rotation-invariant, strictly causal IMU feature extraction.

Every channel produced here is a scalar built only from norms and dot products of the
measured device-frame vectors ``(a_lin, omega, g_hat)``. Under any rotation ``R`` applied
jointly to those three vectors -- which is exactly what re-mounting, pocketing or rotating
the phone does -- each channel is *mathematically unchanged*. The invariance is asserted
by ``engine/test_invariant_features.py`` against random SO(3) samples, not merely claimed.

Causality
---------
The legacy pipeline conditioned trips with ``scipy.signal.filtfilt``, a zero-phase
forward-backward filter that reads future samples. Offline metrics computed that way are
not attainable by a real-time system. Everything here uses one-sided (``lfilter``) or
exponential-moving-average smoothing, so a feature at time ``t`` depends only on samples
``<= t``.

Physical content
----------------
``v_centripetal`` deserves special mention. In a turn the horizontal specific force is
dominated by centripetal acceleration ``a_lat = v * omega_yaw``, so

    v ~= ||a_horizontal|| / |omega_yaw|

is a *direct, first-principles speed observation* that needs no learning and no mount
knowledge. It is supplied to the network as an explicit channel (with a validity flag)
rather than left for the network to rediscover through a division it cannot represent well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import signal

try:
    from sih.data.orientation import (
        GRAVITY,
        decompose_along_gravity,
        estimate_gravity_complementary,
        estimate_gyro_up_axis,
        gravity_direction,
        lean_angle_from_specific_force,
        unit,
    )
except ImportError:
    from engine.orientation import (
        GRAVITY,
        decompose_along_gravity,
        estimate_gravity_complementary,
        estimate_gyro_up_axis,
        gravity_direction,
        lean_angle_from_specific_force,
        unit,
    )

__all__ = ["INVARIANT_CHANNELS", "InvariantFeatureExtractor", "causal_lowpass", "causal_band_energy"]

INVARIANT_CHANNELS: tuple[str, ...] = (
    "a_mag",          # ||a_lin||
    "a_vert",         # a_lin . g_hat            (signed, up positive)
    "a_horiz",        # ||a_lin - (a_lin.g_hat) g_hat||
    "omega_yaw_abs",  # |omega . g_hat|          (yaw-rate magnitude about up)
    "omega_horiz",    # ||omega - (omega.g_hat) g_hat||   (pitch/roll rate: lean, bumps)
    "omega_mag",      # ||omega||
    "jerk_mag",       # ||d a_lin / dt||
    "sf_dev",         # ||f|| - g                (specific-force magnitude excess)
    "vib_low",        # causal 0.2-1.5 Hz energy of ||a_lin||   (chassis dynamics)
    "vib_high",       # causal 1.5-5.0 Hz energy of ||a_lin||   (road/tyre roughness)
    "tilt_rate",      # ||d g_hat / dt||         (how fast the vertical is moving)
    "lean_angle",     # arccos(g / ||f||)
    "v_centripetal",  # ||a_horiz|| / |omega_vert|   (first-principles speed cue)
    "v_cent_valid",   # 1 when the centripetal cue is observable
)


def causal_lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """One-sided Butterworth low-pass. Uses only past and present samples."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    nyq = 0.5 * fs
    wn = float(np.clip(cutoff_hz / nyq, 1e-4, 0.99))
    b, a = signal.butter(order, wn, btype="low", analog=False)
    zi = signal.lfilter_zi(b, a)
    init = x[0] if x.ndim == 1 else x[0]
    y, _ = signal.lfilter(b, a, x, axis=0, zi=zi * init)
    return y


def causal_band_energy(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    smooth_tau_s: float = 1.5,
    order: int = 2,
) -> np.ndarray:
    """Causal RMS energy of ``x`` inside ``[low_hz, high_hz]``.

    Band-passes with a one-sided filter, squares, then applies an exponential moving
    average. The EMA is a strictly causal replacement for the windowed Welch periodogram
    used previously (which, combined with ``filtfilt`` pre-conditioning, was not causal).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    nyq = 0.5 * fs
    lo = float(np.clip(low_hz / nyq, 1e-4, 0.98))
    hi = float(np.clip(high_hz / nyq, lo + 1e-4, 0.99))
    b, a = signal.butter(order, [lo, hi], btype="band", analog=False)
    y = signal.lfilter(b, a, x - float(np.mean(x[: min(len(x), 50)])), axis=0)

    alpha = float(np.clip(1.0 / max(smooth_tau_s * fs, 1.0), 1e-4, 1.0))
    p = np.square(y)
    # Exponential moving average as a first-order IIR recursion:
    #   ema[i] = (1 - alpha) * ema[i-1] + alpha * p[i]
    # Expressed through lfilter so the loop runs in C rather than Python.
    ema, _ = signal.lfilter([alpha], [1.0, -(1.0 - alpha)], p, zi=np.array([(1.0 - alpha) * p[0]]))
    return np.sqrt(np.maximum(ema, 0.0))


@dataclass
class InvariantFeatureExtractor:
    """Builds the ``INVARIANT_CHANNELS`` matrix from raw device-frame IMU streams."""

    sampling_rate: float = 10.0
    gravity_cutoff_hz: float = 0.4
    min_yaw_rate: float = 0.12
    max_centripetal_speed: float = 40.0
    gyro_frame_autocalibrate: bool = True

    @property
    def n_channels(self) -> int:
        return len(INVARIANT_CHANNELS)

    def compute_gravity(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        gravity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Device-frame up direction, from the gravity sensor when present."""
        if gravity is not None:
            g = np.asarray(gravity, dtype=np.float64)
            if g.shape == np.asarray(accel).shape and np.all(np.isfinite(g)):
                mag = np.linalg.norm(g, axis=-1)
                if float(np.median(mag)) > 1.0:
                    return gravity_direction(g)
        return estimate_gravity_complementary(accel, gyro, dt=1.0 / self.sampling_rate)

    def calibrated_yaw_rate(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        gravity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Signed yaw rate (rad/s) about the calibrated up axis of the gyroscope frame.

        Exposed separately from :meth:`extract` because heading propagation and time-sync
        estimation need the sign, while the learned model deliberately consumes only the
        magnitude.
        """
        accel = np.asarray(accel, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        g_hat = self.compute_gravity(accel, gyro, gravity)
        axis = np.mean(g_hat, axis=0)
        if self.gyro_frame_autocalibrate:
            cal = estimate_gyro_up_axis(gyro, accel_up=axis)
            if not cal.agrees_with_accel and cal.confident:
                axis = cal.axis
        return np.sum(gyro * unit(np.asarray(axis).reshape(3)), axis=-1)

    def __call__(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        gravity: Optional[np.ndarray] = None,
        linear_accel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.extract(accel, gyro, gravity, linear_accel)

    def extract(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        gravity: Optional[np.ndarray] = None,
        linear_accel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the ``(N, 14)`` invariant feature matrix.

        Args:
            accel: ``(N, 3)`` raw specific force (gravity included), m/s^2.
            gyro: ``(N, 3)`` angular rate, rad/s.
            gravity: optional ``(N, 3)`` gravity-sensor stream, m/s^2, pointing up.
            linear_accel: optional ``(N, 3)`` gravity-removed acceleration; derived as
                ``accel - gravity`` when omitted.
        """
        accel = np.asarray(accel, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        n = len(accel)
        fs = self.sampling_rate
        dt = 1.0 / fs

        g_hat = self.compute_gravity(accel, gyro, gravity)

        if linear_accel is not None:
            a_lin = np.asarray(linear_accel, dtype=np.float64)
        elif gravity is not None and np.asarray(gravity).shape == accel.shape:
            a_lin = accel - np.asarray(gravity, dtype=np.float64)
        else:
            g_mag = causal_lowpass(np.linalg.norm(accel, axis=-1), fs, self.gravity_cutoff_hz)
            a_lin = accel - g_hat * g_mag[:, None]

        a_vert, a_h_vec = decompose_along_gravity(a_lin, g_hat)
        a_horiz = np.linalg.norm(a_h_vec, axis=-1)

        # Indian Road Surface Shock Soft-Limiter (potholes, speed bumps, rumble strips)
        # For |a_vert| < 6 m/s^2, 12.0 * tanh(a_vert / 12.0) is nearly linear (~ a_vert).
        # For extreme road surface shocks (> 15-35 m/s^2), it smoothly limits to 12.0 m/s^2.
        a_vert_filtered = 12.0 * np.tanh(a_vert / 12.0)
        a_mag_shock_rej = np.sqrt(a_horiz**2 + a_vert_filtered**2)

        # The gyroscope triad is not guaranteed to be published in the accelerometer's
        # frame (see engine.orientation.estimate_gyro_up_axis). Resolve its own up axis so
        # that yaw/tilt separation stays correct either way; on a normal handset the two
        # axes coincide and this is a no-op.
        g_hat_gyro = g_hat
        if self.gyro_frame_autocalibrate:
            cal = estimate_gyro_up_axis(gyro, accel_up=np.mean(g_hat, axis=0))
            if not cal.agrees_with_accel and cal.confident:
                g_hat_gyro = np.tile(cal.axis, (n, 1))

        omega_vert, w_h_vec = decompose_along_gravity(gyro, g_hat_gyro)
        omega_horiz = np.linalg.norm(w_h_vec, axis=-1)
        omega_mag = np.linalg.norm(gyro, axis=-1)

        jerk = np.zeros(n)
        if n > 1:
            a_lin_filtered = a_h_vec + g_hat * a_vert_filtered[:, None]
            d = np.diff(a_lin_filtered, axis=0) / dt
            jerk[1:] = np.linalg.norm(d, axis=-1)
            jerk[0] = jerk[1]

        sf_mag = np.linalg.norm(accel, axis=-1)
        sf_smooth = causal_lowpass(sf_mag, fs, self.gravity_cutoff_hz)
        sf_dev = sf_smooth - GRAVITY

        vib_low = causal_band_energy(a_mag_shock_rej, fs, 0.2, 1.5)
        vib_high = causal_band_energy(a_mag_shock_rej, fs, 1.5, min(4.9, 0.49 * fs))

        tilt_rate = np.zeros(n)
        if n > 1:
            dg = np.linalg.norm(np.diff(g_hat, axis=0), axis=-1) / dt
            tilt_rate[1:] = dg
            tilt_rate[0] = tilt_rate[1]

        lean = lean_angle_from_specific_force(np.stack([sf_smooth, np.zeros(n), np.zeros(n)], axis=-1))

        a_h_s = causal_lowpass(a_horiz, fs, 1.0)
        w_v_s = causal_lowpass(omega_vert, fs, 1.0)
        # Only the magnitude of the yaw rate is exported. The eigenvector that defines the
        # gyroscope's up axis is sign-ambiguous, and a left turn and a right turn are
        # equally informative about speed, so taking the magnitude removes an arbitrary
        # per-trip sign flip at zero cost. Signed yaw for heading propagation is obtained
        # separately via engine.orientation.yaw_rate_about_gravity.
        omega_yaw_abs = np.abs(omega_vert)
        valid = np.abs(w_v_s) >= self.min_yaw_rate
        v_cent = np.zeros(n)
        denom = np.where(valid, np.abs(w_v_s), 1.0)
        v_cent = np.clip(a_h_s / np.maximum(denom, 1e-6), 0.0, self.max_centripetal_speed)
        v_cent = np.where(valid, v_cent, 0.0)

        feats = np.column_stack([
            a_mag_shock_rej, a_vert_filtered, a_horiz,
            omega_yaw_abs, omega_horiz, omega_mag,
            jerk, sf_dev,
            vib_low, vib_high,
            tilt_rate, lean,
            v_cent, valid.astype(np.float64),
        ])
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
