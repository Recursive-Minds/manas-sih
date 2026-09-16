"""Gravity-referenced attitude utilities for mount-invariant inertial processing.

Why gravity is the right reference
----------------------------------
A smartphone can be mounted, held, pocketed or knocked into an arbitrary orientation.
Let ``R`` be the (unknown, possibly time-varying) rotation from the device frame to the
vehicle frame. Every vector the device measures rotates together:

    a_device = R^T a_vehicle ,  omega_device = R^T omega_vehicle ,  g_device = R^T g_vehicle

Consequently any quantity built ONLY from dot products, norms and cross-product norms of
measured vectors is *exactly* invariant to ``R``:

    ||a||            -> ||R^T a||          = ||a||
    a . g_hat        -> (R^T a).(R^T g_hat) = a . g_hat
    ||a - (a.g_hat) g_hat|| -> unchanged

Gravity supplies the one physically privileged direction the device can always observe,
so it converts an under-determined 3-DOF frame problem into a fully determined 1-DOF one:
vertical and yaw quantities become mount independent, and only the heading of the
horizontal axes (forward vs lateral) remains unknown. That residual degree of freedom is
handled separately by :class:`MountFrameEstimator`.

Sign convention
---------------
Following the Android ``TYPE_GRAVITY`` / accelerometer convention, ``gravity`` points
**up** (a device at rest, flat and face up, reports ``+9.81`` on its z axis). ``g_hat`` is
therefore the device-frame *up* direction, so ``a . g_hat > 0`` means accelerating upward
and ``omega . g_hat > 0`` is a left turn by the right-hand rule.

Effective gravity and lean
--------------------------
An accelerometer cannot distinguish gravity from sustained acceleration; it senses
specific force. During a *coordinated* turn (the condition a two-wheeler holds by
construction, and a car approximates) the low-passed specific force aligns with the
vehicle's own vertical and has magnitude

    ||f|| = sqrt(g^2 + a_c^2) = g / cos(phi),      tan(phi) = a_c / g

with ``phi`` the lean angle and ``a_c = v * omega_yaw`` the centripetal acceleration.
This yields three exact, testable relations used below:

    phi        = arccos( g / ||f|| )                        (lean from force magnitude)
    omega_yaw  = (omega . f_hat) / cos(phi)                 (de-leaned yaw rate)
    v          = g * tan(phi) / omega_yaw                    (two-wheeler speed from lean)

They are derived, not fitted, and are covered by synthetic unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "GRAVITY",
    "unit",
    "gravity_direction",
    "estimate_gravity_complementary",
    "decompose_along_gravity",
    "vertical_component",
    "horizontal_component",
    "yaw_rate_about_gravity",
    "lean_angle_from_specific_force",
    "deleaned_yaw_rate",
    "two_wheeler_speed_from_lean",
    "MountFrameEstimator",
    "estimate_gyro_up_axis",
    "GyroFrameCalibration",
    "YawRateCalibration",
    "fit_yaw_rate_from_reference",
    "reference_yaw_rate_from_positions",
]

GRAVITY = 9.80665


def unit(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """Normalise vectors along ``axis``, leaving near-zero vectors untouched in direction."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def gravity_direction(gravity: np.ndarray) -> np.ndarray:
    """Unit *up* direction in the device frame from a gravity-sensor stream."""
    return unit(np.asarray(gravity, dtype=np.float64))


def estimate_gravity_complementary(
    accel: np.ndarray,
    gyro: np.ndarray,
    dt: float = 0.1,
    tau: float = 4.0,
    accel_gate: float = 1.5,
    init_samples: int = 20,
) -> np.ndarray:
    """Estimate the device-frame up direction without a dedicated gravity sensor.

    Propagates the up direction with the gyroscope and corrects it towards the measured
    specific force only when that force is consistent with gravity
    (``| ||f|| - g | < accel_gate``). Gating is what stops sustained cornering, braking
    or a pothole from dragging the vertical reference off true, which is the classic
    failure of a plain low-pass "gravity" estimate.

    Args:
        accel: ``(N, 3)`` raw accelerometer specific force in m/s^2 (gravity included).
        gyro: ``(N, 3)`` angular rate in rad/s.
        dt: sample period in seconds.
        tau: complementary-filter time constant in seconds.
        accel_gate: tolerance in m/s^2 for trusting the accelerometer as a vertical cue.
        init_samples: samples averaged to initialise the vertical.

    Returns:
        ``(N, 3)`` unit up-direction estimates in the device frame.
    """
    accel = np.asarray(accel, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    n = len(accel)
    out = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return out

    k = max(1, min(init_samples, n))
    g_hat = unit(np.mean(accel[:k], axis=0))
    if not np.all(np.isfinite(g_hat)) or np.linalg.norm(g_hat) < 0.5:
        g_hat = np.array([0.0, 0.0, 1.0])

    alpha = float(dt / max(tau, dt))
    for i in range(n):
        w = gyro[i]
        # Rotate the up direction into the new device frame: the device turned by
        # +w*dt, so a fixed world direction moves by -w*dt within the device frame.
        g_hat = g_hat - np.cross(w, g_hat) * dt
        norm = np.linalg.norm(g_hat)
        g_hat = g_hat / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        f = accel[i]
        f_norm = float(np.linalg.norm(f))
        if abs(f_norm - GRAVITY) < accel_gate and f_norm > 1e-6:
            g_hat = unit((1.0 - alpha) * g_hat + alpha * (f / f_norm))
        out[i] = g_hat

    return out


def decompose_along_gravity(
    vec: np.ndarray,
    g_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``vec`` into its component along ``g_hat`` and the remaining horizontal vector.

    Both outputs are exactly invariant to any rotation applied jointly to ``vec`` and
    ``g_hat`` (the scalar is a dot product; the horizontal part rotates with the frame and
    is only ever consumed through its norm).
    """
    vec = np.atleast_2d(np.asarray(vec, dtype=np.float64))
    g_hat = np.atleast_2d(np.asarray(g_hat, dtype=np.float64))
    g_hat = unit(g_hat)
    along = np.sum(vec * g_hat, axis=-1)
    horizontal = vec - along[:, None] * g_hat
    return along, horizontal


def vertical_component(vec: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Signed component of ``vec`` along the up direction. Rotation invariant."""
    along, _ = decompose_along_gravity(vec, g_hat)
    return along


def horizontal_component(vec: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Magnitude of the part of ``vec`` perpendicular to gravity. Rotation invariant."""
    _, horiz = decompose_along_gravity(vec, g_hat)
    return np.linalg.norm(horiz, axis=-1)


def yaw_rate_about_gravity(gyro: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Angular rate about the up axis (rad/s), signed, mount invariant."""
    return vertical_component(gyro, g_hat)


def lean_angle_from_specific_force(
    specific_force: np.ndarray,
    g: float = GRAVITY,
    max_angle_rad: float = np.radians(55.0),
) -> np.ndarray:
    """Coordinated-turn lean angle from the magnitude of the low-passed specific force.

    ``phi = arccos(g / ||f||)``, clamped to ``[0, max_angle_rad]``. Values of ``||f||``
    below ``g`` (free-fall side, or noise) map to zero lean.
    """
    f = np.atleast_2d(np.asarray(specific_force, dtype=np.float64))
    mag = np.linalg.norm(f, axis=-1)
    ratio = np.clip(g / np.maximum(mag, 1e-6), -1.0, 1.0)
    phi = np.arccos(ratio)
    return np.clip(np.nan_to_num(phi), 0.0, max_angle_rad)


def deleaned_yaw_rate(
    gyro: np.ndarray,
    g_hat: np.ndarray,
    lean_angle: np.ndarray,
    min_cos: float = 0.5,
) -> np.ndarray:
    """True yaw rate about the *earth* vertical, given rotation about a leaned vertical.

    On a leaning vehicle the measured up axis is tilted by ``phi`` from true vertical, so
    the observed rate is ``omega_yaw * cos(phi)``. Dividing restores the earth-referenced
    yaw rate. ``min_cos`` bounds the amplification at extreme lean.
    """
    measured = yaw_rate_about_gravity(gyro, g_hat)
    cos_phi = np.maximum(np.cos(np.asarray(lean_angle, dtype=np.float64)), min_cos)
    return measured / cos_phi


def two_wheeler_speed_from_lean(
    lean_angle: np.ndarray,
    yaw_rate: np.ndarray,
    g: float = GRAVITY,
    min_yaw_rate: float = 0.08,
    max_speed: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Independent speed estimate for a coordinated two-wheeler turn.

    From ``tan(phi) = v * omega / g``:  ``v = g tan(phi) / omega``.

    Returns ``(speed, valid)``. The relation is singular as ``omega -> 0`` (a straight
    line carries no lean information), so samples below ``min_yaw_rate`` are marked
    invalid rather than being allowed to blow up.
    """
    phi = np.asarray(lean_angle, dtype=np.float64)
    omega = np.asarray(yaw_rate, dtype=np.float64)
    valid = np.abs(omega) >= min_yaw_rate
    speed = np.zeros_like(phi)
    safe = np.where(valid, np.abs(omega), 1.0)
    speed = g * np.tan(phi) / safe
    speed = np.clip(np.nan_to_num(speed), 0.0, max_speed)
    valid &= np.isfinite(speed) & (phi > np.radians(2.0))
    return speed, valid


@dataclass
class GyroFrameCalibration:
    """Result of resolving which gyroscope direction corresponds to "up"."""

    axis: np.ndarray
    eig_ratio: float
    agrees_with_accel: bool
    used_fallback: bool

    @property
    def confident(self) -> bool:
        return (not self.used_fallback) and self.eig_ratio >= 2.0


def estimate_gyro_up_axis(
    gyro: np.ndarray,
    accel_up: Optional[np.ndarray] = None,
    min_eig_ratio: float = 2.0,
    agreement_deg: float = 25.0,
) -> GyroFrameCalibration:
    """Find the gyroscope-frame direction about which a ground vehicle mainly rotates.

    On any wheeled vehicle the angular-velocity budget is dominated by *yaw* about the
    local vertical, so the principal axis of the gyroscope covariance is the vertical
    expressed in the gyroscope's own frame. Recovering it this way needs no accelerometer
    and no reference sensor, which matters because a device's gyroscope triad is not always
    published in the same frame as its accelerometer triad -- the IO-VNBD release is one
    such case, where the accelerometer reports "up" as +z while the gyroscope's yaw axis
    is +y.

    On a normal handset the two frames agree and this returns (a sign-resolved copy of)
    ``accel_up``, so enabling the calibration is free.

    Args:
        gyro: ``(N, 3)`` angular rate in the gyroscope frame.
        accel_up: optional ``(3,)`` up direction from the accelerometer/gravity stream,
            used to resolve the sign ambiguity of an eigenvector and as a fallback.
        min_eig_ratio: how dominant the principal axis must be before it is trusted.
        agreement_deg: angle within which the two frames are declared consistent.
    """
    gyro = np.asarray(gyro, dtype=np.float64)
    fallback = None
    if accel_up is not None:
        fallback = unit(np.asarray(accel_up, dtype=np.float64).reshape(3))

    if len(gyro) < 50:
        axis = fallback if fallback is not None else np.array([0.0, 0.0, 1.0])
        return GyroFrameCalibration(axis, 0.0, True, True)

    centred = gyro - np.mean(gyro, axis=0)
    cov = np.einsum("ij,ik->jk", centred, centred) / len(centred)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    ratio = float(eigvals[-1] / max(eigvals[-2], 1e-12))

    if ratio < min_eig_ratio and fallback is not None:
        return GyroFrameCalibration(fallback, ratio, True, True)

    agrees = True
    if fallback is not None:
        cos = float(np.dot(axis, fallback))
        if cos < 0:
            axis = -axis
            cos = -cos
        agrees = np.degrees(np.arccos(np.clip(abs(cos), -1, 1))) <= agreement_deg

    return GyroFrameCalibration(unit(axis), ratio, agrees, False)


@dataclass
class YawRateCalibration:
    """Linear map from the raw gyroscope triad to earth-referenced yaw rate."""

    k: np.ndarray            # (3,) so that yaw_rate ~= gyro . k
    r2: float
    n_samples: int
    fallback: bool = False

    @property
    def confident(self) -> bool:
        return (not self.fallback) and self.r2 >= 0.5 and self.n_samples >= 200

    def apply(self, gyro: np.ndarray) -> np.ndarray:
        return np.sum(np.asarray(gyro, dtype=np.float64) * self.k, axis=-1)

    def as_dict(self) -> dict:
        return {"k": [float(v) for v in self.k], "r2": float(self.r2),
                "n_samples": int(self.n_samples), "fallback": bool(self.fallback),
                "confident": bool(self.confident)}


def reference_yaw_rate_from_positions(
    positions: np.ndarray,
    speed: np.ndarray,
    dt: float = 0.1,
    stride: int = 10,
    min_speed: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Yaw rate implied by how a GNSS track is actually curving.

    Course over ground is only meaningful when the vehicle is moving, so samples below
    ``min_speed`` are marked invalid rather than contributing heading noise.
    """
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(speed, dtype=np.float64)
    n = len(p)
    yaw = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    if n < 2 * stride + 1:
        return yaw, valid

    course = np.full(n, np.nan)
    for i in range(stride, n):
        d = p[i] - p[i - stride]
        if np.linalg.norm(d) > 0.5:
            course[i] = np.arctan2(d[1], d[0])

    for i in range(stride + 1, n - stride):
        a, b = course[i - stride], course[i + stride]
        if np.isfinite(a) and np.isfinite(b) and v[i] >= min_speed:
            diff = (b - a + np.pi) % (2 * np.pi) - np.pi
            yaw[i] = diff / (2 * stride * dt)
            valid[i] = True
    return yaw, valid


def fit_yaw_rate_from_reference(
    gyro: np.ndarray,
    reference_yaw_rate: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    fallback_axis: Optional[np.ndarray] = None,
) -> YawRateCalibration:
    """Least-squares map ``yaw_rate ~= gyro . k`` from a reference heading rate.

    Why this replaces picking an axis
    ---------------------------------
    Gravity fixes the *direction* of the vertical but a principal-axis estimate of the
    gyroscope's yaw axis is only defined up to sign, and the sign cannot be recovered from
    gravity when the two triads are published in different frames (their axes are then
    nearly perpendicular, so the disambiguating dot product is pure noise). Measured on this
    corpus that ambiguity flipped the sign on two of four trips, which reverses every turn
    and destroys the dead-reckoned trajectory.

    Regressing the gyroscope onto a yaw rate the vehicle demonstrably had -- taken from how
    the GNSS track curves -- determines axis, sign and scale together, needs no per-dataset
    constant, and uses only information available *before* an outage, so it is causally
    valid. It also absorbs a gyroscope scale-factor error for free.
    """
    g = np.asarray(gyro, dtype=np.float64)
    r = np.asarray(reference_yaw_rate, dtype=np.float64)
    n = min(len(g), len(r))
    g, r = g[:n], r[:n]
    m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)[:n]
    m = m & np.all(np.isfinite(g), axis=1) & np.isfinite(r)

    if int(m.sum()) < 50 or float(np.std(r[m])) < 1e-4:
        axis = (unit(np.asarray(fallback_axis, dtype=np.float64).reshape(3))
                if fallback_axis is not None else np.array([0.0, 0.0, 1.0]))
        return YawRateCalibration(axis, 0.0, int(m.sum()), fallback=True)

    G = g[m]
    y = r[m]
    A = G.T @ G + ridge * np.eye(3)
    k = np.linalg.solve(A, G.T @ y)

    pred = G @ k
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return YawRateCalibration(k, 1.0 - ss_res / ss_tot, int(m.sum()), fallback=False)


@dataclass
class MountFrameEstimator:
    """Recovers the horizontal heading of the vehicle's forward axis in the device frame.

    Gravity fixes two of three rotational degrees of freedom. The last one -- which way is
    "forward" within the horizontal plane -- is observable from centripetal coupling: in a
    turn the horizontal specific force is dominated by lateral acceleration
    ``a_lat = v * omega_yaw``, so the horizontal direction whose projection best correlates
    with ``omega_yaw`` is the *lateral* axis, and forward is perpendicular to it.

    The estimator accumulates a weighted 2x2 scatter of (horizontal acceleration) against
    ``omega_yaw`` and is only reported as converged once enough turning excitation has been
    observed. Until then, consumers must fall back on rotation-invariant quantities.
    """

    min_yaw_rate: float = 0.10
    min_excitation: float = 25.0
    forward: Optional[np.ndarray] = None
    lateral: Optional[np.ndarray] = None
    excitation: float = 0.0

    def fit(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        g_hat: np.ndarray,
    ) -> "MountFrameEstimator":
        accel = np.asarray(accel, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        g_hat = unit(np.asarray(g_hat, dtype=np.float64))

        _, a_h = decompose_along_gravity(accel, g_hat)
        omega_v = yaw_rate_about_gravity(gyro, g_hat)

        mask = np.abs(omega_v) >= self.min_yaw_rate
        self.excitation = float(np.sum(np.abs(omega_v[mask]))) if np.any(mask) else 0.0
        if not np.any(mask) or self.excitation < self.min_excitation:
            self.forward = None
            self.lateral = None
            return self

        # Build an orthonormal horizontal basis (e1, e2) perpendicular to g_hat.
        ref = np.array([1.0, 0.0, 0.0])
        g_mean = unit(np.mean(g_hat, axis=0))
        if abs(float(np.dot(g_mean, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = unit(ref - float(np.dot(ref, g_mean)) * g_mean)
        e2 = np.cross(g_mean, e1)

        # Explicit reduction rather than ``@``: NumPy's BLAS path raises spurious
        # floating-point flag warnings here on some builds even for finite inputs.
        c1 = np.sum(a_h[mask] * e1, axis=1)
        c2 = np.sum(a_h[mask] * e2, axis=1)
        w = omega_v[mask]

        # Least-squares direction d in the (e1, e2) plane maximising correlation with w.
        denom = float(np.dot(w, w)) + 1e-9
        k1 = float(np.dot(c1, w)) / denom
        k2 = float(np.dot(c2, w)) / denom
        norm = float(np.hypot(k1, k2))
        if norm < 1e-6:
            self.forward = None
            self.lateral = None
            return self

        lateral = unit((k1 * e1 + k2 * e2))
        forward = np.cross(lateral, g_mean)
        forward = unit(forward)

        self.lateral = lateral
        self.forward = forward
        return self

    @property
    def converged(self) -> bool:
        return self.forward is not None and self.excitation >= self.min_excitation

    def project(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(longitudinal, lateral)`` projections; zeros when not converged."""
        vec = np.atleast_2d(np.asarray(vec, dtype=np.float64))
        if not self.converged:
            z = np.zeros(len(vec))
            return z, z
        return np.sum(vec * self.forward, axis=-1), np.sum(vec * self.lateral, axis=-1)
