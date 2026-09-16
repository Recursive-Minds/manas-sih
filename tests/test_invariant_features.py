"""Regression tests proving the invariant feature set really is invariant.

The central claim of the final system's front end is that re-orienting the phone does not
change what the network sees. These tests establish that by construction-independent
means: random SO(3) rotations are applied to the raw device-frame vectors and the feature
matrix must come back bit-for-bit close.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.invariant_features import INVARIANT_CHANNELS, InvariantFeatureExtractor
from engine.orientation import (
    GRAVITY,
    MountFrameEstimator,
    deleaned_yaw_rate,
    estimate_gravity_complementary,
    lean_angle_from_specific_force,
    two_wheeler_speed_from_lean,
    yaw_rate_about_gravity,
)

FS = 10.0


def synth_drive(n: int = 900, seed: int = 0):
    """A synthetic but physically coherent drive in the *vehicle* frame.

    Returns raw specific force (gravity included), angular rate, gravity stream and the
    true forward speed.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS

    speed = 8.0 + 6.0 * np.sin(2 * np.pi * t / 45.0) + 1.5 * np.sin(2 * np.pi * t / 7.0)
    speed = np.clip(speed, 0.0, None)
    a_long = np.gradient(speed, 1.0 / FS)

    omega_yaw = 0.35 * np.sin(2 * np.pi * t / 30.0)
    a_lat = speed * omega_yaw

    a_lin = np.column_stack([a_long, a_lat, 0.02 * rng.standard_normal(n)])
    # Road roughness scales with speed; rides on the vertical axis.
    a_lin[:, 2] += 0.15 * (speed / 10.0) * np.sin(2 * np.pi * 3.2 * t) + 0.01 * rng.standard_normal(n)

    omega = np.column_stack([
        0.02 * rng.standard_normal(n),
        0.02 * rng.standard_normal(n),
        omega_yaw,
    ])

    gravity = np.tile(np.array([0.0, 0.0, GRAVITY]), (n, 1))
    accel = a_lin + gravity
    return accel, omega, gravity, a_lin, speed


def random_rotations(k: int, seed: int = 7):
    return Rotation.random(k, random_state=seed).as_matrix()


# --------------------------------------------------------------------------------------
# P1 - static arbitrary rotation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("use_gravity_sensor", [True, False])
def test_features_invariant_under_static_rotation(use_gravity_sensor):
    accel, gyro, gravity, a_lin, _ = synth_drive()
    ex = InvariantFeatureExtractor(sampling_rate=FS)

    base = ex.extract(accel, gyro, gravity if use_gravity_sensor else None,
                      a_lin if use_gravity_sensor else None)

    for R in random_rotations(12):
        acc_r = np.einsum("ij,kj->ik", accel, R)
        gyr_r = np.einsum("ij,kj->ik", gyro, R)
        grav_r = np.einsum("ij,kj->ik", gravity, R) if use_gravity_sensor else None
        lin_r = np.einsum("ij,kj->ik", a_lin, R) if use_gravity_sensor else None
        rot = ex.extract(acc_r, gyr_r, grav_r, lin_r)
        assert np.allclose(base, rot, atol=1e-6, rtol=1e-6), (
            f"feature drift under rotation (gravity_sensor={use_gravity_sensor}): "
            f"max |delta| = {np.max(np.abs(base - rot))}"
        )


def test_every_channel_is_actually_exercised():
    """Guards against a channel being trivially zero, which would fake invariance.

    Uses a drive in which the phone is also slowly re-oriented, so that the tilt-sensitive
    channels receive real stimulus; a rigidly bolted phone legitimately reports zero tilt
    rate and would not exercise them.
    """
    accel, gyro, gravity, a_lin, _ = synth_drive(n=1200, seed=17)
    n = len(accel)
    t = np.arange(n) / FS
    # Continuous re-orientation of the device (P3-style gradual rotation).
    for i in range(n):
        R = Rotation.from_euler("xyz", [12 * np.sin(0.05 * t[i]), 20 * t[i] / t[-1], 8 * t[i] / t[-1]],
                                degrees=True).as_matrix()
        accel[i] = R.T @ accel[i]
        gyro[i] = R.T @ gyro[i]
        gravity[i] = R.T @ gravity[i]
        a_lin[i] = R.T @ a_lin[i]

    f = InvariantFeatureExtractor(sampling_rate=FS).extract(accel, gyro, gravity, a_lin)
    assert f.shape[1] == len(INVARIANT_CHANNELS)
    for i, name in enumerate(INVARIANT_CHANNELS):
        assert np.std(f[:, i]) > 1e-9, f"channel {name} is degenerate (std=0)"


def test_180_degree_flip_and_axis_swaps():
    """Discrete re-mounts (upside down, landscape) must be exactly neutral."""
    accel, gyro, gravity, a_lin, _ = synth_drive()
    ex = InvariantFeatureExtractor(sampling_rate=FS)
    base = ex.extract(accel, gyro, gravity, a_lin)

    for R in [
        Rotation.from_euler("z", 180, degrees=True).as_matrix(),
        Rotation.from_euler("x", 180, degrees=True).as_matrix(),
        Rotation.from_euler("y", 90, degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [90, 90, 90], degrees=True).as_matrix(),
    ]:
        rot = ex.extract(*[np.einsum("ij,kj->ik", v, R) for v in (accel, gyro, gravity, a_lin)])
        assert np.allclose(base, rot, atol=1e-6)


# --------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------

def test_edge_cases_do_not_produce_nan():
    ex = InvariantFeatureExtractor(sampling_rate=FS)
    n = 200
    cases = {
        "stationary": (np.tile([0, 0, GRAVITY], (n, 1)), np.zeros((n, 3))),
        "zero_everything": (np.zeros((n, 3)), np.zeros((n, 3))),
        "huge_spike": (np.tile([0, 0, GRAVITY], (n, 1)) + 500.0 * (np.arange(n)[:, None] == 100),
                       np.zeros((n, 3))),
        "pure_spin": (np.tile([0, 0, GRAVITY], (n, 1)), np.tile([0, 0, 5.0], (n, 1))),
    }
    for name, (a, w) in cases.items():
        f = ex.extract(a, w, None, None)
        assert np.all(np.isfinite(f)), f"non-finite features for case {name}"


def test_short_sequences():
    ex = InvariantFeatureExtractor(sampling_rate=FS)
    for n in (1, 2, 5, 30):
        a = np.tile([0.0, 0.0, GRAVITY], (n, 1))
        w = np.zeros((n, 3))
        f = ex.extract(a, w, None, None)
        assert f.shape == (n, len(INVARIANT_CHANNELS))
        assert np.all(np.isfinite(f))


# --------------------------------------------------------------------------------------
# Physics: centripetal cue, lean, de-leaning
# --------------------------------------------------------------------------------------

def test_centripetal_speed_cue_recovers_true_speed_in_turns():
    accel, gyro, gravity, a_lin, speed = synth_drive(n=1800, seed=3)
    ex = InvariantFeatureExtractor(sampling_rate=FS)
    f = ex.extract(accel, gyro, gravity, a_lin)
    v_cent = f[:, INVARIANT_CHANNELS.index("v_centripetal")]
    valid = f[:, INVARIANT_CHANNELS.index("v_cent_valid")] > 0.5
    # Only meaningful where the turn dominates longitudinal acceleration.
    strong = valid & (f[:, INVARIANT_CHANNELS.index("omega_yaw_abs")] > 0.25)
    assert strong.sum() > 100
    err = np.abs(v_cent[strong] - speed[strong])
    assert np.median(err) < 2.5, f"median centripetal speed error {np.median(err):.2f} m/s"


def test_lean_angle_matches_coordinated_turn_geometry():
    """phi = arctan(a_c/g) must be recovered from ||f|| = sqrt(g^2 + a_c^2)."""
    for phi_true_deg in [0.0, 5.0, 15.0, 30.0, 45.0]:
        phi_true = np.radians(phi_true_deg)
        a_c = GRAVITY * np.tan(phi_true)
        f = np.array([[a_c, 0.0, GRAVITY]])
        phi = lean_angle_from_specific_force(f)[0]
        assert abs(phi - phi_true) < 1e-6, f"lean {np.degrees(phi):.3f} != {phi_true_deg}"


def test_two_wheeler_speed_from_lean_is_exact_for_coordinated_turn():
    """v = g tan(phi)/omega must invert a_c = v*omega exactly."""
    for v_true in [5.0, 12.0, 25.0]:
        for omega in [0.15, 0.35, 0.8]:
            a_c = v_true * omega
            phi = np.arctan2(a_c, GRAVITY)
            v_est, valid = two_wheeler_speed_from_lean(np.array([phi]), np.array([omega]))
            assert valid[0]
            assert abs(v_est[0] - v_true) < 1e-6, f"{v_est[0]} != {v_true}"


def test_two_wheeler_speed_invalid_when_straight():
    v, valid = two_wheeler_speed_from_lean(np.array([0.0, 0.001]), np.array([0.0, 0.01]))
    assert not valid.any(), "speed-from-lean must not be trusted without yaw rate"


def test_deleaned_yaw_rate_recovers_earth_yaw():
    """A bike leaned by phi measures omega*cos(phi) about its own up axis."""
    phi = np.radians(30.0)
    omega_yaw_true = 0.4
    # Bike up axis tilted by phi about the forward (x) axis.
    R = Rotation.from_euler("x", phi).as_matrix()
    g_hat = (R @ np.array([0.0, 0.0, 1.0]))[None, :]
    omega_world = np.array([[0.0, 0.0, omega_yaw_true]])
    measured = yaw_rate_about_gravity(omega_world, g_hat)
    assert abs(measured[0] - omega_yaw_true * np.cos(phi)) < 1e-9
    corrected = deleaned_yaw_rate(omega_world, g_hat, np.array([phi]))
    assert abs(corrected[0] - omega_yaw_true) < 1e-9


# --------------------------------------------------------------------------------------
# Gravity estimation without a gravity sensor
# --------------------------------------------------------------------------------------

def test_complementary_gravity_is_rotation_equivariant():
    accel, gyro, _, _, _ = synth_drive(seed=11)
    g0 = estimate_gravity_complementary(accel, gyro, dt=1 / FS)
    for R in random_rotations(5, seed=21):
        gr = estimate_gravity_complementary(np.einsum("ij,kj->ik", accel, R), np.einsum("ij,kj->ik", gyro, R), dt=1 / FS)
        assert np.allclose(gr, np.einsum('ij,kj->ik', g0, R), atol=1e-6)


def test_complementary_gravity_tracks_true_vertical_under_static_tilt():
    n = 600
    tilt = Rotation.from_euler("y", 25, degrees=True).as_matrix()
    accel = np.tile((tilt.T @ np.array([0.0, 0.0, GRAVITY])), (n, 1))
    gyro = np.zeros((n, 3))
    g = estimate_gravity_complementary(accel, gyro, dt=1 / FS)
    expected = tilt.T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(g[-1], expected, atol=1e-3)


def test_complementary_gravity_rejects_sustained_lateral_acceleration():
    """Once levelled, a hard sustained turn must not drag the vertical reference over.

    The vehicle starts at rest (so the vertical is observable), then enters a sustained
    0.41 g lateral acceleration. A naive low-pass "gravity" estimate would tilt by ~22
    degrees; the gated complementary filter must stay level because ||f|| leaves the
    gravity-consistent band.
    """
    n_rest, n_turn = 200, 400
    rest = np.tile([0.0, 0.0, GRAVITY], (n_rest, 1))
    turn = np.tile([4.0, 0.0, GRAVITY], (n_turn, 1))  # ||f|| = 10.59 -> outside the gate
    accel = np.vstack([rest, turn])
    gyro = np.zeros((n_rest + n_turn, 3))

    g = estimate_gravity_complementary(accel, gyro, dt=1 / FS, accel_gate=0.5)
    up = np.array([0.0, 0.0, 1.0])
    angle = np.degrees(np.arccos(np.clip(float(np.dot(g[-1], up)), -1, 1)))
    naive = np.degrees(np.arctan2(4.0, GRAVITY))
    assert angle < 5.0, f"vertical dragged {angle:.1f} deg (naive low-pass would give {naive:.1f})"


# --------------------------------------------------------------------------------------
# Mount frame estimation
# --------------------------------------------------------------------------------------

def test_mount_frame_recovers_forward_axis_up_to_sign():
    accel, gyro, gravity, a_lin, _ = synth_drive(n=2400, seed=5)
    est = MountFrameEstimator().fit(a_lin, gyro, gravity)
    assert est.converged
    # In the synthetic vehicle frame forward is +x and lateral is +y.
    assert abs(abs(float(np.dot(est.forward, [1.0, 0.0, 0.0]))) - 1.0) < 0.2
    assert abs(abs(float(np.dot(est.lateral, [0.0, 1.0, 0.0]))) - 1.0) < 0.2


def test_mount_frame_reports_not_converged_without_turning():
    n = 1200
    a = np.tile([0.5, 0.0, 0.0], (n, 1))
    w = np.zeros((n, 3))
    g = np.tile([0.0, 0.0, 1.0], (n, 1))
    est = MountFrameEstimator().fit(a, w, g)
    assert not est.converged, "mount frame must not claim convergence on a straight road"
