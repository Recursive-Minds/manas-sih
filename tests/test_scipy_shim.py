"""
tests/test_scipy_shim.py
------------------------
Verifies numerical equivalence of pure NumPy scipy_shim implementations
against scipy.signal and scipy.spatial.transform.
"""

import math
import numpy as np
import pytest
from sih.core import scipy_shim


def test_butter_2nd_order_lowpass():
    try:
        from scipy import signal
        has_scipy = True
    except ImportError:
        has_scipy = False

    sos_shim = scipy_shim.butter_2nd_order_lowpass(3.5, 10.0)
    assert sos_shim.shape == (1, 6)
    assert sos_shim[0, 3] == 1.0

    if has_scipy:
        sos_sp = signal.butter(2, 3.5 / 5.0, btype="low", output="sos")
        diff = np.max(np.abs(sos_shim - sos_sp))
        assert diff < 1e-12, f"Butterworth SOS diff {diff:.3e} exceeds 1e-12"


def test_sosfilt_zi():
    try:
        from scipy import signal
        has_scipy = True
    except ImportError:
        has_scipy = False

    sos = scipy_shim.butter_2nd_order_lowpass(4.0, 50.0)
    zi_shim = scipy_shim.sosfilt_zi(sos)
    assert zi_shim.shape == (1, 2)

    if has_scipy:
        zi_sp = signal.sosfilt_zi(sos)
        diff = np.max(np.abs(zi_shim - zi_sp))
        assert diff < 1e-12, f"sosfilt_zi diff {diff:.3e} exceeds 1e-12"


def test_sosfilt_streaming():
    try:
        from scipy import signal
        has_scipy = True
    except ImportError:
        has_scipy = False

    sos = scipy_shim.butter_2nd_order_lowpass(3.5, 10.0)
    zi = scipy_shim.sosfilt_zi(sos)

    np.random.seed(42)
    x = np.random.randn(50, 3)

    # Stream through shim
    zi_shim = zi[:, :, None] * x[0:1, None, :]
    y_shim = []
    for i in range(len(x)):
        y_i, zi_shim = scipy_shim.sosfilt(sos, x[i : i + 1], axis=0, zi=zi_shim)
        y_shim.append(y_i[0])
    y_shim = np.array(y_shim)

    if has_scipy:
        zi_sp = zi[:, :, None] * x[0:1, None, :]
        y_sp = []
        for i in range(len(x)):
            y_i, zi_sp = signal.sosfilt(sos, x[i : i + 1], axis=0, zi=zi_sp)
            y_sp.append(y_i[0])
        y_sp = np.array(y_sp)

        diff = np.max(np.abs(y_shim - y_sp))
        assert diff < 1e-14, f"sosfilt streaming diff {diff:.3e} exceeds 1e-14"


def test_welch():
    try:
        from scipy import signal
        has_scipy = True
    except ImportError:
        has_scipy = False

    np.random.seed(123)
    x = np.random.randn(60)
    f_shim, p_shim = scipy_shim.welch(x, fs=10.0, nperseg=32, noverlap=16)

    assert len(f_shim) == 17
    assert len(p_shim) == 17

    if has_scipy:
        f_sp, p_sp = signal.welch(x, fs=10.0, nperseg=32, noverlap=16)
        diff_f = np.max(np.abs(f_shim - f_sp))
        diff_p = np.max(np.abs(p_shim - p_sp))
        assert diff_f < 1e-12
        assert diff_p < 1e-12, f"welch PSD diff {diff_p:.3e} exceeds 1e-12"


def test_rotation_shim():
    rotvec = np.array([0.15, -0.35, 0.45])
    r_shim = scipy_shim.RotationShim.from_rotvec(rotvec)
    mat = r_shim.as_matrix()
    assert mat.shape == (3, 3)
    assert np.allclose(mat @ mat.T, np.eye(3), atol=1e-12)

    rotvec_rec = r_shim.as_rotvec()
    assert np.allclose(rotvec, rotvec_rec, atol=1e-12)
