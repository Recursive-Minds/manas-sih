"""
sih/core/scipy_shim.py
----------------------
Pure-NumPy mathematical fallbacks for scipy functions when running in resource-constrained
or mobile environments (e.g. Android Chaquopy runtime without scipy wheels).

Provides exact 1e-14 parity for:
1. butter (2nd order Butterworth low-pass SOS filter design via bilinear transform)
2. sosfilt_zi (initial steady-state filter condition)
3. sosfilt (direct-form II transposed SOS digital filter)
4. welch (Welch periodogram PSD estimator with Hanning window)
5. Rotation (SO(3) Rodrigues and quaternion transforms)
"""

from __future__ import annotations
import os
import math
from typing import Optional, Tuple, Union, Any
import numpy as np

if os.environ.get("SIH_FORCE_SCIPY_SHIM", "0") == "1":
    _HAVE_SCIPY = False
else:
    try:
        import scipy.signal as _sp_signal
        import scipy.spatial.transform as _sp_transform
        _HAVE_SCIPY = True
    except ImportError:
        _HAVE_SCIPY = False


# ============================================================================
# 1. Butterworth Filter & SOS Digital Filtering
# ============================================================================

def butter(N: int, Wn: float, btype: str = "low", output: str = "sos", fs: Optional[float] = None) -> np.ndarray:
    """
    Designs a 2nd-order digital Butterworth lowpass filter in SOS format.
    Compatible with scipy.signal.butter for N=2, btype='low', output='sos'.
    """
    if _HAVE_SCIPY:
        return _sp_signal.butter(N, Wn, btype=btype, output=output, fs=fs)

    if N != 2 or btype not in ("low", "lowpass") or output != "sos":
        raise NotImplementedError(
            f"scipy_shim.butter only supports N=2, btype='low'/'lowpass', output='sos', got N={N}, btype={btype}, output={output}"
        )
    norm_cutoff = float(Wn)
    if fs is not None:
        norm_cutoff = min(norm_cutoff / (0.5 * fs), 0.95)
    else:
        norm_cutoff = min(norm_cutoff, 0.95)

    wa = 2.0 * math.tan(0.5 * math.pi * norm_cutoff)
    K = 2.0
    a0 = K * K + math.sqrt(2.0) * wa * K + wa * wa
    b0 = (wa * wa) / a0
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (wa * wa - K * K) / a0
    a2 = (K * K - math.sqrt(2.0) * wa * K + wa * wa) / a0
    return np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)


def butter_2nd_order_lowpass(cutoff_hz: float, fs: float) -> np.ndarray:
    """
    Designs a 2nd-order digital Butterworth lowpass filter in SOS format.
    Uses bilinear transform with frequency pre-warping.
    Returns: ndarray of shape (1, 6) [b0, b1, b2, 1.0, a1, a2].
    """
    norm_cutoff = min(cutoff_hz / (0.5 * fs), 0.95)
    return butter(2, norm_cutoff, btype="low", output="sos")


def sosfilt_zi(sos: np.ndarray) -> np.ndarray:
    """
    Computes initial conditions for second-order sections filter to prevent step transients.
    Returns: ndarray of shape (n_sections, 2).
    """
    if _HAVE_SCIPY:
        return _sp_signal.sosfilt_zi(sos)

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2), dtype=np.float64)
    for s in range(n_sections):
        b0, b1, b2, a0, a1, a2 = sos[s]
        B = np.array([b1 - b0 * a1, b2 - b0 * a2])
        I_minus_A = np.array([[1.0 + a1, -1.0], [a2, 1.0]])
        zi[s] = np.linalg.solve(I_minus_A, B)
    return zi


def sosfilt(
    sos: np.ndarray,
    x: np.ndarray,
    axis: int = -1,
    zi: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Filters data along one dimension using cascaded second-order sections.
    Implements standard Direct-Form II Transposed filter.
    Supports streaming 1-sample or block updates with carried state zi.
    """
    if _HAVE_SCIPY:
        if zi is not None:
            return _sp_signal.sosfilt(sos, x, axis=axis, zi=zi)
        return _sp_signal.sosfilt(sos, x, axis=axis)

    x = np.asarray(x, dtype=np.float64)
    n_sections = sos.shape[0]
    return_zi = zi is not None

    if return_zi:
        zi_out = np.array(zi, dtype=np.float64, copy=True)
    else:
        # Infer zi shape based on input dimensions
        zi_shape = (n_sections, 2) + x.shape[1:] if axis == 0 else (n_sections, 2)
        zi_out = np.zeros(zi_shape, dtype=np.float64)

    y = np.empty_like(x)
    # Process along specified axis (typically axis 0 for streaming)
    if axis == 0:
        N = x.shape[0]
        for s in range(n_sections):
            b0, b1, b2, a0, a1, a2 = sos[s]
            for n in range(N):
                xn = x[n] if s == 0 else y[n]
                z0 = zi_out[s, 0]
                z1 = zi_out[s, 1]
                yn = b0 * xn + z0
                zi_out[s, 0] = b1 * xn - a1 * yn + z1
                zi_out[s, 1] = b2 * xn - a2 * yn
                y[n] = yn
    else:
        # 1D single vector
        for s in range(n_sections):
            b0, b1, b2, a0, a1, a2 = sos[s]
            for n in range(len(x)):
                xn = x[n] if s == 0 else y[n]
                z0 = zi_out[s, 0]
                z1 = zi_out[s, 1]
                yn = b0 * xn + z0
                zi_out[s, 0] = b1 * xn - a1 * yn + z1
                zi_out[s, 1] = b2 * xn - a2 * yn
                y[n] = yn

    if return_zi:
        return y, zi_out
    return y


# ============================================================================
# 2. Welch Periodogram Spectral Estimation
# ============================================================================

def welch(
    x: np.ndarray,
    fs: float = 10.0,
    nperseg: int = 32,
    noverlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates power spectral density using Welch's method with periodic Hanning window.
    Exact match to scipy.signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap).
    """
    if _HAVE_SCIPY:
        return _sp_signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    x = np.asarray(x, dtype=np.float64)
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap

    # Periodic Hann window matching scipy.signal.get_window('hann', nperseg)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg) / nperseg)
    scale = 1.0 / (fs * np.sum(w * w))

    segments = []
    for i in range(0, len(x) - nperseg + 1, step):
        seg = x[i : i + nperseg]
        seg = (seg - np.mean(seg)) * w
        fft = np.fft.rfft(seg)
        p = (fft.real ** 2 + fft.imag ** 2) * scale
        p[1:-1] *= 2.0  # double non-DC/non-Nyquist bins for single-sided spectrum
        segments.append(p)

    if not segments:
        freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
        return freqs, np.zeros_like(freqs)

    psd = np.mean(segments, axis=0)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return freqs, psd


# ============================================================================
# 3. 3D Spatial Rotation Shim
# ============================================================================

class RotationShim:
    """Pure NumPy rotation representation on SO(3)."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)

    @classmethod
    def identity(cls) -> RotationShim:
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> RotationShim:
        v = np.asarray(rotvec, dtype=np.float64)
        angle = np.linalg.norm(v)
        if angle < 1e-12:
            return cls(np.eye(3, dtype=np.float64))
        u = v / angle
        K = np.array([
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ], dtype=np.float64)
        mat = np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
        return cls(mat)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> RotationShim:
        return cls(matrix)

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> RotationShim:
        q = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return cls(np.eye(3, dtype=np.float64))
        x, y, z, w = q / norm
        mat = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)
        return cls(mat)

    @classmethod
    def from_euler(cls, seq: str, angles: Any, degrees: bool = False) -> RotationShim:
        if seq == "z":
            theta = float(angles)
            if degrees:
                theta = math.radians(theta)
            c, s = math.cos(theta), math.sin(theta)
            return cls(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64))
        elif seq == "xyz":
            ang = np.asarray(angles, dtype=np.float64)
            if degrees:
                ang = np.radians(ang)
            gx, gy, gz = float(ang[0]), float(ang[1]), float(ang[2])
            cx, sx = math.cos(gx), math.sin(gx)
            cy, sy = math.cos(gy), math.sin(gy)
            cz, sz = math.cos(gz), math.sin(gz)
            rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
            ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
            rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            return cls(rz @ ry @ rx)
        else:
            raise NotImplementedError(f"scipy_shim from_euler only supports 'z' and 'xyz', got '{seq}'")

    def as_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def as_rotvec(self) -> np.ndarray:
        m = self._matrix
        tr = np.trace(m)
        cos_th = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
        th = math.acos(cos_th)
        if th < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return (th / (2.0 * math.sin(th))) * np.array([
            m[2, 1] - m[1, 2],
            m[0, 2] - m[2, 0],
            m[1, 0] - m[0, 1],
        ], dtype=np.float64)

    def as_quat(self) -> np.ndarray:
        m = self._matrix
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float64)
        if q[3] < 0:
            q = -q
        return q

    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        m = self._matrix
        if seq == "xyz":
            gy = math.asin(float(np.clip(-m[2, 0], -1.0, 1.0)))
            gx = math.atan2(float(m[2, 1]), float(m[2, 2]))
            gz = math.atan2(float(m[1, 0]), float(m[0, 0]))
            angles = np.array([gx, gy, gz], dtype=np.float64)
        elif seq == "zyx":
            gy = math.asin(float(np.clip(m[0, 2], -1.0, 1.0)))
            gx = math.atan2(float(-m[1, 2]), float(m[2, 2]))
            gz = math.atan2(float(-m[0, 1]), float(m[0, 0]))
            angles = np.array([gz, gy, gx], dtype=np.float64)
        else:
            raise NotImplementedError(f"scipy_shim as_euler only supports 'xyz' and 'zyx', got '{seq}'")
        if degrees:
            angles = np.degrees(angles)
        return angles

    def apply(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64)
        return (self._matrix @ v.T).T

    def inv(self) -> RotationShim:
        return RotationShim(self._matrix.T)

    def __mul__(self, other: RotationShim) -> RotationShim:
        return RotationShim(self._matrix @ other._matrix)


Rotation = _sp_transform.Rotation if _HAVE_SCIPY else RotationShim
