"""Dual-Band Spectral Feature Extractor for IMU Speed Estimation.

Computes spectral energy distribution across macroscopic chassis dynamics (Band A: 0.1-1.5 Hz)
and aliased tyre/road interaction harmonics (Band B: 1.5-4.5 Hz) from sliding IMU windows.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from typing import Tuple, Optional


class DualBandSpectralExtractor:
    """Extracts frequency-domain energy features from sliding IMU windows."""

    def __init__(
        self,
        sampling_rate: float = 10.0,
        band_a: Tuple[float, float] = (0.1, 1.5),
        band_b: Tuple[float, float] = (1.5, 4.5),
        eps: float = 1e-6,
    ):
        self.fs = sampling_rate
        self.band_a = band_a
        self.band_b = band_b
        self.eps = eps

    def compute_window_features(self, window_imu: np.ndarray) -> np.ndarray:
        """Compute 4-dim spectral feature vector for a single window.

        Args:
            window_imu: (L, 6) or (L, 3) window of acceleration or IMU data.
                        If 6 channels, linear acceleration (first 3 channels) is used.

        Returns:
            features: (4,) array [E_bandA, E_bandB, E_ratio, v_proxy]
        """
        if window_imu.ndim != 2:
            raise ValueError(f"Expected 2D array (L, C), got shape {window_imu.shape}")

        accel = window_imu[:, :3]  # Focus on translational accelerations
        # Magnitude of acceleration to be orientation-invariant
        a_mag = np.linalg.norm(accel, axis=1)  # (L,)
        a_detrend = a_mag - np.mean(a_mag)

        L = len(a_detrend)
        if L < 8 or np.all(np.abs(a_detrend) < 1e-6):
            return np.zeros(4, dtype=np.float32)

        # Periodogram power spectral density
        nperseg = min(L, 32)
        freqs, psd = signal.welch(a_detrend, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Mask bands
        mask_a = (freqs >= self.band_a[0]) & (freqs <= self.band_a[1])
        mask_b = (freqs >= self.band_b[0]) & (freqs <= self.band_b[1])

        # Trapezoidal integration across frequency bands
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapz_func is None:
            from scipy import integrate
            trapz_func = integrate.trapezoid

        e_a = float(trapz_func(psd[mask_a], freqs[mask_a])) if np.any(mask_a) else 0.0
        e_b = float(trapz_func(psd[mask_b], freqs[mask_b])) if np.any(mask_b) else 0.0

        e_a = max(0.0, e_a)
        e_b = max(0.0, e_b)

        e_ratio = float(e_b / (e_a + e_b + self.eps))
        v_proxy = float(np.clip(e_b / (e_a + self.eps), 0.0, 10.0))

        return np.array([e_a, e_b, e_ratio, v_proxy], dtype=np.float32)

    def extract_sequence_features(
        self,
        imu_seq: np.ndarray,
        window_len: int = 60,
        stride: int = 1,
    ) -> np.ndarray:
        """Extract spectral features across full sequence with causal trailing windows.

        Args:
            imu_seq: (N, 6) IMU sequence
            window_len: length of window looking back from each timestep
            stride: step size for spectral computation; if > 1, computes on stride and interpolates

        Returns:
            spectral_feats: (N, 4) spectral features aligned with imu_seq
        """
        N = len(imu_seq)
        out = np.zeros((N, 4), dtype=np.float32)

        if stride <= 1:
            for k in range(N):
                start_idx = max(0, k - window_len + 1)
                w = imu_seq[start_idx : k + 1]
                out[k] = self.compute_window_features(w)
        else:
            computed_indices = list(range(0, N, stride))
            if computed_indices[-1] != N - 1:
                computed_indices.append(N - 1)
            vals = np.array(
                [self.compute_window_features(imu_seq[max(0, k - window_len + 1) : k + 1]) for k in computed_indices],
                dtype=np.float32,
            )
            for ch in range(4):
                out[:, ch] = np.interp(np.arange(N), computed_indices, vals[:, ch])

        return out
