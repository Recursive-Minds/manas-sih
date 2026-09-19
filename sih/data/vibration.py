"""DEPRECATED / NON-CAUSAL LEGACY CONDITIONER.
DO NOT USE IN PRODUCTION OR INFERENCE.
Replaced by StreamingFeatureExtractor (sih/features/streaming.py), which uses causal
sosfilt with carried state and physical jerk clamping. This class uses scipy.signal.filtfilt,
which violates causality by filtering backwards across future sequence samples.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy import signal


class VibrationConditioner:
    """[DEPRECATED - NON-CAUSAL] Butterworth forward-backward low-pass filtering."""

    def __init__(
        self,
        sampling_rate: float = 10.0,
        cutoff_hz: float = 3.5,
        max_jerk_mps3: float = 15.0,  # Max allowable jerk (m/s^3) for automotive motion
    ):
        self.sampling_rate = sampling_rate
        self.cutoff_hz = cutoff_hz
        self.max_jerk_mps3 = max_jerk_mps3
        self.dt = 1.0 / sampling_rate

        # 2nd order Butterworth digital filter
        nyquist = 0.5 * sampling_rate
        norm_cutoff = min(cutoff_hz / nyquist, 0.95)
        self.b, self.a = signal.butter(2, norm_cutoff, btype="low", analog=False)

    def filter_imu_sequence(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply zero-phase digital filtering and spike suppression across sequence.

        Args:
            accel: (N, 3) linear dynamic acceleration in m/s^2
            gyro: (N, 3) angular rates in rad/s

        Returns:
            filtered_accel: (N, 3) conditioned acceleration
            filtered_gyro: (N, 3) conditioned gyroscope rates
        """
        N = len(accel)
        if N < 15:
            return accel.copy(), gyro.copy()

        # 1. Zero-phase forward-backward Butterworth low-pass filter
        filtered_accel = signal.filtfilt(self.b, self.a, accel, axis=0)
        filtered_gyro = signal.filtfilt(self.b, self.a, gyro, axis=0)

        # 2. Pothole / Shock Jerk Clamping
        max_delta_a = self.max_jerk_mps3 * self.dt
        for k in range(1, N):
            delta = filtered_accel[k] - filtered_accel[k - 1]
            delta_clamped = np.clip(delta, -max_delta_a, max_delta_a)
            filtered_accel[k] = filtered_accel[k - 1] + delta_clamped

        return filtered_accel, filtered_gyro
