"""Online Recursive Least Squares (RLS) Affine Speed Calibrator.

Dynamically learns vehicle-specific scale factor (alpha) and offset (beta)
during GNSS-aided driving:
    v_true = alpha * v_ai + beta

At blackout entry, parameters are frozen and applied to dead reckoning.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np


class RLSAffineCalibrator:
    """Recursive Least Squares affine calibrator: v_cal = alpha * v_ai + beta."""

    def __init__(
        self,
        forgetting_factor: float = 0.97,
        min_speed_mps: float = 1.0,
        alpha_bounds: Tuple[float, float] = (0.7, 1.4),
        beta_bounds: Tuple[float, float] = (-2.0, 2.0),
        min_updates_for_calibration: int = 10,
    ):
        self.lam = forgetting_factor
        self.min_v = min_speed_mps
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.min_updates = min_updates_for_calibration

        self.theta = np.array([1.0, 0.0], dtype=np.float64)  # [alpha, beta]
        self.P = np.eye(2, dtype=np.float64) * 10.0          # Initial covariance
        self.n_updates = 0
        self._frozen = False
        self._history_ai: List[float] = []
        self._history_gnss: List[float] = []

    def update(self, v_gnss: float, v_ai: float) -> None:
        """Update RLS parameters with synchronized GNSS and AI speed sample.

        Args:
            v_gnss: Ground truth speed from GNSS (m/s)
            v_ai: Raw predicted speed from neural network (m/s)
        """
        if self._frozen or v_gnss < self.min_v or v_ai < 0.1:
            return

        self._history_ai.append(float(v_ai))
        self._history_gnss.append(float(v_gnss))

        phi = np.array([float(v_ai), 1.0], dtype=np.float64)
        denom = float(self.lam + phi @ self.P @ phi)
        if denom < 1e-9:
            return

        K = (self.P @ phi) / denom
        err = float(v_gnss) - float(phi @ self.theta)

        self.theta = self.theta + K * err
        self.P = (self.P - np.outer(K, phi) @ self.P) / self.lam
        self.n_updates += 1

    def freeze(self) -> None:
        """Lock calibration parameters at blackout entry with persistent excitation check."""
        self._frozen = True
        # If speed variation was too low (< 1.0 m/s std), affine offset is unobservable.
        # Fall back to robust scalar scale ratio: alpha = mean(v_gnss) / mean(v_ai), beta = 0.
        if len(self._history_ai) >= self.min_updates:
            std_ai = float(np.std(self._history_ai))
            if std_ai < 1.0:
                mean_ai = float(np.mean(self._history_ai))
                mean_gnss = float(np.mean(self._history_gnss))
                if mean_ai > 0.5:
                    scalar_ratio = float(np.clip(mean_gnss / mean_ai, self.alpha_bounds[0], self.alpha_bounds[1]))
                    self.theta = np.array([scalar_ratio, 0.0], dtype=np.float64)

    def unfreeze(self) -> None:
        """Unlock calibration parameters when GNSS re-acquires."""
        self._frozen = False

    def get_calibrated(self, v_ai: float) -> float:
        """Apply calibrated scale and offset to AI speed estimate.

        Args:
            v_ai: Instantaneous AI speed estimate (m/s)

        Returns:
            Calibrated forward speed (m/s)
        """
        v_ai = float(v_ai)
        if v_ai <= 0.0:
            return 0.0

        if self.n_updates < self.min_updates:
            return v_ai  # Identity fallback before sufficient excitation

        alpha = float(np.clip(self.theta[0], self.alpha_bounds[0], self.alpha_bounds[1]))
        beta = float(np.clip(self.theta[1], self.beta_bounds[0], self.beta_bounds[1]))

        v_cal = alpha * v_ai + beta
        return float(max(0.0, v_cal))

    @property
    def alpha(self) -> float:
        return float(np.clip(self.theta[0], self.alpha_bounds[0], self.alpha_bounds[1]))

    @property
    def beta(self) -> float:
        return float(np.clip(self.theta[1], self.beta_bounds[0], self.beta_bounds[1]))
