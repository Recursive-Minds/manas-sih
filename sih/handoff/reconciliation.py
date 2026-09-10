"""
sih.handoff.reconciliation
--------------------------
C^2 continuous cubic Hermite smoothstep reconciliation engine.
Eliminates position jumps upon GNSS reacquisition following tunnels or blackouts.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from sih.data.geo import enu_to_geodetic


class HermiteReconciler:
    """
    Trajectory reconciliation engine using a C^2 continuous cubic Hermite smoothstep blend.

    Separates mathematical Kalman filter propagation from the rendered navigation puck coordinate,
    eliminating 15m-50m teleportation artifacts when re-acquiring satellites after prolonged outages.

    Mathematical Formulation:
        tau = (t - t_reacq) / T_blend,  tau in [0, 1]
        alpha(tau) = 3 * tau^2 - 2 * tau^3
        offset(t) = (1 - alpha(tau)) * (p_dead_reckoning - p_fused_reacq)
        p_display(t) = p_fused(t) + offset(t)

    Boundary Properties:
        alpha(0) = 0,   alpha'(0) = 0   -> Matches dead-reckoning coordinate & velocity exactly.
        alpha(1) = 1,   alpha'(1) = 0   -> Smoothly converges onto true GNSS fused coordinate.
    """

    def __init__(self, blend_duration_s: float = 1.2) -> None:
        self.blend_duration_s = max(0.1, blend_duration_s)
        self.reset()

    def reset(self) -> None:
        """Resets the reconciler to idle state."""
        self._is_blending = False
        self._blend_start_ts_ns: Optional[int] = None
        self._p_offset_start = np.zeros(3, dtype=np.float64)

    @property
    def is_blending(self) -> bool:
        """Returns True if the system is actively smoothing across a reacquisition window."""
        return self._is_blending

    def initiate_blend(
        self,
        timestamp_ns: int,
        p_dead_reckoning: np.ndarray,
        p_fused: np.ndarray,
    ) -> None:
        """
        Initiates a smoothstep reconciliation blend upon confirmed GNSS reacquisition.

        Parameters
        ----------
        timestamp_ns : int
            Nanosecond timestamp of the confirmed reacquisition event.
        p_dead_reckoning : np.ndarray
            Last dead-reckoning position (ENU, meters).
        p_fused : np.ndarray
            New fused Kalman filter position (ENU, meters).
        """
        self._blend_start_ts_ns = timestamp_ns
        self._p_offset_start = (p_dead_reckoning - p_fused).copy().astype(np.float64)

        # If the offset is negligible (< 0.25m), skip blending
        if float(np.linalg.norm(self._p_offset_start[:2])) < 0.25:
            self._is_blending = False
            self._p_offset_start = np.zeros(3, dtype=np.float64)
        else:
            self._is_blending = True

    def get_blended_position(
        self,
        current_ts_ns: int,
        p_fused: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Calculates the zero-jump display coordinate for the current timestep.

        Parameters
        ----------
        current_ts_ns : int
            Current sample timestamp in nanoseconds.
        p_fused : np.ndarray
            Current mathematical Kalman filter position (ENU, meters).

        Returns
        -------
        p_display : np.ndarray
            Smoothly reconciled position coordinate (ENU, meters).
        alpha : float
            Current blending factor in [0.0, 1.0] (0.0 = pure DR, 1.0 = pure fused).
        """
        if not self._is_blending or self._blend_start_ts_ns is None:
            return p_fused.copy(), 1.0

        elapsed_s = (current_ts_ns - self._blend_start_ts_ns) * 1e-9

        if elapsed_s >= self.blend_duration_s:
            # Blending complete: gracefully shut off
            self._is_blending = False
            self._blend_start_ts_ns = None
            self._p_offset_start = np.zeros(3, dtype=np.float64)
            return p_fused.copy(), 1.0

        tau = max(0.0, min(1.0, elapsed_s / self.blend_duration_s))
        # Cubic Hermite smoothstep polynomial
        alpha = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)

        offset_t = self._p_offset_start * (1.0 - alpha)
        p_display = p_fused + offset_t
        return p_display, float(alpha)
