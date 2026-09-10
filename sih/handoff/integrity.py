"""
sih.handoff.integrity
---------------------
Statistical innovation and kinematic integrity checks for GNSS fix quality evaluation.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from sih.core.contracts import GNSSSample


def compute_position_nis(y_p: np.ndarray, S_p: np.ndarray, dims: int = 2) -> float:
    """
    Computes the Normalized Innovation Squared (NIS / Mahalanobis distance squared).

    Parameters
    ----------
    y_p : np.ndarray
        Position innovation vector (measurement - prediction) in meters. Shape (3,) or (2,).
    S_p : np.ndarray
        Innovation covariance matrix. Shape (3, 3) or (2, 2).
    dims : int
        Number of dimensions to evaluate (default 2 for horizontal East-North).

    Returns
    -------
    float
        NIS value. For 2-DOF, Chi-Square 95% threshold is 5.99, 99% is 9.21.
    """
    y = y_p[:dims].astype(np.float64)
    S = S_p[:dims, :dims].astype(np.float64)

    try:
        # Solve S * x = y -> x = inv(S) * y
        inv_S_y = np.linalg.solve(S, y)
        nis = float(np.dot(y, inv_S_y))
        return max(0.0, nis)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if covariance is ill-conditioned
        inv_S = np.linalg.pinv(S)
        nis = float(y.T @ inv_S @ y)
        return max(0.0, nis)


def check_kinematic_feasibility(
    p_new: np.ndarray,
    p_old: np.ndarray,
    dt_s: float,
    v_max_mps: float = 45.0,
    buffer_m: float = 8.0,
) -> bool:
    """
    Validates whether an apparent position shift is physically plausible.

    Parameters
    ----------
    p_new : np.ndarray
        New incoming position (ENU, meters).
    p_old : np.ndarray
        Previous verified position (ENU, meters).
    dt_s : float
        Elapsed time interval in seconds.
    v_max_mps : float
        Maximum plausible vehicle speed (45 m/s = 162 km/h).
    buffer_m : float
        Receiver measurement noise buffer allowance in meters.

    Returns
    -------
    bool
        True if the motion satisfies physical kinematic velocity limits.
    """
    if dt_s <= 0.0:
        return False

    disp_m = float(np.linalg.norm(p_new[:2] - p_old[:2]))
    max_allowed_dist = v_max_mps * dt_s + buffer_m
    return disp_m <= max_allowed_dist


def evaluate_signal_quality(
    gnss: GNSSSample,
    max_accuracy_h_m: float = 25.0,
) -> bool:
    """
    Evaluates basic receiver validity and reported horizontal accuracy.

    Parameters
    ----------
    gnss : GNSSSample
        Incoming GNSS fix.
    max_accuracy_h_m : float
        Maximum acceptable horizontal accuracy threshold in meters.

    Returns
    -------
    bool
        True if fix passes receiver sanity checks.
    """
    if not gnss.is_valid:
        return False

    if gnss.accuracy_h_m is None or np.isnan(gnss.accuracy_h_m):
        return False

    if gnss.accuracy_h_m > max_accuracy_h_m:
        return False

    # Check for NaN / Inf coordinates
    if np.isnan(gnss.latitude_deg) or np.isnan(gnss.longitude_deg):
        return False

    return True
