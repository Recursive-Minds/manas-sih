"""
Core Evaluation Metrics for Smartphone Intelligent Dead Reckoning (SIH).

Computes standardized navigation metrics:
- Final Position Error (m)
- Drift Percentage of Total Distance Travelled (%)
- Along-Track (Longitudinal) vs Cross-Track (Lateral) Error Decomposition
- Speed Root Mean Square Error (RMSE) and Mean Absolute Error (MAE)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np


def compute_drift_percentage(final_pos_error_m: float, total_distance_m: float) -> float:
    """
    Computes final position error as a percentage of total distance travelled.

    Formula:
    drift_pct = (final_pos_error_m / max(total_distance_m, 1e-3)) * 100.0
    """
    if total_distance_m <= 1e-3:
        return 0.0
    return float((final_pos_error_m / total_distance_m) * 100.0)


def compute_rmse(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Computes Root Mean Square Error between predictions and ground truth."""
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if len(pred) == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_mae(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Computes Mean Absolute Error between predictions and ground truth."""
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if len(pred) == 0:
        return 0.0
    return float(np.mean(np.abs(pred - gt)))


def decompose_along_cross_track(
    est_points: np.ndarray,
    gt_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes 2D horizontal position errors into along-track and cross-track components.

    Parameters:
        est_points: (N, 2) array of estimated [East, North] coordinates in meters.
        gt_points:  (N, 2) array of ground truth [East, North] coordinates in meters.

    Returns:
        along_track: (N,) signed longitudinal errors (positive = leading/overshoot).
        cross_track: (N,) signed lateral errors (positive = left deviation).
    """
    est = np.asarray(est_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)

    if len(est) == 0 or len(gt) == 0:
        return np.array([]), np.array([])

    de = np.gradient(gt[:, 0])
    dn = np.gradient(gt[:, 1])
    ds = np.hypot(de, dn) + 1e-6

    # Unit tangent vector along track
    te = de / ds
    tn = dn / ds

    # Error vector (est - gt)
    diff_e = est[:, 0] - gt[:, 0]
    diff_n = est[:, 1] - gt[:, 1]

    along_track = diff_e * te + diff_n * tn
    cross_track = diff_e * (-tn) + diff_n * te

    return along_track, cross_track


def evaluate_blackout_metrics(
    est_enu: np.ndarray,
    gt_enu: np.ndarray,
    distance_m: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluates complete set of standardized dead-reckoning metrics over a blackout outage.

    Parameters:
        est_enu: (N, 2 or 3) estimated trajectory in Local Tangent Plane ENU.
        gt_enu:  (N, 2 or 3) ground truth trajectory in Local Tangent Plane ENU.
        distance_m: optional pre-calculated total distance (m).

    Returns:
        Dictionary containing final_error_m, max_error_m, drift_pct, along_track_rmse_m,
        cross_track_rmse_m, and mean_error_m.
    """
    est = np.asarray(est_enu, dtype=np.float64)[:, :2]
    gt = np.asarray(gt_enu, dtype=np.float64)[:, :2]

    if len(est) == 0 or len(gt) == 0:
        return {
            "final_error_m": 0.0,
            "max_error_m": 0.0,
            "mean_error_m": 0.0,
            "drift_pct": 0.0,
            "along_track_rmse_m": 0.0,
            "cross_track_rmse_m": 0.0,
        }

    errors = np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1])
    final_error = float(errors[-1])
    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))

    if distance_m is None:
        diffs = np.diff(gt, axis=0)
        distance_m = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    drift_pct = compute_drift_percentage(final_error, distance_m)

    along, cross = decompose_along_cross_track(est, gt)
    along_rmse = float(np.sqrt(np.mean(along ** 2))) if len(along) > 0 else 0.0
    cross_rmse = float(np.sqrt(np.mean(cross ** 2))) if len(cross) > 0 else 0.0

    return {
        "final_error_m": final_error,
        "max_error_m": max_error,
        "mean_error_m": mean_error,
        "drift_pct": drift_pct,
        "along_track_rmse_m": along_rmse,
        "cross_track_rmse_m": cross_rmse,
        "total_distance_m": float(distance_m),
    }
