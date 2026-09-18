"""
NEGATIVE RESULT DOCUMENTATION (PHASE 2 - SPEED RECALIBRATION)
STATUS: NOT IN USE - REJECTED

This script fits monotonic speed recalibration candidates f(v_pred) -> v_corr
(Isotonic Regression and Piecewise-Linear with knots at 20 km/h and 50 km/h)
on held-out Part 2 validation slices of S-M, S-S1, and S-S2 (49,869 samples).

REJECTION SUMMARY:
Baseline (11.59% median / 32.56% P90) beat f-only (11.46% / 41.10% P90)
and f+alpha (12.34% / 32.78% P90).
On held-out trips, baseline won decisively (8.73% vs 9.86% and 9.18%).
Per band, f severely degraded the >50 km/h regime on both held-out trips:
  - S-S3a bias degraded from -2.02 to -3.19 m/s, RMSE degraded from 3.31 to 4.08 m/s
  - S-S4 bias degraded from -5.48 to -6.50 m/s
Because the high-speed band had only 6,259 samples in the validation slice,
f overfit the validation slice and failed to generalize.
This file is preserved solely for reproducibility and negative result records.
DO NOT IMPORT OR INVOKE IN PRODUCTION RUNTIME.
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def fit_and_report():
    cache_npz = os.path.join(ROOT_DIR, "data", "cache", "v_preds_all_trips.npz")
    if not os.path.exists(cache_npz):
        print(f"Cache {cache_npz} not found. Run benchmark precomputation first.")
        return

    from sih.data.split import compute_trip_partition
    z = np.load(cache_npz, allow_pickle=True)

    fit_preds = []
    fit_gts = []
    for tid in ["S-M", "S-S1", "S-S2"]:
        v_pred = z[f"v_pred_{tid}"]
        v_gt = z[f"v_gt_{tid}"]
        part = compute_trip_partition(tid, len(v_pred))
        s_idx, e_idx = part.val_range
        e_idx = min(e_idx, len(v_pred), len(v_gt))
        fit_preds.append(v_pred[s_idx:e_idx])
        fit_gts.append(v_gt[s_idx:e_idx])

    X_fit = np.concatenate(fit_preds)
    y_fit = np.concatenate(fit_gts)

    K1, K2 = 5.56, 13.89  # 20 km/h and 50 km/h
    b1 = np.minimum(np.maximum(X_fit, 0.0), K1)
    b2 = np.clip(X_fit - K1, 0.0, K2 - K1)
    b3 = np.maximum(0.0, X_fit - K2)
    A_fit = np.column_stack([b1, b2, b3])
    slopes, _ = nnls(A_fit, y_fit)

    print("=" * 70)
    print("NEGATIVE RESULT: FITTED PIECEWISE-LINEAR PARAMETERS (NOT IN USE)")
    print(f"Slopes: s1={slopes[0]:.4f}, s2={slopes[1]:.4f}, s3={slopes[2]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    fit_and_report()
