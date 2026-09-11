"""
Evaluation and benchmark harness module.
"""

from sih.eval.benchmark import (
    BlackoutConfig,
    BenchmarkResult,
    BenchmarkRunner,
    plot_benchmark_result,
)
from sih.eval.metrics import (
    compute_drift_percentage,
    compute_rmse,
    compute_mae,
    decompose_along_cross_track,
    evaluate_blackout_metrics,
)

__all__ = [
    "BlackoutConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "plot_benchmark_result",
    "compute_drift_percentage",
    "compute_rmse",
    "compute_mae",
    "decompose_along_cross_track",
    "evaluate_blackout_metrics",
]
