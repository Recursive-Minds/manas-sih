"""
Benchmark execution script: Naive Open-Loop Dead Reckoning on Real Drive Data (S-S1).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")  # Headless backend

import sih
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.core.config import PipelineConfig, FusionFilterConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner, plot_benchmark_result


def run_naive_baseline():
    print("=" * 60)
    print("PHASE 1 EVALUATION: NAIVE DEAD RECKONING BASELINE")
    print("=" * 60)

    # 1. Ensure trip data is loaded
    csv_path = download_iovnbd_trip("S-S1")
    loader = GenericDataLoader()
    trip = loader.load_file(csv_path)

    print(f"Loaded Trip: {trip.trip_id}")
    print(f"Total Duration: {trip.duration_s:.1f} s ({trip.duration_s / 60.0:.2f} min)")
    print(f"Total GNSS Traveled Distance: {trip.total_gnss_distance_m:.1f} m")

    # 2. Configure Pipeline with Naive Dead Reckoning
    config = PipelineConfig(
        fusion=FusionFilterConfig(algorithm="naive_dead_reckoning")
    )
    pipeline = assemble_pipeline(config)

    # 3. Define Blackout Scenarios
    # Scenario 1: 30-second blackout starting at t = 120s (during active driving)
    # Scenario 2: 60-second blackout starting at t = 300s
    scenarios = [
        BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s"),
        BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s"),
    ]

    runner = BenchmarkRunner(pipeline=pipeline)
    os.makedirs("artifacts", exist_ok=True)

    for sc in scenarios:
        print("\n" + "-" * 50)
        print(f"Running Scenario: {sc.name}")
        res = runner.run_trip(trip, sc)

        plot_file = f"artifacts/{sc.name}_trajectory.png"
        plot_benchmark_result(res, plot_file)

        print(f"  Blackout Duration:        {res.blackout_duration_s:.1f} s")
        print(f"  Distance Travelled:       {res.blackout_distance_m:.1f} m")
        print(f"  Final Position Error:     {res.final_position_error_m:.2f} m")
        print(f"  Max Position Error:       {res.max_error_m:.2f} m")
        print(f"  RMSE Position Error:      {res.rmse_position_m:.2f} m")
        print(f"  Drift Percentage:         {res.drift_percentage:.2f}%")
        print(f"  SIH Target (<10%):        {'PASSED' if res.benchmark_passed else 'FAILED (Expected for Naive Baseline)'}")
        print(f"  Diagnostic Plot:          {plot_file}")

    print("=" * 60)


if __name__ == "__main__":
    run_naive_baseline()
