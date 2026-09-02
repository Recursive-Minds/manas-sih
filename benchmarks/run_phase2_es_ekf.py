"""
Phase 2 Benchmark Execution: Comparing Naive Baseline vs ES-EKF vs ES-EKF+NHC on Real Drive Data (S-S1).
Target: NVIDIA RTX 4060 GPU / CUDA acceleration.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")  # Headless backend
import torch

import sih
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.core.config import PipelineConfig, FusionFilterConfig, CalibrationConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner, plot_benchmark_result


def run_phase2_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("PHASE 2 EVALUATION: ES-EKF + NON-HOLONOMIC CONSTRAINTS (NHC)")
    print("=" * 70)
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Real Trip Data
    csv_path = download_iovnbd_trip("S-S1")
    loader = GenericDataLoader()
    trip = loader.load_file(csv_path)

    print(f"Loaded Trip: {trip.trip_id} ({trip.duration_s:.1f} s, {trip.total_gnss_distance_m:.1f} m)")

    # 2. Define Algorithms to Compare
    algorithms = [
        ("Naive Baseline", "naive_dead_reckoning", {}),
        ("ES-EKF (No NHC)", "es_ekf", {}),
        ("ES-EKF + NHC", "es_ekf_nhc", {"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15}),
    ]

    # 3. Define Blackout Scenarios
    scenarios = [
        BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s"),
        BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s"),
    ]

    os.makedirs("artifacts", exist_ok=True)

    for sc in scenarios:
        print("\n" + "#" * 70)
        print(f"SCENARIO: {sc.name} (Duration: ~{sc.duration_s}s)")
        print("#" * 70)

        for label, algo_name, params in algorithms:
            config = PipelineConfig(
                calibration=CalibrationConfig(algorithm="auto"),
                fusion=FusionFilterConfig(algorithm=algo_name, params=params)
            )
            runner = BenchmarkRunner(config=config)
            res = runner.run_trip(trip, sc)

            clean_label = label.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
            plot_file = f"artifacts/{sc.name}_{clean_label}.png"
            plot_benchmark_result(res, plot_file)

            print(f"\n--- {label} ---")
            print(f"  Distance Travelled:       {res.blackout_distance_m:.1f} m")
            print(f"  Final Position Error:     {res.final_position_error_m:.2f} m")
            print(f"  Max Position Error:       {res.max_error_m:.2f} m")
            print(f"  RMSE Position Error:      {res.rmse_position_m:.2f} m")
            print(f"  Drift Percentage:         {res.drift_percentage:.2f}%")
            print(f"  SIH Target (<10%):        {'PASSED' if res.benchmark_passed else 'FAILED'}")
            print(f"  Plot:                     {plot_file}")

    print("=" * 70)


if __name__ == "__main__":
    run_phase2_benchmarks()
