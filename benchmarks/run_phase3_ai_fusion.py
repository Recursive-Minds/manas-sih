"""
Phase 3 Benchmark Execution: Evaluating ES-EKF with AI TCN-Attention Velocity Fusion on S-S1.
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
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner, plot_benchmark_result


def run_phase3_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PHASE 3 EVALUATION: ES-EKF FUSED WITH AI TCN-ATTENTION VELOCITY MODEL")
    print("=" * 75)
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Real Trip Data
    csv_path = download_iovnbd_trip("S-S1")
    loader = GenericDataLoader()
    trip = loader.load_file(csv_path)

    print(f"Loaded Trip: {trip.trip_id} ({trip.duration_s:.1f} s, {trip.total_gnss_distance_m:.1f} m)")

    # 2. Configurations to Compare
    configs = [
        (
            "Naive Baseline",
            PipelineConfig(
                calibration=CalibrationConfig(algorithm="pass_through"),
                fusion=FusionFilterConfig(algorithm="naive_dead_reckoning")
            )
        ),
        (
            "Phase 2: ES-EKF + NHC (No AI)",
            PipelineConfig(
                calibration=CalibrationConfig(algorithm="auto"),
                fusion=FusionFilterConfig(
                    algorithm="es_ekf_nhc",
                    params={"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15}
                )
            )
        ),
        (
            "Phase 3: ES-EKF + AI Velocity (Bayesian MoE Champion)",
            PipelineConfig(
                calibration=CalibrationConfig(algorithm="auto"),
                velocity=VelocityEstimatorConfig(
                    algorithm="moe_bayesian",
                    params={
                        "checkpoint_path": "models/checkpoints/best_moe_velocity_model.pt" if os.path.exists("models/checkpoints/best_moe_velocity_model.pt") else "models/checkpoints/best_velocity_model.pt",
                        "device": str(device)
                    }
                ),
                fusion=FusionFilterConfig(
                    algorithm="es_ekf_nhc",
                    params={"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15}
                )
            )
        ),
    ]

    # 3. Define Trips and Blackout Scenarios
    eval_trips = [
        ("S-S1", "Training Trip S-S1", [
            BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s"),
            BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s"),
        ]),
        ("S-S2", "UNSEEN Validation Trip S-S2 (Zero Data Leakage)", [
            BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="S-S2_30s_blackout_at_120s"),
            BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="S-S2_60s_blackout_at_300s"),
        ]),
    ]

    os.makedirs("artifacts", exist_ok=True)
    summary_rows = []

    for trip_id, trip_title, scenarios in eval_trips:
        csv_p = download_iovnbd_trip(trip_id)
        current_trip = loader.load_file(csv_p)
        print("\n" + "=" * 75)
        print(f"EVALUATING ON: {trip_title} ({current_trip.duration_s:.1f} s, {current_trip.total_gnss_distance_m:.1f} m)")
        print("=" * 75)

        for sc in scenarios:
            print("\n" + "#" * 75)
            print(f"TRIP: {trip_id} | SCENARIO: {sc.name} (Duration: ~{sc.duration_s}s)")
            print("#" * 75)

            for label, config in configs:
                runner = BenchmarkRunner(config=config)
                res = runner.run_trip(current_trip, sc)

                clean_label = label.lower().replace(" ", "_").replace("+", "plus").replace(":", "").replace("(", "").replace(")", "")
                plot_file = f"artifacts/{trip_id}_{sc.name}_{clean_label}.png"
                plot_benchmark_result(res, plot_file)

                summary_rows.append({
                    "Trip": trip_id,
                    "Scenario": sc.name,
                    "Pipeline": label,
                    "Distance (m)": f"{res.blackout_distance_m:.1f}",
                    "Final Error (m)": f"{res.final_position_error_m:.2f}",
                    "RMSE (m)": f"{res.rmse_position_m:.2f}",
                    "Drift %": f"{res.drift_percentage:.2f}%",
                    "SIH (<10%)": "PASSED" if res.benchmark_passed else "FAIL"
                })

                print(f"\n--- {label} ---")
                print(f"  Distance Travelled:       {res.blackout_distance_m:.1f} m")
                print(f"  Final Position Error:     {res.final_position_error_m:.2f} m")
                print(f"  Max Position Error:       {res.max_error_m:.2f} m")
                print(f"  RMSE Position Error:      {res.rmse_position_m:.2f} m")
                print(f"  Drift Percentage:         {res.drift_percentage:.2f}%")
                print(f"  SIH Target (<10%):        {'PASSED' if res.benchmark_passed else 'FAILED'}")
                print(f"  Plot:                     {plot_file}")

    print("\n" + "=" * 75)
    print("PHASE 1 - 3 MULTI-TRIP BENCHMARK MATRIX (TRAINING TRIP + UNSEEN VALIDATION TRIP)")
    print("=" * 75)
    import pandas as pd
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    run_phase3_benchmarks()
