import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

def test_runs():
    csv_path = download_iovnbd_trip("S-S1")
    loader = GenericDataLoader()
    trip = loader.load_file(csv_path)
    print(f"Loaded Trip: {trip.trip_id} ({trip.duration_s:.1f} s, {trip.total_gnss_distance_m:.1f} m)")

    sc = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")

    configs = [
        ("Naive Baseline", PipelineConfig(fusion=FusionFilterConfig(algorithm="naive_dead_reckoning"))),
        ("ES-EKF (No NHC)", PipelineConfig(fusion=FusionFilterConfig(algorithm="es_ekf"))),
        ("ES-EKF + NHC", PipelineConfig(fusion=FusionFilterConfig(algorithm="es_ekf_nhc"))),
        ("ES-EKF + AI Velocity", PipelineConfig(
            velocity=VelocityEstimatorConfig(algorithm="tcn_attention", params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}),
            fusion=FusionFilterConfig(algorithm="es_ekf_nhc")
        ))
    ]

    for name, cfg in configs:
        runner = BenchmarkRunner(config=cfg)
        res = runner.run_trip(trip, sc)
        print(f"\n--- {name} ---")
        print(f"  Distance Travelled:   {res.blackout_distance_m:.1f} m")
        print(f"  Final Position Error: {res.final_position_error_m:.2f} m")
        print(f"  Drift %:              {res.drift_percentage:.2f}%")
        print(f"  Max Error:            {res.max_error_m:.2f} m")
        print(f"  RMSE Error:           {res.rmse_position_m:.2f} m")
        print(f"  Passed (<10%):        {res.benchmark_passed}")

if __name__ == "__main__":
    test_runs()
