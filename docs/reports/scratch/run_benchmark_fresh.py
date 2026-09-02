import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner, plot_benchmark_result

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

sc30 = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")
sc60 = BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s")

cfg = PipelineConfig(
    calibration=CalibrationConfig(algorithm="auto"),
    velocity=VelocityEstimatorConfig(
        algorithm="tcn_attention",
        params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}
    ),
    fusion=FusionFilterConfig(
        algorithm="es_ekf_nhc",
        params={"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15}
    )
)

runner = BenchmarkRunner(config=cfg)

res30 = runner.run_trip(trip, sc30)
plot_benchmark_result(res30, "artifacts/S-S1_30s_blackout_at_120s_phase_3_es-ekf_plus_ai_velocity_tcn-attention.png")

res60 = runner.run_trip(trip, sc60)
plot_benchmark_result(res60, "artifacts/S-S1_60s_blackout_at_300s_phase_3_es-ekf_plus_ai_velocity_tcn-attention.png")

print(f"30s Blackout: Final Error = {res30.final_position_error_m:.2f} m | Drift = {res30.drift_percentage:.2f}% | Passed = {res30.benchmark_passed}")
print(f"60s Blackout: Final Error = {res60.final_position_error_m:.2f} m | Drift = {res60.drift_percentage:.2f}% | Passed = {res60.benchmark_passed}")
