import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

trip_train = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
trip_val = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

def test_pipeline(trip, start_s, dur_s, name):
    sc = BlackoutConfig(start_time_s=start_s, duration_s=dur_s, name=name)
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
    res = runner.run_trip(trip, sc)
    df = res.history_df
    df_bo = df[df["in_blackout"]]
    print(f"[{trip.trip_id} - {name}]")
    print(f"  Distance: {res.blackout_distance_m:.1f} m | Final Error: {res.final_position_error_m:.2f} m | Drift: {res.drift_percentage:.2f}%")
    print(f"  Start Hdg: {df_bo['heading_deg'].iloc[0]:.1f}° | End Hdg: {df_bo['heading_deg'].iloc[-1]:.1f}°\n")

print("=== TRIP S-S1 (Training Trip) ===")
test_pipeline(trip_train, 120.0, 30.0, "30s_blackout_at_120s")
test_pipeline(trip_train, 300.0, 60.0, "60s_blackout_at_300s")

print("=== TRIP S-S2 (UNSEEN Validation Trip - Zero Leakage) ===")
test_pipeline(trip_val, 120.0, 30.0, "S-S2_30s_blackout_at_120s")
test_pipeline(trip_val, 300.0, 60.0, "S-S2_60s_blackout_at_300s")
