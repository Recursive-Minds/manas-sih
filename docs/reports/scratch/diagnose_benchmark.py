import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

def test():
    trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
    sc = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")
    
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
    print(f"Blackout rows: {len(df_bo)}")
    print("Start of blackout:")
    print(df_bo.iloc[0][["time_s", "est_e", "est_n", "gt_e", "gt_n", "error_m", "speed_mps", "heading_deg"]])
    print("End of blackout:")
    print(df_bo.iloc[-1][["time_s", "est_e", "est_n", "gt_e", "gt_n", "error_m", "speed_mps", "heading_deg"]])
    print(f"Drift %: {res.drift_percentage:.2f}%")

if __name__ == "__main__":
    test()
