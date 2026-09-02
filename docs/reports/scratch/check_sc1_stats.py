import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

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
df_bo = df[df["in_blackout"]].copy()

print(f"Scenario 1 Speed & Heading Stats:")
print(f"  AI Speed:  mean={df_bo['speed_mps'].mean():.2f} m/s, min={df_bo['speed_mps'].min():.2f}, max={df_bo['speed_mps'].max():.2f}")
print(f"  GT Speed:  mean={df_bo['gt_speed_mps'].mean():.2f} m/s, min={df_bo['gt_speed_mps'].min():.2f}, max={df_bo['gt_speed_mps'].max():.2f}")
print(f"  Speed Error RMSE: {np.sqrt(np.mean((df_bo['speed_mps'] - df_bo['gt_speed_mps'])**2)):.2f} m/s")
print(f"  Start Heading: Est={df_bo['heading_deg'].iloc[0]:.1f}°, GT={df_bo['gt_heading_deg'].iloc[0]:.1f}°")
print(f"  End Heading:   Est={df_bo['heading_deg'].iloc[-1]:.1f}°, GT={df_bo['gt_heading_deg'].iloc[-1]:.1f}°")
