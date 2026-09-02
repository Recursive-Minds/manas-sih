import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
sc = BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s")

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

print(f"Blackout 300s-360s:")
print(f"Start pos: Est=({df_bo['est_e'].iloc[0]:.1f}, {df_bo['est_n'].iloc[0]:.1f}), GT=({df_bo['gt_e'].iloc[0]:.1f}, {df_bo['gt_n'].iloc[0]:.1f})")
print(f"Start heading: Est={df_bo['heading_deg'].iloc[0]:.1f}°, GT_bearing at 303s: 242.2°")
print(f"End pos:   Est=({df_bo['est_e'].iloc[-1]:.1f}, {df_bo['est_n'].iloc[-1]:.1f}), GT=({df_bo['gt_e'].iloc[-1]:.1f}, {df_bo['gt_n'].iloc[-1]:.1f})")
print(f"End heading:   Est={df_bo['heading_deg'].iloc[-1]:.1f}°, GT_bearing at 357s: 323.5°")
print(f"Final error: {df_bo['error_m'].iloc[-1]:.2f} m, Drift: {res.drift_percentage:.2f}%")
