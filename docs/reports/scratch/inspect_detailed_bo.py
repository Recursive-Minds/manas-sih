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

sc30 = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")
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
res = runner.run_trip(trip, sc30)
df_bo = res.history_df[res.history_df["in_blackout"]].copy()

df_bo["time_in_bo"] = (df_bo["timestamp_ns"] - df_bo["timestamp_ns"].iloc[0]) * 1e-9
print(df_bo[["time_in_bo", "speed_mps", "heading_deg", "est_e", "est_n", "gt_e", "gt_n", "error_m"]].iloc[::30].to_string())

gt_dist = np.sum(np.sqrt(np.diff(df_bo["gt_e"])**2 + np.diff(df_bo["gt_n"])**2))
est_dist = np.sum(np.sqrt(np.diff(df_bo["est_e"])**2 + np.diff(df_bo["est_n"])**2))
print(f"\nGT Distance Travelled:  {gt_dist:.2f} m")
print(f"Est Distance Travelled: {est_dist:.2f} m (Ratio: {est_dist / gt_dist:.4f})")
print(f"Final Pos Error:        {df_bo['error_m'].iloc[-1]:.2f} m")
