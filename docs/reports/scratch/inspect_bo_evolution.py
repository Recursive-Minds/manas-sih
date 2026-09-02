import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Ground truth path vs estimated path coordinates
t = (df_bo["timestamp_ns"] - df_bo["timestamp_ns"].iloc[0]) * 1e-9
e_est = df_bo["est_e"].values
n_est = df_bo["est_n"].values
e_gt = df_bo["gt_e"].values
n_gt = df_bo["gt_n"].values

# Error at each 5 seconds
for s in [0, 5, 10, 15, 20, 25, 30]:
    idx = int(s * 10)
    if idx >= len(t):
        idx = -1
    err = np.sqrt((e_est[idx] - e_gt[idx])**2 + (n_est[idx] - n_gt[idx])**2)
    print(f"t={t.iloc[idx]:.1f}s | Est=({e_est[idx]:.1f}, {n_est[idx]:.1f}) | GT=({e_gt[idx]:.1f}, {n_gt[idx]:.1f}) | Error={err:.2f} m | Est Heading={df_bo['heading_deg'].iloc[idx]:.1f}°")
