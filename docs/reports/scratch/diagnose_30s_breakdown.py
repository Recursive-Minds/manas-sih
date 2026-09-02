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
print("Columns in history_df:", df.columns.tolist())
df_bo = df[df["in_blackout"]].copy()

# Compute GT cumulative distance from gt_e and gt_n
d_gt = np.sqrt(np.diff(df_bo["gt_e"], prepend=df_bo["gt_e"].iloc[0])**2 + np.diff(df_bo["gt_n"], prepend=df_bo["gt_n"].iloc[0])**2)
d_est = np.sqrt(np.diff(df_bo["est_e"], prepend=df_bo["est_e"].iloc[0])**2 + np.diff(df_bo["est_n"], prepend=df_bo["est_n"].iloc[0])**2)

print(f"GT Total Distance Travelled:  {np.sum(d_gt):.1f} m")
print(f"Est Total Distance Travelled: {np.sum(d_est):.1f} m (Diff: {np.sum(d_est) - np.sum(d_gt):+.1f} m)")
print(f"Final Position Error:         {res.final_position_error_m:.2f} m")
print(f"Drift Percentage:             {res.drift_percentage:.2f}%")

print("\nStep-by-step during blackout:")
for i in range(0, len(df_bo), 50):
    r = df_bo.iloc[i]
    t_s = (r["timestamp_ns"] - trip.imu_samples[0].timestamp_ns) * 1e-9
    print(f"  t={t_s:.1f}s | Err={r['error_m']:.1f}m | EstPos=({r['est_e']:.1f}, {r['est_n']:.1f}) | GTPos=({r['gt_e']:.1f}, {r['gt_n']:.1f}) | EstSpeed={r['speed_mps']:.1f} m/s | EstHdg={r['heading_deg']:.1f}°")
