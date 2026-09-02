import os
import sys
import numpy as np

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

# Let's inspect the pipeline filter state right before 120s
t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(120.0 * 1e9)
bo_end_ns = bo_start_ns + int(30.0 * 1e9)

runner.pipeline.reset(initial_gnss=trip.gnss_samples[0])
gnss_idx = 0
n_gnss = len(trip.gnss_samples)

scales = []
bg_list = []
times = []

for imu in trip.imu_samples:
    t_curr = imu.timestamp_ns
    if t_curr > bo_end_ns:
        break
        
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
        g = trip.gnss_samples[gnss_idx]
        in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
        if not in_bo:
            runner.pipeline.process_gnss(g)
        gnss_idx += 1
        
    calib, vel, fused, matched = runner.pipeline.process_imu(imu)
    
    t_s = (t_curr - t0_ns) * 1e-9
    times.append(t_s)
    scales.append(runner.pipeline.fusion_filter._speed_scale)
    bg_list.append(runner.pipeline.fusion_filter._bg[2])

print(f"At t=120s (Blackout Start):")
idx_120 = np.searchsorted(times, 120.0)
print(f"  Speed Scale Factor: {scales[idx_120]:.4f}")
print(f"  Gyro Bias State Z:  {bg_list[idx_120]:.6f} rad/s ({np.degrees(bg_list[idx_120]):.4f}°/s)")
