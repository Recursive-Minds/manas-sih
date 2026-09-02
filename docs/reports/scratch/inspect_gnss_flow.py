import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.core.config import PipelineConfig, FusionFilterConfig, CalibrationConfig
from sih.core.pipeline import assemble_pipeline

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
cfg = PipelineConfig(
    calibration=CalibrationConfig(algorithm="auto"),
    fusion=FusionFilterConfig(algorithm="es_ekf")
)
pipeline = assemble_pipeline(cfg)
pipeline.reset(trip.gnss_samples[0])

gnss_idx = 0
for imu in trip.imu_samples[:1300]:
    t = imu.timestamp_ns
    while gnss_idx < len(trip.gnss_samples) and trip.gnss_samples[gnss_idx].timestamp_ns <= t:
        g = trip.gnss_samples[gnss_idx]
        print(f"GNSS fix @ t={(g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9:.1f}s: spd={g.speed_mps}, brg={g.bearing_deg}")
        res = pipeline.process_gnss(g)
        print(f"   -> EKF Pos after GNSS: {res.position_enu_m}, heading: {np.degrees(res.heading_rad):.1f}")
        gnss_idx += 1
    calib, vel, fused, matched = pipeline.process_imu(imu)
    if int((t - trip.imu_samples[0].timestamp_ns)*1e-9) % 20 == 0 and (t - trip.imu_samples[0].timestamp_ns) % 1e9 < 1e8:
        print(f"t={(t - trip.imu_samples[0].timestamp_ns)*1e-9:.1f}s: EKF Pos={fused.position_enu_m}, vel={fused.velocity_enu_mps}, heading={np.degrees(fused.heading_rad):.1f}")
