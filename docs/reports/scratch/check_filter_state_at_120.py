import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig
from sih.core.pipeline import assemble_pipeline

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")

p = assemble_pipeline(PipelineConfig(fusion=FusionFilterConfig(algorithm="es_ekf")))
p.reset(initial_gnss=trip.gnss_samples[0])

gnss_idx = 0
n_gnss = len(trip.gnss_samples)
t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(120 * 1e9)

for imu in trip.imu_samples:
    t = imu.timestamp_ns
    if t >= bo_start:
        break
    while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t:
        p.process_gnss(trip.gnss_samples[gnss_idx])
        gnss_idx += 1
    p.process_imu(imu)

filter_obj = p.fusion_filter
print(f"Filter State at t=120s (Blackout start):")
print(f"  ba (accel bias): {filter_obj.b_a}")
print(f"  bg (gyro bias):  {filter_obj.b_g} rad/s ({np.degrees(filter_obj.b_g)} deg/s)")
print(f"  Pos: {filter_obj.p}")
print(f"  Vel: {filter_obj.v} norm={np.linalg.norm(filter_obj.v):.2f} m/s")
C = filter_obj.q.as_matrix()
print(f"  C (Rotation):\n{C}")
yaw_enu = np.arctan2(C[1, 0], C[0, 0])
print(f"  Heading: {(np.pi/2 - yaw_enu)% (2*np.pi) * 180 / np.pi:.1f} deg")
