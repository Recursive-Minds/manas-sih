import os
import sys

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

calib1 = MountCalibrator()
for g in [g for g in trip.gnss_samples if (g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 120.0]:
    calib1.observe_gnss(g)
for imu in [s for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 120.0]:
    calib1.update(imu)
for g in [g for g in trip.gnss_samples if (g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 120.0]:
    calib1.observe_gnss(g)

print("Alignment at 120s:")
print(f"  yaw_idx: {calib1.alignment.yaw_axis_index}, yaw_sign: {calib1.alignment.yaw_axis_sign}")

calib2 = MountCalibrator()
for g in [g for g in trip.gnss_samples if (g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 300.0]:
    calib2.observe_gnss(g)
for imu in [s for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 300.0]:
    calib2.update(imu)
for g in [g for g in trip.gnss_samples if (g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 300.0]:
    calib2.observe_gnss(g)

print("Alignment at 300s:")
print(f"  yaw_idx: {calib2.alignment.yaw_axis_index}, yaw_sign: {calib2.alignment.yaw_axis_sign}")
