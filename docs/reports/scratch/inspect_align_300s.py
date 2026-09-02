import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

calib = MountCalibrator()
gnss_samples_300 = [g for g in trip.gnss_samples if (g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 300.0]
for g in gnss_samples_300:
    calib.observe_gnss(g)
imu_samples_300 = [s for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 300.0]
for s in imu_samples_300:
    calib.update(s)
for g in gnss_samples_300:
    calib.observe_gnss(g)

align = calib.alignment
print("Alignment at 300s:")
print("  yaw_axis_index:", align.yaw_axis_index)
print("  yaw_axis_sign:", align.yaw_axis_sign)
