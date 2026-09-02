import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.calibration.mount import MountCalibrator
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

calib = MountCalibrator()
for g in trip.gnss_samples[:10]:
    calib.observe_gnss(g)
for imu in trip.imu_samples[:1200]:
    calib.update(imu)
for g in trip.gnss_samples[:10]:
    calib.observe_gnss(g)

align = calib.alignment
print("Alignment after 120s:")
print(f"  yaw_axis_index: {align.yaw_axis_index} (0=x, 1=y, 2=z)")
print(f"  yaw_axis_sign:  {align.yaw_axis_sign}")
