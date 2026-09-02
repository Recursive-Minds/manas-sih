import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

def test_at(t_max):
    calib = MountCalibrator()
    for g in [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= t_max]:
        calib.observe_gnss(g)
    for s in [s for s in trip.imu_samples if (s.timestamp_ns - t0)*1e-9 <= t_max]:
        calib.update(s)
    for g in [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= t_max]:
        calib.observe_gnss(g)
    align = calib.alignment
    print(f"At t={t_max:.1f}s: yaw_idx={align.yaw_axis_index}, yaw_sign={align.yaw_axis_sign}")

test_at(120.0)
test_at(300.0)
