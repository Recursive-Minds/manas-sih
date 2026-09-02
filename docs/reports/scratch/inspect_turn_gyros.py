import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")

t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(120 * 1e9)
bo_end = bo_start + int(30 * 1e9)

for s in trip.imu_samples:
    t_s = (s.timestamp_ns - t0) * 1e-9
    if 136.0 <= t_s <= 142.0:
        print(f"t={t_s:5.1f}s | Accel=({s.accel[0]:5.2f}, {s.accel[1]:5.2f}, {s.accel[2]:5.2f}) | Gyro: gx={s.gyro[0]:6.3f}, gy={s.gyro[1]:6.3f}, gz={s.gyro[2]:6.3f}")
