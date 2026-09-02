import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0_ns = trip.imu_samples[0].timestamp_ns

# Let's inspect IMU samples around 330s-350s
samples = [s for s in trip.imu_samples if 330.0 <= (s.timestamp_ns - t0_ns)*1e-9 <= 350.0]
df_s = pd.DataFrame([{
    "t_s": (s.timestamp_ns - t0_ns)*1e-9,
    "gx": s.gyro[0], "gy": s.gyro[1], "gz": s.gyro[2],
    "ax": s.accel[0], "ay": s.accel[1], "az": s.accel[2],
} for s in samples])

print("Gyro values during the right turn at 330s-350s:")
print(df_s[["t_s", "gx", "gy", "gz"]].describe())
print("\nMean gyro during turn:")
print(df_s[["gx", "gy", "gz"]].mean())
