import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

samples = [s for s in trip.imu_samples if 300.0 <= (s.timestamp_ns - t0)*1e-9 <= 360.0]
df_s = pd.DataFrame([{
    "t_s": (s.timestamp_ns - t0)*1e-9,
    "gx": s.gyro[0], "gy": s.gyro[1], "gz": s.gyro[2],
    "ax": s.accel[0], "ay": s.accel[1], "az": s.accel[2],
} for s in samples])

print("Mean and sum of gyros between 300s and 360s:")
print(f"gx: mean={df_s['gx'].mean():.4f}, sum*dt*180/pi={df_s['gx'].sum()*0.1*180/np.pi:.2f}°")
print(f"gy: mean={df_s['gy'].mean():.4f}, sum*dt*180/pi={df_s['gy'].sum()*0.1*180/np.pi:.2f}°")
print(f"gz: mean={df_s['gz'].mean():.4f}, sum*dt*180/pi={df_s['gz'].sum()*0.1*180/np.pi:.2f}°")
