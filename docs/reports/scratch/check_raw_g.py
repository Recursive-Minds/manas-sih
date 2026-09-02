import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
accels_all = np.array([s.accel for s in trip.imu_samples[:500]])
mean_acc = np.mean(accels_all, axis=0)
print(f"Mean raw accelerometer: X={mean_acc[0]:.2f}, Y={mean_acc[1]:.2f}, Z={mean_acc[2]:.2f}")
