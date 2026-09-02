import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")

t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(120 * 1e9)
bo_end = bo_start + int(30 * 1e9)

gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
gnss_speeds = np.array([g.speed_mps for g in trip.gnss_samples])
gnss_bearings = np.array([g.bearing_deg for g in trip.gnss_samples])

idx_bo = np.where((gnss_ts >= bo_start - 10e9) & (gnss_ts <= bo_end + 10e9))[0]
print("GNSS fixes around blackout at 120s:")
for i in idx_bo:
    rel_t = (gnss_ts[i] - t0) * 1e-9
    is_in = " [BLACKOUT]" if bo_start <= gnss_ts[i] <= bo_end else ""
    print(f"  t={rel_t:6.1f}s | Speed={gnss_speeds[i]:5.2f} m/s ({gnss_speeds[i]*3.6:5.1f} km/h) | Bearing={gnss_bearings[i]:5.1f} deg{is_in}")
