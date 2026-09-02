import os
import sys
import numpy as np
import pandas as pd

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
gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples])
gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)

# Gyro Z bias
bg_z = -0.000100

heading_rad = np.radians(272.0)
last_t = bo_start
records = []

for imu in trip.imu_samples:
    t = imu.timestamp_ns
    if t < bo_start:
        continue
    if t > bo_end:
        break
    dt = (t - last_t) * 1e-9
    last_t = t
    
    gz = imu.gyro[2] - bg_z
    # Gyro yaw rate integration:
    # If vehicle turns RIGHT (from 272° to 315° = clockwise), does gyro Z measure positive or negative?
    heading_rad = (heading_rad - gz * dt) % (2 * np.pi)
    
    if len(records) % 10 == 0:
        t_s = (t - t0) * 1e-9
        # GT heading
        e1 = np.interp(t - 0.5e9, gnss_ts, gnss_enu[:, 0])
        n1 = np.interp(t - 0.5e9, gnss_ts, gnss_enu[:, 1])
        e2 = np.interp(t + 0.5e9, gnss_ts, gnss_enu[:, 0])
        n2 = np.interp(t + 0.5e9, gnss_ts, gnss_enu[:, 1])
        heading_gt = (np.degrees(np.arctan2(e2 - e1, n2 - n1)) + 360.0) % 360.0
        print(f"t={t_s:5.1f}s | gz={gz:6.3f} rad/s | Heading Est={np.degrees(heading_rad):5.1f}° | Heading GT={heading_gt:5.1f}° | Diff={np.degrees(heading_rad)-heading_gt:5.1f}°")
    records.append(1)
