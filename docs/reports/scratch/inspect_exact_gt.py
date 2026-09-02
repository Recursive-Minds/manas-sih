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
gnss_speeds = np.array([g.speed_mps for g in trip.gnss_samples])
gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)

# Interpolated ground truth speed and heading
times = np.linspace(bo_start, bo_end, 31)
for t in times:
    t_s = (t - t0) * 1e-9
    # Compute GT position at t-0.5s and t+0.5s to get exact GT heading and speed
    e1 = np.interp(t - 0.5e9, gnss_ts, gnss_enu[:, 0])
    n1 = np.interp(t - 0.5e9, gnss_ts, gnss_enu[:, 1])
    e2 = np.interp(t + 0.5e9, gnss_ts, gnss_enu[:, 0])
    n2 = np.interp(t + 0.5e9, gnss_ts, gnss_enu[:, 1])
    
    de = e2 - e1
    dn = n2 - n1
    speed_gt = np.sqrt(de**2 + dn**2)
    heading_gt = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    print(f"t={t_s:5.1f}s | GT Pos=({e1:6.1f}, {n1:6.1f}) | Speed GT={speed_gt:5.2f}m/s | Heading GT={heading_gt:5.1f}°")
