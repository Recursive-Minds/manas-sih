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
gnss_bearings = np.array([g.bearing_deg for g in trip.gnss_samples])
gnss_speeds = np.array([g.speed_mps for g in trip.gnss_samples])
gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)

# Let's inspect the actual trajectory of GNSS during blackout
gt_e = np.interp(np.linspace(bo_start, bo_end, 300), gnss_ts, gnss_enu[:, 0])
gt_n = np.interp(np.linspace(bo_start, bo_end, 300), gnss_ts, gnss_enu[:, 1])

print(f"GT Start Position at 120s: ({gt_e[0]:.1f}, {gt_n[0]:.1f})")
print(f"GT End Position at 150s:   ({gt_e[-1]:.1f}, {gt_n[-1]:.1f})")
print(f"GT Displacement Vector:    DeltaE = {gt_e[-1] - gt_e[0]:.1f}m, DeltaN = {gt_n[-1] - gt_n[0]:.1f}m")
print(f"GT Net Straight Distance:  {np.sqrt((gt_e[-1]-gt_e[0])**2 + (gt_n[-1]-gt_n[0])**2):.1f}m")

# Look at the ground truth bearings from GNSS:
for g in trip.gnss_samples:
    if bo_start - 5e9 <= g.timestamp_ns <= bo_end + 5e9:
        t_s = (g.timestamp_ns - t0) * 1e-9
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, g.altitude_m, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)
        print(f"t={t_s:5.1f}s | Pos=({enu[0]:6.1f}, {enu[1]:6.1f}) | Speed={g.speed_mps:5.2f}m/s | Bearing={g.bearing_deg:5.1f}°")
