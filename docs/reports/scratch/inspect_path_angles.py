import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

g_sub = [g for g in trip.gnss_samples if 290.0 <= (g.timestamp_ns - t0)*1e-9 <= 370.0]
lats = np.array([g.latitude_deg for g in g_sub])
lons = np.array([g.longitude_deg for g in g_sub])
alts = np.array([g.altitude_m for g in g_sub])
enu = geodetic_to_enu(lats, lons, alts, lats[0], lons[0], alts[0])

print("GNSS fixes from 290s to 370s:")
for i in range(len(g_sub)):
    t_s = (g_sub[i].timestamp_ns - t0)*1e-9
    de = enu[i, 0] - enu[0, 0]
    dn = enu[i, 1] - enu[0, 1]
    if i > 0:
        segment_angle = np.degrees(np.arctan2(enu[i, 0] - enu[i-1, 0], enu[i, 1] - enu[i-1, 1])) % 360.0
    else:
        segment_angle = g_sub[i].bearing_deg
    print(f"  t={t_s:.1f}s | Speed={g_sub[i].speed_mps:.1f} m/s | GNSS Bearing={g_sub[i].bearing_deg:.1f}° | Path Segment Angle={segment_angle:.1f}° | Pos=({enu[i, 0]:.1f}, {enu[i, 1]:.1f})")
