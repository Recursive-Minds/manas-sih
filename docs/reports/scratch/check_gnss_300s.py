import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
g_brg = np.array([g.bearing_deg for g in trip.gnss_samples])
g_spd = np.array([g.speed_mps for g in trip.gnss_samples])

t0 = trip.imu_samples[0].timestamp_ns
bo300_ns = t0 + int(300.0 * 1e9)

# Find GNSS fixes right before 300s
pre = [g for g in trip.gnss_samples if g.timestamp_ns <= bo300_ns]
print("Last 3 GNSS fixes before 300s:")
for g in pre[-3:]:
    t_s = (g.timestamp_ns - t0)*1e-9
    print(f"  t={t_s:.1f}s, speed={g.speed_mps:.2f} m/s, bearing={g.bearing_deg:.2f}°")
