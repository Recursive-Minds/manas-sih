import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))
t0 = trip_s2.imu_samples[0].timestamp_ns

g_samples_120 = [g for g in trip_s2.gnss_samples if 110.0 <= (g.timestamp_ns - t0)*1e-9 <= 160.0]
for g in g_samples_120:
    t_s = (g.timestamp_ns - t0)*1e-9
    print(f"S-S2 GNSS fix: t={t_s:.1f}s | Speed={g.speed_mps} m/s | Bearing={g.bearing_deg}°")
