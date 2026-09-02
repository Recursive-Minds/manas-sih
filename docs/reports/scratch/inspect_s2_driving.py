import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))
t0 = trip_s2.imu_samples[0].timestamp_ns

moving_fixes = [g for g in trip_s2.gnss_samples if (g.speed_mps or 0) > 3.0]
first_move = moving_fixes[0]
print(f"First moving GNSS fix in S-S2: t={(first_move.timestamp_ns - t0)*1e-9:.1f}s, speed={first_move.speed_mps:.2f} m/s, bearing={first_move.bearing_deg:.2f}°")

# Let's inspect intervals in S-S2 where the car is actively driving for 100+ seconds
g_df = pd.DataFrame([{
    "t_s": (g.timestamp_ns - t0)*1e-9,
    "speed": g.speed_mps,
    "bearing": g.bearing_deg
} for g in trip_s2.gnss_samples if (g.speed_mps or 0) > 5.0])
print(g_df.head(10).to_string())
