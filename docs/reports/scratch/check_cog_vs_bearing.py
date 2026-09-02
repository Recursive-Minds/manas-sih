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

gnss = trip.gnss_samples
print(f"Total GNSS fixes: {len(gnss)}")

records = []
for i in range(len(gnss) - 1):
    g1 = gnss[i]
    g2 = gnss[i+1]
    dt = (g2.timestamp_ns - g1.timestamp_ns) * 1e-9
    if dt < 0.1 or dt > 30.0:
        continue
    
    enu1 = geodetic_to_enu(g1.latitude_deg, g1.longitude_deg, g1.altitude_m, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)
    enu2 = geodetic_to_enu(g2.latitude_deg, g2.longitude_deg, g2.altitude_m, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)
    
    de = enu2[0] - enu1[0]
    dn = enu2[1] - enu1[1]
    dist = np.sqrt(de**2 + dn**2)
    cog_deg = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    
    diff = (cog_deg - g1.bearing_deg + 180.0) % 360.0 - 180.0 if g1.bearing_deg is not None else 0.0
    records.append({
        "t_s": (g1.timestamp_ns - trip.imu_samples[0].timestamp_ns) * 1e-9,
        "dt": dt,
        "dist": dist,
        "speed_calc": dist / dt,
        "speed_rep": g1.speed_mps,
        "cog_deg": cog_deg,
        "bearing_rep": g1.bearing_deg,
        "diff_bearing": diff
    })

df = pd.DataFrame(records)
print(f"Computed records: {len(df)}")
df_valid = df[df["speed_calc"] > 5.0]
print("Mean absolute bearing diff when moving:", df_valid["diff_bearing"].abs().mean())
print("Median absolute bearing diff:", df_valid["diff_bearing"].abs().median())
print("\nSample records:")
print(df_valid[["t_s", "dt", "dist", "speed_calc", "speed_rep", "cog_deg", "bearing_rep", "diff_bearing"]].head(20))
