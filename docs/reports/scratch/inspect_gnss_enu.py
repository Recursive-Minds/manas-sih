import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.data.geo import geodetic_to_enu

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
g0 = trip.gnss_samples[0]
for g in trip.gnss_samples[:10]:
    enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, g.altitude_m, g0.latitude_deg, g0.longitude_deg, g0.altitude_m)
    print(f"t={(g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9:.1f}s: GNSS lat={g.latitude_deg}, lon={g.longitude_deg} -> ENU={enu}")
