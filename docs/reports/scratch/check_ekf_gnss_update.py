import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.fusion.es_ekf import ErrorStateEKF

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
ekf = ErrorStateEKF()
ekf.init_from_gnss(trip.gnss_samples[0])

for g in trip.gnss_samples[1:10]:
    t = (g.timestamp_ns - trip.gnss_samples[0].timestamp_ns) * 1e-9
    st_before = ekf.get_state()
    ekf.update_gnss(g)
    st_after = ekf.get_state()
    print(f"t={t:.1f}s | GNSS: spd={g.speed_mps:.1f}, brg={g.bearing_deg:.1f}°")
    print(f"   Before: pos={st_before.position_enu_m[:2]}, hdg={np.degrees(st_before.heading_rad):.1f}°")
    print(f"   After:  pos={st_after.position_enu_m[:2]}, hdg={np.degrees(st_after.heading_rad):.1f}°")
