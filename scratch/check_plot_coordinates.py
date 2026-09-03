import sys
import numpy as np
sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

loader = GenericDataLoader()
trip = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")

t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(1956.0 * 1e9)
warmup_start_ns = bo_start_ns - int(30.0 * 1e9)

warmup_gnss = min([g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns))
bo_gnss = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_start_ns))

enu_trip_ref = geodetic_to_enu(bo_gnss.latitude_deg, bo_gnss.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
enu_warmup_ref = geodetic_to_enu(bo_gnss.latitude_deg, bo_gnss.longitude_deg, 0.0, warmup_gnss.latitude_deg, warmup_gnss.longitude_deg, 0.0)[:2]

print(f"Trip Reference Lat/Lon: {trip.reference_lat_deg:.6f}, {trip.reference_lon_deg:.6f}")
print(f"Warmup GNSS Lat/Lon:    {warmup_gnss.latitude_deg:.6f}, {warmup_gnss.longitude_deg:.6f}")
print(f"Blackout GNSS Lat/Lon:  {bo_gnss.latitude_deg:.6f}, {bo_gnss.longitude_deg:.6f}")
print(f"Blackout point in Trip Reference ENU:   {enu_trip_ref}")
print(f"Blackout point in Warmup Reference ENU: {enu_warmup_ref}")
