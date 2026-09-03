"""
Inspect Scenario #20 trajectory, candidates, and ground truth path.
"""
import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")
import numpy as np
import pandas as pd
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork

loader = GenericDataLoader()
trip = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv")
valid_gnss = [g for g in trip.gnss_samples if g.is_valid]

df = pd.read_csv(r"C:\Users\carpe\SIH\artifacts\phase4_unseen_sm_benchmark_results.csv")
row = df[df["scenario_id"] == 20].iloc[0]
t_start = float(row["start_time_s"])
dur = float(row["duration_s"])
t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(t_start * 1e9)
bo_end_ns = bo_start_ns + int(dur * 1e9)

gt_samples = [g for g in valid_gnss if bo_start_ns <= g.timestamp_ns <= bo_end_ns]
pts_enu = np.array([geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2] for g in gt_samples])

print("Scenario #20 Ground Truth Path:")
print(f"Start ENU: {pts_enu[0]}, End ENU: {pts_enu[-1]}")
disp = pts_enu[-1] - pts_enu[0]
print(f"Overall displacement: East={disp[0]:.1f}m, North={disp[1]:.1f}m, dist={np.linalg.norm(disp):.1f}m, bearing={np.degrees(np.arctan2(disp[0], disp[1]))%360.0:.1f}°")

# Bearings along the path
for i in range(len(pts_enu)-1):
    d = pts_enu[i+1] - pts_enu[i]
    b = np.degrees(np.arctan2(d[0], d[1])) % 360.0
    print(f"  Step {i} -> {i+1}: dist={np.linalg.norm(d):.1f}m, bearing={b:.1f}°")
