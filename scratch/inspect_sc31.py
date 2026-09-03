"""
Detailed inspection of Scenario #31 pre-blackout kinematics.
"""
import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.data.geo import geodetic_to_enu

loader = GenericDataLoader()
trip = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv")
valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
calibrator = MountCalibrator(window_size=100)
for g in trip.gnss_samples: calibrator.observe_gnss(g)
calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]

df = pd.read_csv(r"C:\Users\carpe\SIH\artifacts\phase4_unseen_sm_benchmark_results.csv")
row = df[df["scenario_id"] == 31].iloc[0]
t_start = float(row["start_time_s"])
t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(t_start * 1e9)

# Show last 4 GNSS fixes before bo_start_ns
print("GNSS fixes around Scenario #31 start:")
for g in valid_gnss:
    dt = (g.timestamp_ns - bo_start_ns) * 1e-9
    if -30.0 <= dt <= 10.0:
        print(f"  t={dt:+.2f}s: lat={g.latitude_deg:.6f}, lon={g.longitude_deg:.6f}, bearing={g.bearing_deg}, speed={g.speed_mps}")

# Check displacement course between consecutive GNSS fixes
g_prev = [g for g in valid_gnss if g.timestamp_ns < bo_start_ns][-1]
g_curr = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
enu_p = geodetic_to_enu(g_prev.latitude_deg, g_prev.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
enu_c = geodetic_to_enu(g_curr.latitude_deg, g_curr.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
disp = enu_c - enu_p
disp_bearing = float(np.degrees(np.arctan2(disp[0], disp[1]))) % 360.0
dt_p = (g_curr.timestamp_ns - g_prev.timestamp_ns) * 1e-9
print(f"\nDisplacement course between last 2 GNSS fixes (dt={dt_p:.2f}s): {disp_bearing:.2f}°")

# Gyro integration over those seconds:
wz_sum = 0.0
for j, imu in enumerate(trip.imu_samples):
    if g_prev.timestamp_ns < imu.timestamp_ns <= g_curr.timestamp_ns:
        dt = (imu.timestamp_ns - trip.imu_samples[j-1].timestamp_ns) * 1e-9
        wz_sum += np.degrees(calib_samples[j].gyro_vehicle[2] * dt)
print(f"Integrated gyro turning between last 2 GNSS fixes: {wz_sum:+.2f}°")
print(f"Previous GNSS bearing ({g_prev.bearing_deg}°) + gyro ({wz_sum:+.2f}°) = {(g_prev.bearing_deg + wz_sum)%360.0:.2f}°")
