import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator

def main():
    print("=== RESEARCH: GROUND TRUTH YAW RATE & HEADING DELTA EXTRACTION ===", flush=True)

    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")

    calibrator = MountCalibrator(window_size=100)
    for g in trip_s1.gnss_samples: calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip_s1.imu_samples]

    # Extract calibrated gyro z (vehicle yaw rate) and GNSS bearings
    imu_ts = np.array([s.timestamp_ns for s in trip_s1.imu_samples], dtype=np.int64)
    w_z_veh = np.array([c.gyro_vehicle[2] for c in calib_samples], dtype=np.float64)

    gnss_valid = [g for g in trip_s1.gnss_samples if g.speed_mps is not None and g.speed_mps > 2.0 and g.bearing_deg is not None]
    g_ts = np.array([g.timestamp_ns for g in gnss_valid], dtype=np.int64)
    g_brg = np.array([g.bearing_deg for g in gnss_valid], dtype=np.float64)

    # Unwrap bearings and compute GNSS derivative turn rate (rad/s)
    g_brg_rad = np.unwrap(np.radians(g_brg))
    g_dt = np.diff(g_ts) / 1e9
    g_w_z = np.diff(g_brg_rad) / np.maximum(g_dt, 1e-3)
    g_mid_ts = (g_ts[:-1] + g_ts[1:]) // 2

    # Match IMU vehicle w_z with GNSS turn rate
    matched_w_z_veh = []
    matched_g_w_z   = []

    for i in range(len(g_w_z)):
        t_mid = g_mid_ts[i]
        # Find IMU index closest to t_mid
        idx_imu = np.argmin(np.abs(imu_ts - t_mid))
        if abs(imu_ts[idx_imu] - t_mid) < 2e8: # Within 200ms
            # Average IMU w_z over 1s window centered at t_mid
            mask_w = np.abs(imu_ts - t_mid) < 5e8
            if np.sum(mask_w) > 5:
                w_mean = np.mean(w_z_veh[mask_w])
                matched_w_z_veh.append(w_mean)
                matched_g_w_z.append(-g_w_z[i]) # Note ENU CCW vs bearing sign

    matched_w_z_veh = np.array(matched_w_z_veh)
    matched_g_w_z   = np.array(matched_g_w_z)

    print(f"Total matched turn-rate evaluation pairs: {len(matched_w_z_veh)}")
    if len(matched_w_z_veh) > 0:
        corr = np.corrcoef(matched_w_z_veh, matched_g_w_z)[0, 1]
        rmse = np.sqrt(np.mean((matched_w_z_veh - matched_g_w_z)**2))
        print(f"Correlation (Calibrated Vehicle Gyro Yaw Rate vs GNSS Turn Rate): {corr:.4f}")
        print(f"Turn Rate RMSE: {np.degrees(rmse):.4f} deg/s ({rmse:.6f} rad/s)")

if __name__ == "__main__":
    main()
