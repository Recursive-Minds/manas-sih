import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

def find_mount_rotation(trip, max_t_s=120.0):
    # 1. Leveling from gravity
    accels = np.array([s.accel for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 < max_t_s])
    g_body = np.mean(accels, axis=0)
    g_unit = g_body / np.linalg.norm(g_body)
    
    # In S-S1, we saw that accelerometer was logged with Z vertical, but gyro had yaw on Y axis!
    # Let's check which gyro axis correlates with GNSS heading turns or lateral accel
    g_samples = [g for g in trip.gnss_samples if (g.timestamp_ns - trip.gnss_samples[0].timestamp_ns)*1e-9 < max_t_s and (g.speed_mps or 0) > 2.0]
    
    # Calculate GNSS turns
    g_ts = np.array([g.timestamp_ns for g in g_samples])
    g_brg = np.array([g.bearing_deg for g in g_samples])
    g_spd = np.array([g.speed_mps for g in g_samples])
    
    dt_g = np.diff(g_ts) * 1e-9
    dbrg = np.diff(np.unwrap(np.radians(g_brg)))
    turn_rates = dbrg / np.maximum(dt_g, 1.0)
    
    # Gyro samples
    imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 < max_t_s])
    gx = np.array([s.gyro[0] for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 < max_t_s])
    gy = np.array([s.gyro[1] for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 < max_t_s])
    gz = np.array([s.gyro[2] for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 < max_t_s])
    
    g_mids = (g_ts[:-1] + g_ts[1:]) // 2
    gx_mid = np.interp(g_mids, imu_ts, gx)
    gy_mid = np.interp(g_mids, imu_ts, gy)
    gz_mid = np.interp(g_mids, imu_ts, gz)
    
    # Turn rate is d(bearing)/dt. When vehicle turns right (d(bearing)/dt > 0), check which gyro axis has the highest positive correlation
    corrs = {
        "-y": np.corrcoef(turn_rates, -gy_mid)[0, 1],
        "+y": np.corrcoef(turn_rates, gy_mid)[0, 1],
        "-z": np.corrcoef(turn_rates, -gz_mid)[0, 1],
        "+z": np.corrcoef(turn_rates, gz_mid)[0, 1],
        "-x": np.corrcoef(turn_rates, -gx_mid)[0, 1],
        "+x": np.corrcoef(turn_rates, gx_mid)[0, 1],
    }
    print("Pre-120s turn rate correlations:", corrs)

find_mount_rotation(trip, 120.0)
