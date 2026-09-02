import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.fusion.es_ekf import ErrorStateEKF
from sih.velocity.ai_estimator import AIVelocityEstimator

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(300.0 * 1e9)

calib = MountCalibrator()
ekf = ErrorStateEKF()
vel_est = AIVelocityEstimator(checkpoint_path="models/checkpoints/best_velocity_model.pt", device="cuda:0")

for s in trip.imu_samples:
    if s.timestamp_ns > bo_start:
        break
    for g in trip.gnss_samples:
        if ekf._last_gnss_ts is None or g.timestamp_ns > ekf._last_gnss_ts:
            if g.timestamp_ns <= s.timestamp_ns:
                calib.observe_gnss(g)
                ekf.update_gnss(g)
    c_s = calib.update(s)
    v_s = vel_est.estimate(c_s)
    ekf.predict(c_s, v_s)

print(f"EKF state at 300s:")
print(f"  _bg (gyro bias): {ekf._bg}")
print(f"  _bg[2] in deg/s: {np.degrees(ekf._bg[2]):.4f} deg/s")
print(f"  _heading_rad: {np.degrees(ekf._heading_rad):.2f}°")
print(f"  _p: {ekf._p}")
