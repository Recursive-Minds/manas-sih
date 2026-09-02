import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.fusion.es_ekf import ErrorStateEKF
from sih.velocity.ai_estimator import AIVelocityEstimator

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(300.0 * 1e9)
bo_end = t0 + int(360.0 * 1e9)

calib = MountCalibrator()
ekf = ErrorStateEKF()
vel_est = AIVelocityEstimator(checkpoint_path="models/checkpoints/best_velocity_model.pt", device="cuda:0")

# Run up to 300s
for s in trip.imu_samples:
    if s.timestamp_ns > bo_end:
        break
    
    # GNSS fixes
    for g in trip.gnss_samples:
        if ekf._last_gnss_ts is None or g.timestamp_ns > ekf._last_gnss_ts:
            if g.timestamp_ns <= s.timestamp_ns and g.timestamp_ns <= bo_start:
                calib.observe_gnss(g)
                ekf.update_gnss(g)
                
    c_s = calib.update(s)
    v_s = vel_est.estimate(c_s)
    
    # Predict
    st = ekf.predict(c_s, v_s)
    
    if bo_start <= s.timestamp_ns <= bo_end:
        t_sec = (s.timestamp_ns - t0)*1e-9
        if int(t_sec * 10) % 50 == 0: # every 5s
            print(f"t={t_sec:.1f}s | Heading={np.degrees(ekf._heading_rad):.1f}° | Gyro_v_z={c_s.gyro_vehicle[2]:.4f} rad/s | Pos=({st.position_enu_m[0]:.1f}, {st.position_enu_m[1]:.1f})")
