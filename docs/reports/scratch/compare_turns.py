import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0_ns = trip.imu_samples[0].timestamp_ns

# Scenario 1 turn (135s-145s)
s1_gy = [s.gyro[1] for s in trip.imu_samples if 135.0 <= (s.timestamp_ns - t0_ns)*1e-9 <= 145.0]
s1_ax = [s.accel[0] for s in trip.imu_samples if 135.0 <= (s.timestamp_ns - t0_ns)*1e-9 <= 145.0]

# Scenario 2 turn (330s-350s)
s2_gy = [s.gyro[1] for s in trip.imu_samples if 330.0 <= (s.timestamp_ns - t0_ns)*1e-9 <= 350.0]
s2_ax = [s.accel[0] for s in trip.imu_samples if 330.0 <= (s.timestamp_ns - t0_ns)*1e-9 <= 350.0]

print("Scenario 1 (135s-145s) Right turn:")
print(f"  Mean gy: {np.mean(s1_gy):.4f}, Sum gy*dt: {np.sum(s1_gy)*0.1*180/np.pi:.2f}°")
print(f"  Mean ax: {np.mean(s1_ax):.4f}")

print("\nScenario 2 (330s-350s) Right turn:")
print(f"  Mean gy: {np.mean(s2_gy):.4f}, Sum gy*dt: {np.sum(s2_gy)*0.1*180/np.pi:.2f}°")
print(f"  Mean ax: {np.mean(s2_ax):.4f}")
