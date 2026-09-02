import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip

csv_path = download_iovnbd_trip("S-S1")
df = pd.read_csv(csv_path, encoding="latin-1", nrows=50)

print("Columns:")
for c in df.columns:
    print(f"'{c}'")

# Look at correlations between orientation and gyro / accel
df_full = pd.read_csv(csv_path, encoding="latin-1")
print("\nSample stats:")
print("ACCEL X mean, std:", df_full[' ACCELEROMETER X (m/s) '].mean(), df_full[' ACCELEROMETER X (m/s) '].std())
print("ACCEL Y mean, std:", df_full[' ACCELEROMETER Y (m/s)'].mean(), df_full[' ACCELEROMETER Y (m/s)'].std())
print("ACCEL Z mean, std:", df_full[' ACCELEROMETER Z (m/s)'].mean(), df_full[' ACCELEROMETER Z (m/s)'].std())
print("GYRO Roll mean, std:", df_full[' GYROSCOPE Roll (rad/s)'].mean(), df_full[' GYROSCOPE Roll (rad/s)'].std())
print("GYRO Pitch mean, std:", df_full[' GYROSCOPE Pitch (rad/s)'].mean(), df_full[' GYROSCOPE Pitch (rad/s)'].std())
print("GYRO Yaw mean, std:", df_full[' GYROSCOPE Yaw (rad/s)'].mean(), df_full[' GYROSCOPE Yaw (rad/s)'].std())
