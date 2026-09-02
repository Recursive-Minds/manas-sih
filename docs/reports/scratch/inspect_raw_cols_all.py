import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip

raw_csv = download_iovnbd_trip("S-S1")
df = pd.read_csv(raw_csv, encoding="latin-1")
print("All columns in CSV:")
for i, c in enumerate(df.columns):
    print(f"  {i}: {c}")

# Print first 5 rows of IMU and GPS
cols_to_print = [c for c in df.columns if any(k in c.lower() for k in ["acc", "gyro", "lat", "lon", "speed", "orient"])]
print("\nSample values:")
print(df[cols_to_print].head(3).to_string())
