import os
import sys
import pandas as pd
import numpy as np

csv_path = "data/raw/iovnbd_trips/S-S1.csv"
df = pd.read_csv(csv_path, encoding="latin-1")

print("Shape:", df.shape)
for i, col in enumerate(df.columns):
    print(f"Col {i}: {repr(col)}")

print("\nSample values at rows 1000-1005:")
for col in df.columns:
    print(f"  {col.strip()}: {df[col].iloc[1000]}")
