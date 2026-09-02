import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.schema import ColumnMapping, find_column

raw_csv = download_iovnbd_trip("S-S1")
df = pd.read_csv(raw_csv, encoding="latin-1")
mapping = ColumnMapping()

col_gx = find_column(df, mapping.gyro_x)
col_gy = find_column(df, mapping.gyro_y)
col_gz = find_column(df, mapping.gyro_z)

col_ax = find_column(df, mapping.accel_x)
col_ay = find_column(df, mapping.accel_y)
col_az = find_column(df, mapping.accel_z)

print("Column mappings found:")
print(f"Accel X: {col_ax}")
print(f"Accel Y: {col_ay}")
print(f"Accel Z: {col_az}")
print(f"Gyro X: {col_gx}")
print(f"Gyro Y: {col_gy}")
print(f"Gyro Z: {col_gz}")
