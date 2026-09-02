import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip

raw_csv = download_iovnbd_trip("S-S1")
df = pd.read_csv(raw_csv, encoding="latin-1")
grav_cols = [c for c in df.columns if "gravity" in c.lower()]
print(df[grav_cols].describe())
