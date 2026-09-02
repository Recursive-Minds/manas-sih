import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip

raw_csv_s2 = download_iovnbd_trip("S-S2")
df_s2 = pd.read_csv(raw_csv_s2, encoding="latin-1")
t_col = [c for c in df_s2.columns if "time" in c.lower() or "sec" in c.lower() or "index" in c.lower() or "ms" in c.lower()][0]
print(f"Time column: {t_col}")
print(df_s2[t_col].head(10).to_string())
print(df_s2[t_col].tail(10).to_string())

# Check if timestamps are monotonically increasing
t_vals = df_s2[t_col].values
diffs = pd.Series(t_vals).diff()
neg_diffs = diffs[diffs < 0]
print(f"Negative time jumps in S-S2: {len(neg_diffs)}")
if len(neg_diffs) > 0:
    print("Negative jump indices:", neg_diffs.head(5))
