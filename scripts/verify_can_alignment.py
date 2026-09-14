import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np

from sih.data.loader import GenericDataLoader

loader = GenericDataLoader()

for trip_id in ["S-S1", "S-S2", "S-M"]:
    trip = loader.load_file(f"data/raw/iovnbd_trips/{trip_id}.csv")
    v_df = pd.read_csv(f"data/raw/iovnbd_trips/V-{trip_id[2:]}.csv", encoding="latin-1")

    v_cols = {c.strip(): c for c in v_df.columns}
    v_spd_col = v_cols.get("Velocity (km/hr)", v_cols.get("Indicated Vehicle Speed (km/hr)"))
    v_speed_mps = (v_df[v_spd_col].fillna(0).to_numpy() / 3.6).astype(np.float32)

    # Smartphone GPS interpolated speed
    imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.int64)
    gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples if g.speed_mps is not None], dtype=np.int64)
    gnss_spd = np.array([g.speed_mps for g in trip.gnss_samples if g.speed_mps is not None], dtype=np.float32)
    s_interp = np.interp(imu_ts, gnss_ts, gnss_spd).astype(np.float32)

    train_end = int(len(imu_ts) * 0.60)
    s_train = s_interp[:train_end]
    v_train = v_speed_mps[:train_end]

    s_norm = s_train - np.mean(s_train)
    v_norm = v_train - np.mean(v_train)
    corr = np.correlate(v_norm, s_norm, mode="full")
    lags = np.arange(-len(s_train) + 1, len(s_train))
    best_lag = lags[np.argmax(corr)]

    print(f"\nTrip {trip_id}:")
    print(f"  - Mean speeds: V={np.mean(v_train):.2f} m/s, S={np.mean(s_train):.2f} m/s")
    print(f"  - Max speeds:  V={np.max(v_train):.2f} m/s, S={np.max(s_train):.2f} m/s")
    print(f"  - Best lag (V relative to S): {best_lag} ticks ({best_lag*0.1:+.2f} s)")
    print(f"  - Zero-lag RMSE: {np.sqrt(np.mean((v_train - s_train)**2)):.2f} m/s")
    # Shifted RMSE
    if best_lag > 0:
        shifted_rmse = np.sqrt(np.mean((v_train[best_lag:] - s_train[:-best_lag])**2))
    elif best_lag < 0:
        shifted_rmse = np.sqrt(np.mean((v_train[:best_lag] - s_train[-best_lag:])**2))
    else:
        shifted_rmse = np.sqrt(np.mean((v_train - s_train)**2))
    print(f"  - Best-lag RMSE: {shifted_rmse:.2f} m/s")
