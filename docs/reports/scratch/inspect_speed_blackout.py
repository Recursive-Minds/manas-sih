import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.models.tcn_attention import TCNAttentionVelocityModel

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")

t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(120 * 1e9)
bo_end = bo_start + int(30 * 1e9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location=device, weights_only=False)
model = TCNAttentionVelocityModel().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
gnss_speeds = np.array([g.speed_mps for g in trip.gnss_samples])

buf = []
records = []
for s in trip.imu_samples:
    t = s.timestamp_ns
    vec = np.hstack([s.accel, s.gyro]).astype(np.float32)
    buf.append(vec)
    if len(buf) > 100:
        buf.pop(0)

    if bo_start <= t <= bo_end:
        w_arr = np.array(buf, dtype=np.float32).T
        w_norm = (w_arr - norm_mean) / norm_std
        x_tensor = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)
        with torch.no_grad():
            speed_pred, log_var = model(x_tensor)
            v_pred = float(speed_pred[0, 0].item())
            var_pred = float(torch.exp(log_var[0, 0]).item())
        
        gt_spd = np.interp(t, gnss_ts, gnss_speeds)
        records.append({
            "time_s": (t - t0) * 1e-9,
            "v_pred": v_pred,
            "v_gt": gt_spd,
            "var_pred": var_pred,
            "err": v_pred - gt_spd
        })

df = pd.DataFrame(records)
print("Velocity Estimation during Blackout:")
print(f"  Mean Pred Speed: {df['v_pred'].mean():.2f} m/s ({df['v_pred'].mean()*3.6:.1f} km/h)")
print(f"  Mean GT Speed:   {df['v_gt'].mean():.2f} m/s ({df['v_gt'].mean()*3.6:.1f} km/h)")
print(f"  Speed RMSE:      {np.sqrt(np.mean(df['err']**2)):.2f} m/s ({np.sqrt(np.mean(df['err']**2))*3.6:.1f} km/h)")
print(f"  Speed MAE:       {np.mean(np.abs(df['err'])):.2f} m/s ({np.mean(np.abs(df['err']))*3.6:.1f} km/h)")

print("\nSample predictions every 3 seconds:")
print(df.iloc[::30][["time_s", "v_pred", "v_gt", "err", "var_pred"]])
