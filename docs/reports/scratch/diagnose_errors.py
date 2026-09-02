import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.dataset import IMUVelocityDataset

def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    loader = GenericDataLoader()
    trips = {}
    for name in ["S-S1", "S-S2", "S-M"]:
        csv_path = download_iovnbd_trip(name)
        trip = loader.load_file(csv_path)
        trips[name] = trip
        print(f"Loaded {name}: {len(trip.imu_samples)} IMU samples, {len(trip.gnss_samples)} GNSS samples, dist={trip.total_gnss_distance_m:.1f}m, dur={trip.duration_s:.1f}s")
        
        # Check IMU statistics
        accels = np.array([s.accel for s in trip.imu_samples])
        gyros = np.array([s.gyro for s in trip.imu_samples])
        print(f"  {name} Accel mean: {np.mean(accels, axis=0)}, norm: {np.mean(np.linalg.norm(accels, axis=1)):.3f}")
        print(f"  {name} Gyro  mean: {np.mean(gyros, axis=0)}, norm: {np.mean(np.linalg.norm(gyros, axis=1)):.3f}")

    # Inspect checkpoint
    ckpt_path = "models/checkpoints/best_velocity_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print("\nCheckpoint metadata:")
        print("  Epoch:", ckpt.get("epoch"))
        print("  Val RMSE:", ckpt.get("val_rmse"))
        print("  Norm Mean:", ckpt.get("norm_mean").ravel())
        print("  Norm Std:", ckpt.get("norm_std").ravel())
        print("  Config:", ckpt.get("config"))
        
        model = TCNAttentionVelocityModel().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        # Evaluate model on S-S1, S-S2, S-M
        for name, trip in trips.items():
            ds = IMUVelocityDataset([trip], window_size=100, step_size=5, mean=ckpt["norm_mean"], std=ckpt["norm_std"])
            dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
            preds, targets = [], []
            with torch.no_grad():
                for x, y in dl:
                    x = x.to(device)
                    p, _ = model(x)
                    preds.extend(p.cpu().squeeze().numpy())
                    targets.extend(y.numpy())
            preds = np.array(preds)
            targets = np.array(targets)
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            mae = np.mean(np.abs(preds - targets))
            corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
            print(f"\nModel Performance on {name}:")
            print(f"  Target Speed Mean: {np.mean(targets):.2f} m/s, Max: {np.max(targets):.2f} m/s")
            print(f"  Pred Speed Mean:   {np.mean(preds):.2f} m/s, Max: {np.max(preds):.2f} m/s")
            print(f"  RMSE: {rmse:.3f} m/s ({rmse*3.6:.2f} km/h), MAE: {mae:.3f} m/s ({mae*3.6:.2f} km/h)")
            print(f"  Pearson Correlation: {corr:.3f}")

if __name__ == "__main__":
    diagnose()
