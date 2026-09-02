import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.models.tcn_attention import TCNAttentionVelocityModel, gaussian_nll_loss

class FastAugmentedDataset(Dataset):
    def __init__(self, trips, window_size=100, step_size=2, is_train=True, mean=None, std=None):
        self.window_size = window_size
        self.step_size = step_size
        self.is_train = is_train
        
        all_windows = []
        all_targets = []
        
        for trip in trips:
            if len(trip.imu_samples) < window_size or len(trip.gnss_samples) == 0:
                continue
            imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.int64)
            accels = np.array([s.accel for s in trip.imu_samples], dtype=np.float32)
            gyros = np.array([s.gyro for s in trip.imu_samples], dtype=np.float32)
            
            # GNSS speeds
            gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
            gnss_speeds = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in trip.gnss_samples], dtype=np.float32)
            interp_speeds = np.interp(imu_ts, gnss_ts, gnss_speeds).astype(np.float32)
            interp_speeds[interp_speeds < 0.2] = 0.0
            
            raw_6 = np.hstack([accels, gyros]) # (N, 6)
            
            # Fast numpy stride rolling windows
            n_samples = len(raw_6)
            num_windows = (n_samples - window_size) // step_size + 1
            if num_windows <= 0:
                continue
                
            # Shape: (num_windows, window_size, 6)
            sub_windows = np.lib.stride_tricks.sliding_window_view(raw_6, window_shape=(window_size, 6))[::step_size, 0, :, :]
            # Transpose to (num_windows, 6, window_size)
            sub_windows = np.transpose(sub_windows, (0, 2, 1))
            
            sub_targets = interp_speeds[window_size - 1 :: step_size][:num_windows]
            
            all_windows.append(sub_windows)
            all_targets.append(sub_targets)
            
        self.windows = np.concatenate(all_windows, axis=0) # (N, 6, 100)
        self.targets = np.concatenate(all_targets, axis=0) # (N,)
        
        if mean is None or std is None:
            norms_a = np.linalg.norm(self.windows[:, :3, :], axis=1, keepdims=True)
            norms_w = np.linalg.norm(self.windows[:, 3:6, :], axis=1, keepdims=True)
            all_8 = np.concatenate([self.windows, norms_a, norms_w], axis=1) # (N, 8, 100)
            self.mean = np.mean(all_8, axis=(0, 2), keepdims=True)[0] # (8, 1)
            self.std = np.std(all_8, axis=(0, 2), keepdims=True)[0] # (8, 1)
            self.std[self.std < 1e-5] = 1.0
        else:
            self.mean = mean
            self.std = std
            
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        w = self.windows[idx].copy() # (6, 100)
        
        if self.is_train:
            # 1. 3D SO(3) random rotation perturbation
            angles = np.random.uniform(-0.25, 0.25, size=3).astype(np.float32)
            rot = R.from_euler("xyz", angles).as_matrix().astype(np.float32)
            w[:3, :] = rot @ w[:3, :]
            w[3:6, :] = rot @ w[3:6, :]
            
            # 2. Add realistic IMU jitter
            w[:3, :] += np.random.normal(0, 0.02, size=w[:3, :].shape).astype(np.float32)
            w[3:6, :] += np.random.normal(0, 0.005, size=w[3:6, :].shape).astype(np.float32)
            
        norm_a = np.linalg.norm(w[:3, :], axis=0, keepdims=True)
        norm_w = np.linalg.norm(w[3:6, :], axis=0, keepdims=True)
        w8 = np.concatenate([w, norm_a, norm_w], axis=0) # (8, 100)
        
        w_norm = (w8 - self.mean) / self.std
        x = torch.from_numpy(w_norm).float()
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def run():
    print("Ingesting trips for training...", flush=True)
    loader = GenericDataLoader()
    trips = [loader.load_file(download_iovnbd_trip(k)) for k in ["S-S1", "S-S2", "S-M"]]
    train_trips = [t for t in trips if t.trip_id in ["S-S1", "S-M"]]
    val_trips = [t for t in trips if t.trip_id in ["S-S2"]]
    
    t0 = time.time()
    train_ds = FastAugmentedDataset(train_trips, window_size=100, step_size=2, is_train=True)
    val_ds = FastAugmentedDataset(val_trips, window_size=100, step_size=5, is_train=False, mean=train_ds.mean, std=train_ds.std)
    print(f"Dataset ready in {time.time() - t0:.2f}s! Train: {len(train_ds):,} windows, Val: {len(val_ds):,} windows", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    
    best_val_rmse = float("inf")
    
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            sp, log_var = model(x)
            loss = gaussian_nll_loss(y, sp, log_var)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        scheduler.step()
        train_loss /= len(train_ds)
        
        # Validation on completely unseen trip S-S2
        model.eval()
        val_sq_err = 0.0
        val_abs_err = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                sp, log_var = model(x)
                diff = (y - sp).cpu().numpy()
                val_sq_err += np.sum(diff**2)
                val_abs_err += np.sum(np.abs(diff))
                
        val_rmse = float(np.sqrt(val_sq_err / len(val_ds)))
        val_mae = float(val_abs_err / len(val_ds))
        
        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                "epoch": epoch,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            }, "models/checkpoints/best_velocity_model.pt")
            
        print(f"Epoch {epoch:02d}/30 | Train NLL: {train_loss:.4f} | Val RMSE (UNSEEN S-S2): {val_rmse:.3f} m/s ({val_rmse*3.6:.1f} km/h) | Val MAE: {val_mae:.3f} m/s {'[BEST CHECKPOINT]' if is_best else ''}", flush=True)

if __name__ == "__main__":
    run()
