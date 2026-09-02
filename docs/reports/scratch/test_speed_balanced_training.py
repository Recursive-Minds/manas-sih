import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.models.tcn_attention import TCNAttentionVelocityModel

# 1. Load S-S1 (Train) and S-S2 (Unseen Val)
train_trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
val_trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

def build_stratified_dataset(trip, is_train=True):
    # Extract IMU
    imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.int64)
    acc = np.array([s.accel for s in trip.imu_samples], dtype=np.float32)
    gyr = np.array([s.gyro for s in trip.imu_samples], dtype=np.float32)
    
    # GNSS interpolation for ground truth forward speed
    gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
    gnss_v = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in trip.gnss_samples], dtype=np.float32)
    imu_v = np.interp(imu_ts, gnss_ts, gnss_v).astype(np.float32)
    
    # Norm features
    norm_a = np.linalg.norm(acc, axis=1, keepdims=True)
    norm_w = np.linalg.norm(gyr, axis=1, keepdims=True)
    feats = np.hstack([acc, gyr, norm_a, norm_w]) # (N, 8)
    
    # Build windows
    window_size = 100
    X_list = []
    y_list = []
    
    N = len(feats)
    for i in range(0, N - window_size, 1 if is_train else 5):
        target_v = imu_v[i + window_size - 1]
        
        # Speed-stratified sampling: if high speed (>12 m/s), keep every sample; if low speed (<5 m/s), sub-sample
        if is_train:
            if target_v < 3.0 and (i % 4 != 0):
                continue
            elif target_v < 10.0 and (i % 2 != 0):
                continue
                
        w = feats[i : i + window_size].T # (8, 100)
        X_list.append(w)
        y_list.append(target_v)
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr

print("Building Speed-Stratified Balanced Dataset...")
X_train, y_train = build_stratified_dataset(train_trip, is_train=True)
X_val, y_val = build_stratified_dataset(val_trip, is_train=False)

print(f"Train samples: {len(X_train)} | Mean speed: {np.mean(y_train):.2f} m/s | Max speed: {np.max(y_train):.2f} m/s")
print(f"Val samples:   {len(X_val)} | Mean speed: {np.mean(y_val):.2f} m/s | Max speed: {np.max(y_val):.2f} m/s")

# Compute normalization statistics
mean_v = np.mean(X_train, axis=(0, 2), keepdims=True)
std_v = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-6

X_train_norm = (X_train - mean_v) / std_v
X_val_norm = (X_val - mean_v) / std_v

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ArrayDataset(X_train_norm, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(ArrayDataset(X_val_norm, y_val), batch_size=256, shuffle=False)

model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

print("\nStarting Stratified Weighted Training on GPU...")
for epoch in range(1, 21):
    model.train()
    train_loss_sum = 0.0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda().unsqueeze(1)
        optimizer.zero_grad()
        pred, log_var = model(x)
        
        # High-speed importance weight: w = 1.0 + y / 10.0
        weights = 1.0 + (y / 10.0)
        sq_err = (y - pred) ** 2
        loss = torch.mean(weights * sq_err)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_loss_sum += loss.item() * len(y)
        
    scheduler.step()
    
    # Eval on unseen S-S2
    model.eval()
    val_preds = []
    val_gts = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.cuda()
            pred, _ = model(x)
            val_preds.extend(pred.cpu().numpy().flatten())
            val_gts.extend(y.numpy().flatten())
            
    val_rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_gts)) ** 2))
    scale_ratio = np.sum(val_preds) / max(np.sum(val_gts), 1e-5)
    print(f"Epoch {epoch:02d}/20 | Train Loss: {train_loss_sum/len(X_train):.4f} | Val RMSE (Unseen S-S2): {val_rmse:.3f} m/s ({val_rmse*3.6:.1f} km/h) | Scale Ratio: {scale_ratio:.4f}")

# Save the checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "norm_mean": mean_v[0],
    "norm_std": std_v[0],
    "val_rmse": val_rmse,
}, "models/checkpoints/best_velocity_model.pt")
print("\nSaved balanced velocity checkpoint to models/checkpoints/best_velocity_model.pt")
