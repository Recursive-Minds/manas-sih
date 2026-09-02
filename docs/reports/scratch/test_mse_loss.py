import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.models.dataset import IMUVelocityDataset
from sih.models.tcn_attention import TCNAttentionVelocityModel

loader = GenericDataLoader()
trips = [loader.load_file(download_iovnbd_trip(k)) for k in ["S-S1", "S-S2", "S-M"]]
train_trips = [t for t in trips if t.trip_id in ["S-S1", "S-M"]]
val_trips = [t for t in trips if t.trip_id in ["S-S2"]]

train_ds = IMUVelocityDataset(train_trips, window_size=100, step_size=2, is_train=True)
val_ds = IMUVelocityDataset(val_trips, window_size=100, step_size=5, is_train=False, mean=train_ds.mean, std=train_ds.std)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

print("Training with Balanced MSE + Scale Consistency Loss on GPU...")
for epoch in range(1, 26):
    model.train()
    train_mse = 0.0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda().unsqueeze(1)
        optimizer.zero_grad()
        sp, log_var = model(x)
        # Direct MSE Loss without artificial shrinkage
        mse = F.mse_loss(sp, y)
        loss = mse
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_mse += mse.item() * len(y)
    scheduler.step()
    train_mse /= len(train_ds)
    
    model.eval()
    val_sq_err = 0.0
    val_y_sum = 0.0
    val_sp_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda().unsqueeze(1)
            sp, log_var = model(x)
            diff = (y - sp).cpu().numpy()
            val_sq_err += np.sum(diff**2)
            val_y_sum += np.sum(y.cpu().numpy())
            val_sp_sum += np.sum(sp.cpu().numpy())
            val_count += len(y)
            
    val_rmse = float(np.sqrt(val_sq_err / val_count))
    scale_ratio = float(val_sp_sum / max(val_y_sum, 1e-5))
    print(f"Epoch {epoch:02d}/25 | Train MSE: {train_mse:.4f} | Val RMSE (Unseen S-S2): {val_rmse:.3f} m/s ({val_rmse*3.6:.1f} km/h) | Predicted Scale Ratio: {scale_ratio:.3f}")
