import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.models.tcn_attention import TCNAttentionVelocityModel, gaussian_nll_loss

def level_trip_imu(trip):
    accels = np.array([s.accel for s in trip.imu_samples], dtype=np.float32)
    gyros = np.array([s.gyro for s in trip.imu_samples], dtype=np.float32)
    
    # Gravity leveling
    g_body = np.mean(accels[:min(500, len(accels))], axis=0)
    g_norm = np.linalg.norm(g_body)
    g_unit = g_body / max(g_norm, 1e-4)
    
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross = np.cross(g_unit, up)
    dot = float(np.dot(g_unit, up))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-6:
        R_level = R.identity() if dot > 0 else R.from_rotvec(np.array([np.pi, 0, 0]))
    else:
        axis = cross / cross_norm
        angle = float(np.arctan2(cross_norm, dot))
        R_level = R.from_rotvec(axis * angle)
        
    accels_lev = np.array([R_level.apply(a) for a in accels], dtype=np.float32)
    gyros_lev = np.array([R_level.apply(g) for g in gyros], dtype=np.float32)
    
    norm_a = np.linalg.norm(accels, axis=1, keepdims=True)
    norm_g = np.linalg.norm(gyros, axis=1, keepdims=True)
    
    # 8 channels: [ax_lev, ay_lev, az_lev, gx_lev, gy_lev, gz_lev, norm_a, norm_g]
    feats = np.hstack([accels_lev, gyros_lev, norm_a, norm_g]).T # (8, N)
    return feats

def test_quick_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    loader = GenericDataLoader()
    trips = [loader.load_file(download_iovnbd_trip(k)) for k in ["S-S1", "S-M", "S-S2"]]
    train_trips = [t for t in trips if t.trip_id in ["S-S1", "S-M"]]
    val_trips = [t for t in trips if t.trip_id in ["S-S2"]]
    
    # Build windows
    def extract_windows(trips_list, window_size=100, step_size=5):
        windows, targets = [], []
        for trip in trips_list:
            feats = level_trip_imu(trip)
            imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples])
            gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
            gnss_speeds = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in trip.gnss_samples])
            interp_spd = np.interp(imu_ts, gnss_ts, gnss_speeds).astype(np.float32)
            
            n = feats.shape[1]
            for start in range(0, n - window_size + 1, step_size):
                end = start + window_size
                w = feats[:, start:end]
                # Target speed
                target_spd = float(interp_spd[end - 1])
                windows.append(w)
                targets.append(target_spd)
        return windows, targets

    train_w, train_y = extract_windows(train_trips, 100, 5)
    val_w, val_y = extract_windows(val_trips, 100, 5)
    print(f"Train windows: {len(train_w)}, Val windows: {len(val_w)}")
    
    # Normalization
    all_data = np.concatenate(train_w, axis=1) # (8, total)
    mean = np.mean(all_data, axis=1, keepdims=True).astype(np.float32)
    std = np.std(all_data, axis=1, keepdims=True).astype(np.float32)
    std[std < 1e-5] = 1.0
    
    class DS(torch.utils.data.Dataset):
        def __init__(self, w, y, mean, std):
            self.w = w
            self.y = y
            self.mean = mean
            self.std = std
        def __len__(self): return len(self.w)
        def __getitem__(self, idx):
            x = (self.w[idx] - self.mean) / self.std
            return torch.from_numpy(x).float(), torch.tensor(self.y[idx], dtype=torch.float32)

    train_ds = DS(train_w, train_y, mean, std)
    val_ds = DS(val_w, val_y, mean, std)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
    
    best_rmse = float("inf")
    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            p, log_var = model(x)
            loss = gaussian_nll_loss(y, p, log_var)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y)
        sched.step()
        
        model.eval()
        val_sq_err, val_cnt = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                p, _ = model(x)
                diff = (y - p).cpu().numpy()
                val_sq_err += np.sum(diff**2)
                val_cnt += len(y)
        rmse = np.sqrt(val_sq_err / val_cnt)
        if rmse < best_rmse:
            best_rmse = rmse
            star = " ★ Best"
        else:
            star = ""
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss/len(train_ds):.4f} | Val RMSE on unseen S-S2: {rmse:.3f} m/s ({rmse*3.6:.2f} km/h){star}")
    print(f"\nFinal Best Val RMSE on unseen S-S2: {best_rmse:.3f} m/s ({best_rmse*3.6:.2f} km/h)")

if __name__ == "__main__":
    test_quick_train()
