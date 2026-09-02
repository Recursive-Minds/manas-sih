"""
Interactive Training Script for the Physics-Informed, Rotation-Augmented TCN-Attention Velocity Model.

Features:
- Fast vectorized dataset loading (<1s).
- Online 3D SO(3) rotational data augmentation & sensor jitter for mount-angle invariance.
- Multi-scale dilated TCN (d in {1, 2, 4, 8, 16}) + Multi-head self-attention.
- Huber-regularized Gaussian NLL loss with kinematic consistency penalties.
- Strict trip-level dataset partitioning (zero leakage).
- Automatic checkpointing to models/checkpoints/best_velocity_model.pt.
- Automatic diagnostic plot generation to artifacts/training_curves.png.
- Target: NVIDIA RTX 4060 GPU / CUDA acceleration.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.models.dataset import IMUVelocityDataset
from sih.models.tcn_attention import TCNAttentionVelocityModel, gaussian_nll_loss


def train():
    parser = argparse.ArgumentParser(description="Train TCN-Attention Forward Velocity Estimator")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--window_size", type=int, default=100, help="IMU window size (samples = 10s)")
    parser.add_argument("--step_size", type=int, default=2, help="Stride between windows (samples = 0.2s)")
    parser.add_argument("--in_channels", type=int, default=8, help="Number of input channels")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Save directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PHASE 3: TRAINING ROTATION-AUGMENTED TCN-ATTENTION VELOCITY ESTIMATOR")
    print("=" * 75)
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Real Trip Datasets
    print("\n[1/4] Ingesting IO-VNBD Trip Datasets...")
    trip_keys = ["S-S1", "S-S2", "S-M"]
    trips = []
    loader = GenericDataLoader()
    for k in trip_keys:
        csv_file = download_iovnbd_trip(k)
        t = loader.load_file(csv_file)
        trips.append(t)
        print(f"  - Loaded {t.trip_id}: {len(t.imu_samples):,} IMU ticks, {len(t.gnss_samples):,} GNSS fixes, {t.total_gnss_distance_m:.1f} m")

    # 2. Strict Cross-Trip Partitioning (Train on S-S1 & S-M, Validate on unseen S-S2 - Zero Leakage)
    print("\n[2/4] Partitioning Datasets Strictly by Trip (Zero Leakage)...")
    train_trips = [t for t in trips if t.trip_id in ["S-S1", "S-M"]]
    val_trips = [t for t in trips if t.trip_id in ["S-S2"]]

    print(f"  - Train Trips ({len(train_trips)}): {[t.trip_id for t in train_trips]}")
    print(f"  - Val Trips   ({len(val_trips)}): {[t.trip_id for t in val_trips]} (Held-Out Unseen)")

    t0_ds = time.time()
    train_ds = IMUVelocityDataset(train_trips, window_size=args.window_size, step_size=args.step_size, in_channels=args.in_channels, is_train=True)
    val_ds = IMUVelocityDataset(val_trips, window_size=args.window_size, step_size=5, in_channels=args.in_channels, is_train=False,
                                mean=train_ds.mean, std=train_ds.std)
    print(f"  - Dataset Built in {time.time() - t0_ds:.2f}s!")
    print(f"  - Train Windows Generated: {len(train_ds):,}")
    print(f"  - Val Windows Generated:   {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    # 3. Model & Optimizer Setup
    print("\n[3/4] Initializing Multi-Scale TCN-Attention Model...")
    model = TCNAttentionVelocityModel(in_channels=args.in_channels, base_channels=32, num_attention_heads=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Trainable Parameters: {total_params:,} (~{total_params * 4 / 1024:.1f} KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    best_val_rmse = float("inf")
    history = {"train_nll": [], "val_rmse": [], "val_mae": []}

    # 4. Training Loop with Live Progress Bars
    print("\n[4/4] Starting Interactive Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_nll_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred_speed, log_var = model(x)
            loss = gaussian_nll_loss(y, pred_speed, log_var)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_nll_sum += loss.item() * len(y)
            train_count += len(y)
            pbar.set_postfix({"NLL Loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})

        scheduler.step()
        train_epoch_nll = train_nll_sum / max(train_count, 1)

        # Validation Loop on UNSEEN S-S2
        model.eval()
        val_sq_err_sum = 0.0
        val_abs_err_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)

                pred_speed, log_var = model(x)
                diff = (y - pred_speed).cpu().numpy()
                val_sq_err_sum += np.sum(diff ** 2)
                val_abs_err_sum += np.sum(np.abs(diff))
                val_count += len(y)

        val_rmse = float(np.sqrt(val_sq_err_sum / max(val_count, 1)))
        val_mae = float(val_abs_err_sum / max(val_count, 1))

        history["train_nll"].append(train_epoch_nll)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        # Checkpoint Best Model
        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            ckpt_path = os.path.join(args.checkpoint_dir, "best_velocity_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                "in_channels": args.in_channels,
            }, ckpt_path)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train NLL: {train_epoch_nll:.4f} | Val RMSE (UNSEEN S-S2): {val_rmse:.3f} m/s ({val_rmse*3.6:.1f} km/h) | Val MAE: {val_mae:.3f} m/s {'[BEST CHECKPOINT]' if is_best else ''}", flush=True)

    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETE! Best Validation RMSE on UNSEEN S-S2: {best_val_rmse:.3f} m/s ({best_val_rmse*3.6:.1f} km/h)")
    print(f"Saved Checkpoint: models/checkpoints/best_velocity_model.pt")
    print("=" * 75)

    # 5. Save Diagnostic Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(1, args.epochs + 1), history["train_nll"], label="Train NLL Loss", color="blue", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Huber NLL Loss Curve")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(range(1, args.epochs + 1), history["val_rmse"], label="Val RMSE (m/s) [Unseen S-S2]", color="green", lw=2)
    ax2.plot(range(1, args.epochs + 1), history["val_mae"], label="Val MAE (m/s) [Unseen S-S2]", color="orange", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Velocity Error (m/s)")
    ax2.set_title("Generalization Validation on Unseen S-S2")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    curve_plot = "artifacts/training_curves.png"
    plt.savefig(curve_plot, dpi=150)
    plt.close()
    print(f"Diagnostic Plot Saved: {curve_plot}")


if __name__ == "__main__":
    train()
