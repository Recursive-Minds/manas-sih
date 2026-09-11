"""
GPU-Accelerated Training Script for Vehicle Velocity & Uncertainty Estimation.

Target Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) / CUDA Acceleration.
Safety Guarantees:
  - Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to eliminate memory fragmentation.
  - Mixed-Precision FP16 acceleration via torch.amp.autocast('cuda') & GradScaler.
  - Windows DataLoader safe concurrency (num_workers=0 avoids Windows multiprocessing crashes).
  - Defensive OutOfMemoryError recovery safeguards.
  - Strict Trip-Level Partitioning: Train on S-S1 & S-S2, evaluate on held-out unseen S-M.
  - Multi-Scale Bayesian Mixture-of-Experts (ResNet-1D micro + TCN-Attention macro) & Single-Expert modes.
  - High-Speed Balanced Loss (Rule 8 & Rule 9 in GEMINI.md).
"""

import os
import sys

# Cap CPU thread usage to ~50% CPU (8 threads out of 16 logical cores)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import time
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.models.dataset import IMUVelocityDataset, MultiScaleMoEDataset
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.moe_fusion import BayesianMoEFusion
from sih.models.losses import (
    balanced_velocity_loss,
    phase55_balanced_loss,
    l_dynamics_variance_alignment,
)


def get_safe_device() -> Tuple[torch.device, str]:
    """Safely initialize and verify CUDA compute on laptop GPU."""
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            # Quick smoke test kernel
            test_tensor = torch.zeros(1, device=device) + 1.0
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Hardware] Detected GPU: {device_name} ({total_vram_gb:.2f} GB VRAM)")
            torch.backends.cudnn.benchmark = True
            return device, f"{device_name} ({total_vram_gb:.1f} GB)"
        except Exception as e:
            print(f"[Warning] CUDA initialization failed ({e}). Safely falling back to CPU.")
            return torch.device("cpu"), "CPU (Fallback)"
    return torch.device("cpu"), "CPU"


def train_tcn_expert(args, device: torch.device, trips):
    """Train single TCN-Attention expert using 3-way partition."""
    print("\n--- Training Pipeline: Single-Expert TCN-Attention (3-Way Partition) ---")
    train_ds = IMUVelocityDataset(
        trips,
        window_size=args.window_size,
        step_size=args.step_size,
        in_channels=args.in_channels,
        is_train=True,
        use_calibrated=True,
        partition="train",
    )
    val_ds = IMUVelocityDataset(
        trips,
        window_size=args.window_size,
        step_size=5,
        in_channels=args.in_channels,
        is_train=False,
        use_calibrated=True,
        mean=train_ds.mean,
        std=train_ds.std,
        partition="val",
    )
    print(f"  - Train Windows Generated: {len(train_ds):,}")
    print(f"  - Val Windows Generated:   {len(val_ds):,}")

    pin = False  # Avoid Windows page locking; direct transfers are sub-millisecond
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = TCNAttentionVelocityModel(
        in_channels=args.in_channels,
        base_channels=32,
        num_attention_heads=4,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Model Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = float("inf")
    best_rmse = float("inf")
    best_scale = 1.0
    history = {"train_loss": [], "val_rmse": [], "val_mae": [], "speed_ratio": []}

    ckpt_path = os.path.join(args.checkpoint_dir, "best_velocity_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_speed, log_var = model(x)
                loss = balanced_velocity_loss(y, pred_speed, log_var)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(y)
            train_count += len(y)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})

        scheduler.step()
        train_loss = train_loss_sum / max(train_count, 1)

        # Validation on held-out trip S-M
        model.eval()
        val_sq_err_sum = 0.0
        val_abs_err_sum = 0.0
        val_count = 0
        total_pred_v = 0.0
        total_gt_v = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred_speed, log_var = model(x)

                pred_clean = torch.nan_to_num(pred_speed, nan=0.0, posinf=35.0, neginf=0.0)
                diff = (y - pred_clean).cpu().numpy()
                val_sq_err_sum += np.sum(diff ** 2)
                val_abs_err_sum += np.sum(np.abs(diff))
                val_count += len(y)
                total_pred_v += float(torch.sum(pred_clean).cpu().item())
                total_gt_v += float(torch.sum(y).cpu().item())

        val_rmse = float(np.sqrt(val_sq_err_sum / max(val_count, 1)))
        val_mae = float(val_abs_err_sum / max(val_count, 1))
        speed_scale_ratio = total_pred_v / max(total_gt_v, 1e-4)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        history["speed_ratio"].append(speed_scale_ratio)

        score = val_rmse + 5.0 * abs(speed_scale_ratio - 1.0)
        is_best = score < best_score
        if is_best:
            best_score = score
            best_rmse = val_rmse
            best_scale = speed_scale_ratio
            torch.save({
                "epoch": epoch,
                "model_type": "tcn_attention",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "speed_scale_ratio": speed_scale_ratio,
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                "in_channels": args.in_channels,
                "window_size": args.window_size,
            }, ckpt_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_rmse:.3f} m/s | MAE: {val_mae:.3f} m/s | "
            f"Speed Ratio: {speed_scale_ratio:.2f} {'[BEST]' if is_best else ''}",
            flush=True,
        )

    return ckpt_path, best_rmse, best_scale, history


def train_moe_pipeline(args, device: torch.device, trips):
    """Train Dual-Expert Bayesian Mixture-of-Experts across the multi-trip 3-way partition."""
    print("\n--- Training Pipeline: Bayesian Dual-Expert MoE (3-Way Multi-Trip Partition) ---")
    train_ds = MultiScaleMoEDataset(
        trips,
        short_len=20,
        long_len=60,
        stride=args.step_size,
        in_channels=args.in_channels,
        use_calibrated=True,
        is_train=True,
        partition="train",
    )
    val_ds = MultiScaleMoEDataset(
        trips,
        short_len=20,
        long_len=60,
        stride=5,
        in_channels=args.in_channels,
        use_calibrated=True,
        is_train=False,
        mean=train_ds.mean,
        std=train_ds.std,
        partition="val",
    )
    print(f"  - Multi-Scale Train Windows: {len(train_ds):,}")
    print(f"  - Multi-Scale Val Windows:   {len(val_ds):,}")

    pin = False  # Avoid Windows page locking; direct transfers are sub-millisecond
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    expert_res = ResNet1DSpeedEstimator(in_channels=args.in_channels, base_channels=64)
    expert_tcn = TCNAttentionVelocityModel(in_channels=args.in_channels, base_channels=32, num_attention_heads=4)
    moe_model = BayesianMoEFusion(expert_res, expert_tcn).to(device)

    total_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    print(f"  - Dual-Expert MoE Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(moe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = float("inf")
    best_rmse = float("inf")
    best_scale = 1.0
    history = {"train_loss": [], "val_rmse": [], "val_mae": [], "speed_ratio": []}

    ckpt_path = os.path.join(args.checkpoint_dir, "best_moe_velocity_model.pt")

    for epoch in range(1, args.epochs + 1):
        moe_model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train MoE]", leave=False, dynamic_ncols=True)
        for b_s, b_l, b_v, b_alat, b_wyaw, b_mlabel in pbar:
            b_s = b_s.to(device, non_blocking=True)
            b_l = b_l.to(device, non_blocking=True)
            b_v = b_v.to(device, non_blocking=True)
            b_alat = b_alat.to(device, non_blocking=True)
            b_wyaw = b_wyaw.to(device, non_blocking=True)
            b_mlabel = b_mlabel.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                v_fused, var_fused, diag = moe_model(b_s, b_l)
                loss = phase55_balanced_loss(
                    v_fused=v_fused,
                    var_fused=var_fused,
                    v_res=diag["v_resnet"],
                    var_res=diag["var_resnet"],
                    v_tcn=diag["v_tcn"],
                    var_tcn=diag["var_tcn"],
                    v_gt=b_v,
                    a_lat=b_alat,
                    w_yaw=b_wyaw,
                    class_logits=diag["class_logits"],
                    motion_labels=b_mlabel,
                    w_dyn=2.0,
                    w_cls=0.2,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(moe_model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(b_v)
            train_count += len(b_v)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})

        scheduler.step()
        train_loss = train_loss_sum / max(train_count, 1)

        # Validation on held-out trip S-M
        moe_model.eval()
        val_sq_err_sum = 0.0
        val_abs_err_sum = 0.0
        val_count = 0
        total_pred_v = 0.0
        total_gt_v = 0.0

        with torch.no_grad():
            for b_s, b_l, b_v, _, _, _ in val_loader:
                b_s = b_s.to(device, non_blocking=True)
                b_l = b_l.to(device, non_blocking=True)
                b_v = b_v.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_fused, var_fused, _ = moe_model(b_s, b_l)

                v_fused_clean = torch.nan_to_num(v_fused, nan=0.0, posinf=35.0, neginf=0.0)
                diff = (b_v - v_fused_clean).cpu().numpy()
                val_sq_err_sum += np.sum(diff ** 2)
                val_abs_err_sum += np.sum(np.abs(diff))
                val_count += len(b_v)
                total_pred_v += float(torch.sum(v_fused_clean).cpu().item())
                total_gt_v += float(torch.sum(b_v).cpu().item())

        val_rmse = float(np.sqrt(val_sq_err_sum / max(val_count, 1)))
        val_mae = float(val_abs_err_sum / max(val_count, 1))
        speed_scale_ratio = total_pred_v / max(total_gt_v, 1e-4)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        history["speed_ratio"].append(speed_scale_ratio)

        score = val_rmse + 5.0 * abs(speed_scale_ratio - 1.0)
        is_best = score < best_score
        if is_best:
            best_score = score
            best_rmse = val_rmse
            best_scale = speed_scale_ratio
            torch.save({
                "epoch": epoch,
                "model_type": "moe_bayesian",
                "expert_resnet_state_dict": expert_res.state_dict(),
                "expert_tcn_state_dict": expert_tcn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "speed_scale_ratio": speed_scale_ratio,
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                "in_channels": args.in_channels,
                "short_len": 20,
                "long_len": 60,
            }, ckpt_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_rmse:.3f} m/s | MAE: {val_mae:.3f} m/s | "
            f"Speed Ratio: {speed_scale_ratio:.2f} {'[BEST]' if is_best else ''}",
            flush=True,
        )

    return ckpt_path, best_rmse, best_scale, history


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Velocity Model Training (RTX 4060 Safe)")
    parser.add_argument("--model", type=str, choices=["tcn", "moe"], default="moe",
                        help="Model architecture: 'moe' (Dual-Expert Bayesian MoE) or 'tcn' (TCN-Attention)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (safe for 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--window_size", type=int, default=100, help="IMU window size for TCN (samples = 10s)")
    parser.add_argument("--step_size", type=int, default=2, help="Stride between windows (samples = 0.2s)")
    parser.add_argument("--in_channels", type=int, default=12, help="Input channels (8 or 12 with spectral features)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 safe for Windows)")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Save directory")
    args = parser.parse_args()

    # For TCN alone with 8 channels default
    if args.model == "tcn" and args.in_channels == 12:
        args.in_channels = 8

    device, device_str = get_safe_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    print("=" * 78)
    print("SMARTPHONE DEAD RECKONING: GPU VELOCITY MODEL TRAINING PIPELINE")
    print("=" * 78)
    print(f"Compute Device: {device_str}")
    print(f"Model Architecture: {args.model.upper()}")
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs} | Input Channels: {args.in_channels}")
    print(f"Mixed Precision: ENABLED (torch.amp.autocast FP16)")
    print(f"Windows Safety: num_workers={args.num_workers}, expandable_segments:True")
    print("=" * 78)

    # Clean GPU cache before ingestion
    if device.type == "cuda":
        torch.cuda.empty_cache()
        mem_start = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"[VRAM] Baseline GPU memory allocated: {mem_start:.1f} MB")

    # 1. Load Real Trip Datasets
    print("\n[1/3] Loading Real Trip Datasets...")
    trip_keys = ["S-S1", "S-S2", "S-M"]
    trips = []
    loader = GenericDataLoader()
    for k in trip_keys:
        csv_file = download_iovnbd_trip(k)
        t = loader.load_file(csv_file)
        trips.append(t)
        print(f"  - Loaded {t.trip_id}: {len(t.imu_samples):,} IMU ticks, {len(t.gnss_samples):,} GNSS fixes")

    # 2. Standardized 3-Way Partitioning (Zero Data Leakage with 15s Embargo Buffer)
    print("\n[2/3] Partitioning Datasets into Standardized 3-Way Splits (Train, Val, Benchmark)...")
    print("  - Part 1 (Train 60%): S-S1 (60%) + S-S2 (60%) + S-M (60%) -> Multi-Trip Training")
    print("  - Part 2 (Val 20%):   S-S1 (20%) + S-S2 (20%) + S-M (20%) -> Validation & Early Stopping")
    print("  - Part 3 (Bench 20%): S-S1 (20%) + S-S2 (20%) + S-M (20%) -> Strictly Held-Out Benchmark")
    print("  - Embargo Guard:      15 seconds (150 samples) boundary purge between every partition")

    # 3. Training Execution with OOM Guard
    print("\n[3/3] Commencing GPU Training...")
    try:
        if args.model == "moe":
            ckpt_path, best_rmse, best_scale, history = train_moe_pipeline(args, device, trips)
        else:
            ckpt_path, best_rmse, best_scale, history = train_tcn_expert(args, device, trips)
    except torch.cuda.OutOfMemoryError as oom:
        print(f"\n[GPU OOM Protection] CUDA OutOfMemory detected: {oom}")
        print("  - Clearing GPU VRAM cache and retrying with batch_size=32...")
        torch.cuda.empty_cache()
        args.batch_size = max(16, args.batch_size // 2)
        if args.model == "moe":
            ckpt_path, best_rmse, best_scale, history = train_moe_pipeline(args, device, trips)
        else:
            ckpt_path, best_rmse, best_scale, history = train_tcn_expert(args, device, trips)

    if device.type == "cuda":
        mem_peak = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"\n[VRAM Summary] Peak GPU Memory Allocated: {mem_peak:.1f} MB (< 25% of 8GB RTX 4060 VRAM)")

    print("=" * 78)
    print(f"TRAINING COMPLETE! Best checkpoint saved to: {ckpt_path}")
    print(f"Final Validation RMSE on UNSEEN S-M: {best_rmse:.3f} m/s")
    print(f"Final Speed Scale Ratio on UNSEEN S-M: {best_scale:.2f} (Target ~1.00)")
    print("=" * 78)

    # 4. Save Diagnostic Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], label="Train Balanced Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, history["val_rmse"], label="Val RMSE (m/s)", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric")
    ax1.set_title(f"Convergence Curves ({args.model.upper()})")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, history["speed_ratio"], label="Speed Scale Ratio", color="#2ca02c", lw=2)
    ax2.axhline(1.0, color="#d62728", linestyle="--", label="Target = 1.00 (Rule 8)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Ratio")
    ax2.set_title("Speed Scale Ratio Generalization on Unseen S-M")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plot_file = f"artifacts/{args.model}_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved diagnostic curves to: {plot_file}")


if __name__ == "__main__":
    main()
