"""
CAN-Supervised GPU Training Pipeline for Bayesian Dual-Expert MoE Velocity Estimator.

Strict Rules Enforced:
1. Zero Data Leakage: Trains strictly on Part 1 (60% Train) and validates on Part 2 (20% Val) of S-S1, S-S2, and S-M.
2. Unseen Sequence Isolation: S-S3a and S-S4 are NEVER loaded or touched.
3. Part 3 Benchmark Isolation: Part 3 (80-100%) is strictly isolated and NEVER exposed.
4. Physical Ground-Truth Supervision: Supervised by time-synchronized 10 Hz vehicle CAN wheel speeds.
5. Rule 8 (High-Speed Scaling) & Rule 9 (3D Mount Invariance) strictly obeyed.
"""

import os
import sys

# Cap CPU thread usage to ~50% (8 threads)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
torch.set_num_threads(8)

import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.models.can_dataset import MultiScaleCANMoEDataset
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.moe_fusion import BayesianMoEFusion
from sih.models.losses import phase55_balanced_loss


def train_can_moe(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    in_channels: int = 12,
    seed: int = 42,
    checkpoint_path: str = "models/checkpoints/causal_moe_v1.pt",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("   10 Hz CAN-SUPERVISED BAYESIAN DUAL-EXPERT MoE TRAINING PIPELINE")
    print("   Strict Zero-Leakage: Part 1 (60% Train) + Part 2 (20% Val) of S-M, S-S1, S-S2")
    print(f"   Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"   Random Seed: {seed} | Checkpoint: {checkpoint_path}")
    print("=" * 80)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # 1. Load ONLY the 3 Training Trips (S-M, S-S1, S-S2). S-S3a and S-S4 are strictly excluded.
    loader = GenericDataLoader()
    train_trip_ids = ["S-S1", "S-S2", "S-M"]
    trips = []
    for tid in train_trip_ids:
        path = os.path.join("data/raw/iovnbd_trips", f"{tid}.csv")
        t = loader.load_file(path)
        trips.append(t)
        print(f"  - Loaded Trip {tid}: {len(t.imu_samples):,} IMU samples")

    # 2. Build Datasets
    print("\nBuilding CAN-Supervised Multi-Scale Datasets...")
    train_ds = MultiScaleCANMoEDataset(
        trips,
        short_len=20,
        long_len=60,
        stride=2,
        in_channels=in_channels,
        use_calibrated=True,
        is_train=True,
        partition="train",
    )
    val_ds = MultiScaleCANMoEDataset(
        trips,
        short_len=20,
        long_len=60,
        stride=5,
        in_channels=in_channels,
        use_calibrated=True,
        is_train=False,
        mean=train_ds.mean,
        std=train_ds.std,
        partition="val",
    )

    print(f"  - Training Windows (60% Train):    {len(train_ds):,}")
    print(f"  - Validation Windows (20% Val):    {len(val_ds):,}")
    print(f"  - Isolated Benchmark (20% Test):   STRICTLY HELD OUT (0 samples exposed)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 3. Model Architecture
    expert_res = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
    expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
    moe_model = BayesianMoEFusion(expert_res, expert_tcn).to(device)

    total_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    print(f"  - Trainable Model Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(moe_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = float("inf")
    best_rmse = float("inf")
    best_scale = 1.0
    history = {"train_loss": [], "val_rmse": [], "val_mae": [], "speed_ratio": []}

    print("\nStarting Training on GPU...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        moe_model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [CAN-MoE]", leave=False, dynamic_ncols=True)
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

        # Validation on Part 2 (held-out 20% validation split)
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

                v_clean = torch.nan_to_num(v_fused, nan=0.0, posinf=35.0, neginf=0.0)
                diff = (b_v - v_clean).cpu().numpy()
                val_sq_err_sum += np.sum(diff ** 2)
                val_abs_err_sum += np.sum(np.abs(diff))
                val_count += len(b_v)
                total_pred_v += float(torch.sum(v_clean).cpu().item())
                total_gt_v += float(torch.sum(b_v).cpu().item())

        val_rmse = float(np.sqrt(val_sq_err_sum / max(val_count, 1)))
        val_mae = float(val_abs_err_sum / max(val_count, 1))
        speed_scale_ratio = total_pred_v / max(total_gt_v, 1e-4)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        history["speed_ratio"].append(speed_scale_ratio)

        # Score balancing RMSE and Rule 8 high-speed scale ratio (~1.00)
        score = val_rmse + 5.0 * abs(speed_scale_ratio - 1.0)
        is_best = score < best_score
        marker = " [BEST]" if is_best else ""

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.3f} m/s | Val MAE: {val_mae:.3f} m/s | Scale Ratio: {speed_scale_ratio:.2f}{marker}")

        if is_best:
            best_score = score
            best_rmse = val_rmse
            best_scale = speed_scale_ratio
            save_payload = {
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
                "in_channels": in_channels,
                "short_len": 20,
                "long_len": 60,
            }
            torch.save(save_payload, checkpoint_path)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"CAN-Supervised Training Complete in {elapsed:.1f}s ({elapsed/60:.1f} min)!")
    print(f"Best Checkpoint: {checkpoint_path}")
    print(f"Best Val RMSE:   {best_rmse:.3f} m/s")
    print(f"Best Scale:      {best_scale:.2f} (Target ~1.00)")
    print("=" * 80)

    # Plot convergence curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss (CAN-Supervised)", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, history["val_rmse"], label="Val RMSE (m/s)", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric")
    ax1.set_title("CAN-Supervised MoE Convergence")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, history["speed_ratio"], label="Speed Scale Ratio", color="#2ca02c", lw=2)
    ax2.axhline(1.0, color="#d62728", linestyle="--", label="Target = 1.00 (Rule 8)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Ratio")
    ax2.set_title("Speed Scale Ratio on Val Split")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plot_path = "artifacts/can_moe_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic curves to: {plot_path}")

    return checkpoint_path, best_rmse, best_scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal MoE Velocity Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--checkpoint-path", type=str, default="models/checkpoints/causal_moe_v1.pt", help="Path to save best checkpoint")
    args = parser.parse_args()

    train_can_moe(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
    )
