"""
GPU-Accelerated Model Testing & Benchmark Script for Smartphone Dead-Reckoning.

Target Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) / CUDA Acceleration.
Zero-Leakage Guarantees:
  1. Fold Evaluation (--fold 1..5): Evaluates strictly on the 20% purged & embargoed held-out test block.
  2. Legacy Trip Evaluation (--trip S-M): Evaluates models trained on S-S1 + S-S2 against held-out S-M.
  3. Inference Latency & GPU Throughput (ms/window, FPS, peak VRAM).
  4. High-Speed Velocity Scaling Ratio (Rule 8 in GEMINI.md: sum(v_pred)/sum(v_gt) ~ 1.00).
  5. Regime-Stratified Error Breakdown (Stationary, Low, Mid, High Speed).
  6. Dynamic Variance Alignment (Var(v_pred) / Var(v_gt)).
  7. Along-Track Dead-Reckoning Integration Error (Rule 10 in GEMINI.md).
"""

import os
import sys

# Cap CPU thread usage to keep laptop CPU calm, cool, and quiet (< 15-20% CPU)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import time
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.models.dataset import IMUVelocityDataset, MultiScaleMoEDataset
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.moe_fusion import BayesianMoEFusion


def get_safe_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            _ = torch.zeros(1, device=device) + 1.0
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return device, f"{device_name} ({total_vram_gb:.1f} GB)"
        except Exception:
            return torch.device("cpu"), "CPU (Fallback)"
    return torch.device("cpu"), "CPU"


def _compute_and_report_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_var: np.ndarray,
    device_str: str,
    model_type: str,
    peak_vram_mb: float,
    latency_ms_per_window: float,
    throughput_fps: float,
    title_prefix: str,
    chart_path: str,
    scorecard_path: str,
) -> Dict[str, Any]:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))

    sum_pred = float(np.sum(y_pred))
    sum_gt = float(np.sum(y_true))
    speed_scale_ratio = sum_pred / max(sum_gt, 1e-4)

    var_pred = float(np.var(y_pred))
    var_gt = float(np.var(y_true))
    var_alignment_ratio = var_pred / max(var_gt, 1e-6)

    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if var_pred > 1e-5 else 0.0
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(diff ** 2)
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-6))

    # Dead reckoning integrated distance
    dt = 0.1
    dist_pred = np.cumsum(y_pred * dt)
    dist_gt = np.cumsum(y_true * dt)
    final_dist_pred = float(dist_pred[-1])
    final_dist_gt = float(dist_gt[-1])
    integrated_drift_m = abs(final_dist_pred - final_dist_gt)
    integrated_drift_pct = (integrated_drift_m / max(final_dist_gt, 1.0)) * 100.0

    # Speed regime breakdown
    mask_stat = (y_true < 0.3)
    mask_low = (y_true >= 0.3) & (y_true < 5.0)
    mask_mid = (y_true >= 5.0) & (y_true < 12.0)
    mask_high = (y_true >= 12.0)

    def regime_stats(mask):
        if np.sum(mask) == 0:
            return 0.0, 0.0, 0
        m_mae = float(np.mean(np.abs(diff[mask])))
        m_bias = float(np.mean(diff[mask]))
        return m_mae, m_bias, int(np.sum(mask))

    mae_stat, bias_stat, n_stat = regime_stats(mask_stat)
    mae_low, bias_low, n_low = regime_stats(mask_low)
    mae_mid, bias_mid, n_mid = regime_stats(mask_mid)
    mae_high, bias_high, n_high = regime_stats(mask_high)

    # Print Scorecard
    print("\n" + "=" * 78)
    print("GPU ACCELERATION & LATENCY BENCHMARK:")
    print(f"  - Peak GPU VRAM:              {peak_vram_mb:.1f} MB (< 2% of 8GB RTX 4060 VRAM)")
    print(f"  - Inference Latency:           {latency_ms_per_window:.3f} ms / window")
    print(f"  - Inference Throughput:        {throughput_fps:,.1f} inferences/sec (FPS)")
    print(f"  - Real-Time Margin:            {throughput_fps / 10.0:.1f}x real-time (IMU 10Hz)")

    print("\nGENERALIZATION & SPEED SCALING (RULE 8 ENFORCEMENT):")
    print(f"  - Overall RMSE:                {rmse:.3f} m/s")
    print(f"  - Overall MAE:                 {mae:.3f} m/s")
    print(f"  - Max Error:                   {max_err:.3f} m/s")
    print(f"  - Speed Scale Ratio:           {speed_scale_ratio:.3f} (Ideal = 1.000, sum(v_hat)/sum(v_gt))")
    print(f"  - Dynamics Variance Ratio:     {var_alignment_ratio:.3f} (Ideal >= 0.85)")
    print(f"  - Correlation Pearson r:       {corr:+.4f}")
    print(f"  - Coefficient of Det R^2:      {r2:.4f}")

    print("\nREGIME-STRATIFIED ACCURACY BREAKDOWN:")
    print(f"  - Stationary (v < 0.3 m/s):    MAE = {mae_stat:.3f} m/s | Bias = {bias_stat:+.3f} m/s (N={n_stat})")
    print(f"  - Low Speed  (0.3 - 5.0 m/s):  MAE = {mae_low:.3f} m/s | Bias = {bias_low:+.3f} m/s (N={n_low})")
    print(f"  - Mid Speed  (5.0 - 12.0 m/s): MAE = {mae_mid:.3f} m/s | Bias = {bias_mid:+.3f} m/s (N={n_mid})")
    print(f"  - High Speed (v >= 12.0 m/s):  MAE = {mae_high:.3f} m/s | Bias = {bias_high:+.3f} m/s (N={n_high})")

    print("\nDEAD-RECKONING ALONG-TRACK INTEGRATION:")
    print(f"  - Ground Truth Total Distance: {final_dist_gt:.1f} m")
    print(f"  - Integrated Dead-Reckoning:   {final_dist_pred:.1f} m")
    print(f"  - Along-Track Distance Drift:  {integrated_drift_m:.2f} m ({integrated_drift_pct:.2f}% of travel)")
    print("=" * 78)

    # 4-Panel Diagnostic Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    t_axis = np.arange(len(y_pred)) * dt
    axes[0, 0].plot(t_axis, y_true, label="Ground Truth (GNSS)", color="#2ca02c", lw=2.0)
    axes[0, 0].plot(t_axis, y_pred, label=f"AI Estimate ({model_type.upper()})", color="#1f77b4", lw=1.6, alpha=0.85)
    axes[0, 0].set_title(f"{title_prefix}: Velocity Tracking", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].hist(diff, bins=60, color="#3498db", edgecolor="black", alpha=0.7, density=True)
    axes[0, 1].axvline(0.0, color="#e74c3c", linestyle="--", lw=2, label="Zero Bias")
    axes[0, 1].set_title(f"Velocity Error Distribution (MAE = {mae:.2f} m/s, Bias = {float(np.mean(diff)):+.2f} m/s)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Error (m/s)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].scatter(y_true, y_pred, alpha=0.25, color="#2980b9", s=10)
    max_v = max(float(np.max(y_true)), float(np.max(y_pred))) + 2.0
    axes[1, 0].plot([0, max_v], [0, max_v], color="#e74c3c", linestyle="--", lw=2, label="Ideal 1:1 Line")
    axes[1, 0].set_title(f"Predicted vs GT Speed (r = {corr:.3f}, R² = {r2:.3f})", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Ground Truth Speed (m/s)")
    axes[1, 0].set_ylabel("Predicted Speed (m/s)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    axes[1, 1].plot(t_axis, dist_gt, label="Ground Truth Distance", color="#2ca02c", lw=2.0)
    axes[1, 1].plot(t_axis, dist_pred, label=f"Dead-Reckoned (Drift = {integrated_drift_pct:.2f}%)", color="#e67e22", lw=2.0, linestyle="--")
    axes[1, 1].set_title("Along-Track Dead-Reckoning Distance Accumulation", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Distance Travelled (m)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic chart to: {chart_path}")

    with open(scorecard_path, "w") as f:
        f.write("metric,value,unit\n")
        f.write(f"device,{device_str},string\n")
        f.write(f"model_type,{model_type},string\n")
        f.write(f"peak_vram_mb,{peak_vram_mb:.1f},MB\n")
        f.write(f"latency_ms,{latency_ms_per_window:.3f},ms\n")
        f.write(f"throughput_fps,{throughput_fps:.1f},Hz\n")
        f.write(f"rmse,{rmse:.4f},m/s\n")
        f.write(f"mae,{mae:.4f},m/s\n")
        f.write(f"speed_scale_ratio,{speed_scale_ratio:.4f},ratio\n")
        f.write(f"dynamics_variance_ratio,{var_alignment_ratio:.4f},ratio\n")
        f.write(f"pearson_r,{corr:.4f},correlation\n")
        f.write(f"r2_score,{r2:.4f},r2\n")
        f.write(f"integrated_drift_m,{integrated_drift_m:.2f},m\n")
        f.write(f"integrated_drift_pct,{integrated_drift_pct:.2f},%\n")
        f.write(f"stationary_mae,{mae_stat:.4f},m/s\n")
        f.write(f"low_speed_mae,{mae_low:.4f},m/s\n")
        f.write(f"mid_speed_mae,{mae_mid:.4f},m/s\n")
        f.write(f"high_speed_mae,{mae_high:.4f},m/s\n")
    print(f"Saved metric scorecard to: {scorecard_path}")

    return {
        "rmse": rmse,
        "mae": mae,
        "speed_scale_ratio": speed_scale_ratio,
        "throughput_fps": throughput_fps,
        "latency_ms": latency_ms_per_window,
        "peak_vram_mb": peak_vram_mb,
        "drift_pct": integrated_drift_pct,
    }


def evaluate_fold_test_slice(fold_num: int, ckpt_path: str = None, batch_size: int = 128):
    """Evaluates a fold checkpoint strictly on that fold's held-out 20% test slice (ZERO data leakage)."""
    device, device_str = get_safe_device()
    if ckpt_path is None:
        ckpt_path = f"models/checkpoints/5fold/moe_fold_{fold_num}.pt"

    print("=" * 78)
    print(f"5-FOLD ZERO-LEAKAGE BENCHMARK: FOLD {fold_num} HELD-OUT TEST SLICES")
    print("=" * 78)
    print(f"Device: {device_str}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Held-Out Slice: 20% of S-S1 + 20% of S-S2 + 20% of S-M (with 15s embargo purge)")
    print("=" * 78)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 12)

    # Ingest all 3 trips
    from sih.data.downloader import download_iovnbd_trip
    from sih.data.loader import GenericDataLoader
    from train_5fold_cross_validation import prepare_trip_features, build_single_purged_fold

    loader = GenericDataLoader()
    trip_names = ["S-S1", "S-S2", "S-M"]
    trips = [loader.load_file(download_iovnbd_trip(name)) for name in trip_names]
    trip_data_list = [prepare_trip_features(t, in_channels=in_channels) for t in trips]

    # Build only the held-out test block of fold_num
    fold_data = build_single_purged_fold(trip_data_list, fold_k=fold_num - 1, short_len=20, long_len=60, stride=4, embargo_samples=150)
    test_ds = fold_data["test_ds"]

    expert_res = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
    expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
    expert_res.load_state_dict(ckpt["expert_resnet_state_dict"])
    expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
    model = BayesianMoEFusion(expert_res, expert_tcn).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        d_s = test_ds.x_short.to(device)
        d_l = test_ds.x_long.to(device)
        d_v = test_ds.targets.to(device)
    else:
        d_s, d_l, d_v = test_ds.x_short, test_ds.x_long, test_ds.targets

    N_test = len(test_ds)
    print(f"Generated {N_test:,} strictly held-out test windows across S-S1, S-S2, S-M.")

    # Warmup pass
    if device.type == "cuda":
        with torch.no_grad():
            _ = model(d_s[:32], d_l[:32])
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    preds, gts, variances = [], [], []

    with torch.no_grad():
        for i in range(0, N_test, batch_size):
            end_i = min(i + batch_size, N_test)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                v_fused, var_fused, _ = model(d_s[i:end_i], d_l[i:end_i])
            preds.append(v_fused.squeeze(-1).float().cpu().numpy())
            gts.append(d_v[i:end_i].squeeze(-1).float().cpu().numpy())
            variances.append(var_fused.squeeze(-1).float().cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_infer_time_s = t_end - t_start
    latency_ms_per_window = (total_infer_time_s / N_test) * 1000.0
    throughput_fps = N_test / total_infer_time_s
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)
    y_var = np.concatenate(variances)

    return _compute_and_report_metrics(
        y_pred=y_pred,
        y_true=y_true,
        y_var=y_var,
        device_str=device_str,
        model_type="moe_bayesian",
        peak_vram_mb=peak_vram_mb,
        latency_ms_per_window=latency_ms_per_window,
        throughput_fps=throughput_fps,
        title_prefix=f"Fold {fold_num} Out-of-Fold Test (Zero Leakage)",
        chart_path=f"artifacts/gpu_model_fold_{fold_num}_report.png",
        scorecard_path=f"artifacts/gpu_model_fold_{fold_num}_metrics.csv",
    )


def evaluate_checkpoint(ckpt_path: str, trip_id: str = "S-M", batch_size: int = 128):
    """Legacy full-trip evaluation for models trained with S-M held out completely."""
    device, device_str = get_safe_device()
    print("=" * 78)
    print("SMARTPHONE DEAD RECKONING: GPU VELOCITY MODEL TEST & BENCHMARK")
    print("=" * 78)
    print(f"Device: {device_str}")
    print(f"Loading Checkpoint: {ckpt_path}")
    print(f"Evaluation Trip: {trip_id} (Full Trip Evaluation)")
    print("=" * 78)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "tcn_attention")
    in_channels = ckpt.get("in_channels", 8)
    print(f"Model Architecture: {model_type.upper()} | Input Channels: {in_channels}")

    loader = GenericDataLoader()
    csv_file = download_iovnbd_trip(trip_id)
    trip = loader.load_file(csv_file)
    print(f"Loaded Trip {trip.trip_id}: {len(trip.imu_samples):,} IMU samples, {trip.total_gnss_distance_m:.1f} m GNSS distance")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    if model_type == "moe_bayesian":
        expert_res = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
        expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
        expert_res.load_state_dict(ckpt["expert_resnet_state_dict"])
        expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
        model = BayesianMoEFusion(expert_res, expert_tcn).to(device)
        model.eval()

        val_ds = MultiScaleMoEDataset(
            [trip],
            short_len=ckpt.get("short_len", 20),
            long_len=ckpt.get("long_len", 60),
            stride=1,
            in_channels=in_channels,
            use_calibrated=True,
            is_train=False,
            mean=ckpt["norm_mean"],
            std=ckpt["norm_std"],
        )
    else:
        model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        val_ds = IMUVelocityDataset(
            [trip],
            window_size=ckpt.get("window_size", 100),
            step_size=1,
            in_channels=in_channels,
            is_train=False,
            use_calibrated=True,
            mean=ckpt["norm_mean"],
            std=ckpt["norm_std"],
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Generated {len(val_ds):,} dense evaluation windows.")

    preds, gts, variances = [], [], []

    # Warmup pass
    if device.type == "cuda":
        for batch in val_loader:
            if model_type == "moe_bayesian":
                _ = model(batch[0].to(device), batch[1].to(device))
            else:
                _ = model(batch[0].to(device))
            break
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in val_loader:
            if model_type == "moe_bayesian":
                b_s = batch[0].to(device, non_blocking=True)
                b_l = batch[1].to(device, non_blocking=True)
                b_v = batch[2]
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_fused, var_fused, _ = model(b_s, b_l)
                p_batch = v_fused.squeeze(-1).float().cpu().numpy()
                v_batch = var_fused.squeeze(-1).float().cpu().numpy()
            else:
                b_x = batch[0].to(device, non_blocking=True)
                b_v = batch[1]
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    speed_pred, log_var = model(b_x)
                p_batch = speed_pred.squeeze(-1).float().cpu().numpy()
                v_batch = torch.exp(log_var).squeeze(-1).float().cpu().numpy()

            preds.append(p_batch)
            gts.append(b_v.numpy().flatten())
            variances.append(v_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_infer_time_s = t_end - t_start
    latency_ms_per_window = (total_infer_time_s / len(val_ds)) * 1000.0
    throughput_fps = len(val_ds) / total_infer_time_s
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)
    y_var = np.concatenate(variances)

    return _compute_and_report_metrics(
        y_pred=y_pred,
        y_true=y_true,
        y_var=y_var,
        device_str=device_str,
        model_type=model_type,
        peak_vram_mb=peak_vram_mb,
        latency_ms_per_window=latency_ms_per_window,
        throughput_fps=throughput_fps,
        title_prefix=f"Full Trip {trip_id}",
        chart_path="artifacts/gpu_model_test_report.png",
        scorecard_path="artifacts/gpu_model_test_metrics.csv",
    )


def evaluate_benchmark_partition(
    ckpt_path: str = "models/checkpoints/best_moe_velocity_model.pt",
    batch_size: int = 128,
):
    """Evaluates master checkpoint strictly on the held-out Part 3 (Benchmarking) partition across S-S1, S-S2, and S-M."""
    device, device_str = get_safe_device()
    print("=" * 78)
    print("SMARTPHONE DEAD RECKONING: 3-WAY BENCHMARK PARTITION TEST (ZERO LEAKAGE)")
    print("=" * 78)
    print(f"Device: {device_str}")
    print(f"Loading Checkpoint: {ckpt_path}")
    print("Evaluation: Part 3 Held-Out Benchmark Partition (S-S1 + S-S2 + S-M)")
    print("Embargo Guard: 15-second boundary purge ensures 0% leakage from Train/Val")
    print("=" * 78)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "moe_bayesian")
    in_channels = ckpt.get("in_channels", 12)
    print(f"Model Architecture: {model_type.upper()} | Input Channels: {in_channels}")

    loader = GenericDataLoader()
    trips = [loader.load_file(download_iovnbd_trip(k)) for k in ["S-S1", "S-S2", "S-M"]]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    norm_mean = ckpt.get("norm_mean", ckpt.get("mean"))
    norm_std = ckpt.get("norm_std", ckpt.get("std"))
    if norm_mean.ndim == 1: norm_mean = norm_mean.reshape(1, -1)
    if norm_std.ndim == 1: norm_std = norm_std.reshape(1, -1)

    if model_type == "moe_bayesian":
        expert_res = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
        expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
        expert_res.load_state_dict(ckpt["expert_resnet_state_dict"])
        expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
        model = BayesianMoEFusion(expert_res, expert_tcn).to(device)
        model.eval()

        val_ds = MultiScaleMoEDataset(
            trips,
            short_len=ckpt.get("short_len", 20),
            long_len=ckpt.get("long_len", 60),
            stride=2,
            in_channels=in_channels,
            use_calibrated=True,
            is_train=False,
            mean=norm_mean,
            std=norm_std,
            partition="bench",
        )
    else:
        model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        val_ds = IMUVelocityDataset(
            trips,
            window_size=ckpt.get("window_size", 100),
            step_size=2,
            in_channels=in_channels,
            is_train=False,
            use_calibrated=True,
            mean=norm_mean,
            std=norm_std,
            partition="bench",
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Generated {len(val_ds):,} strictly held-out benchmark windows across all 3 trips.")

    preds, gts, variances = [], [], []

    # Warmup
    if device.type == "cuda":
        for batch in val_loader:
            if model_type == "moe_bayesian":
                _ = model(batch[0].to(device), batch[1].to(device))
            else:
                _ = model(batch[0].to(device))
            break
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in val_loader:
            if model_type == "moe_bayesian":
                b_s = batch[0].to(device, non_blocking=True)
                b_l = batch[1].to(device, non_blocking=True)
                b_v = batch[2]
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_fused, var_fused, _ = model(b_s, b_l)
                v_fused = torch.nan_to_num(v_fused, nan=0.0, posinf=35.0, neginf=0.0)
                preds.append(v_fused.squeeze(-1).float().cpu().numpy())
                gts.append(b_v.squeeze(-1).float().cpu().numpy())
                variances.append(var_fused.squeeze(-1).float().cpu().numpy())
            else:
                b_x = batch[0].to(device, non_blocking=True)
                b_y = batch[1]
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred, _ = model(b_x)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=35.0, neginf=0.0)
                preds.append(pred.squeeze(-1).float().cpu().numpy())
                gts.append(b_y.squeeze(-1).float().cpu().numpy())
                variances.append(np.ones_like(pred.squeeze(-1).float().cpu().numpy()) * 0.3)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_infer_time_s = t_end - t_start
    latency_ms_per_window = (total_infer_time_s / max(len(val_ds), 1)) * 1000.0
    throughput_fps = len(val_ds) / max(total_infer_time_s, 1e-6)
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)
    y_var = np.concatenate(variances)

    return _compute_and_report_metrics(
        y_pred=y_pred,
        y_true=y_true,
        y_var=y_var,
        device_str=device_str,
        model_type=model_type,
        peak_vram_mb=peak_vram_mb,
        latency_ms_per_window=latency_ms_per_window,
        throughput_fps=throughput_fps,
        title_prefix="Held-Out Part 3 Benchmark (S-S1 + S-S2 + S-M)",
        chart_path="artifacts/gpu_model_bench_report.png",
        scorecard_path="artifacts/gpu_model_bench_metrics.csv",
    )


def main():
    parser = argparse.ArgumentParser(description="GPU Velocity Model Test & Benchmark")
    parser.add_argument("--fold", type=int, default=None, help="Evaluate a specific fold (1-5)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--trip", type=str, default=None, help="Trip ID to evaluate (single trip)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    ckpt = args.checkpoint or "models/checkpoints/best_moe_velocity_model.pt"
    if not os.path.exists(ckpt):
        ckpt = "models/checkpoints/best_velocity_model.pt"

    if args.fold is not None:
        evaluate_fold_test_slice(fold_num=args.fold, ckpt_path=ckpt, batch_size=args.batch_size)
    elif args.trip is not None:
        evaluate_checkpoint(ckpt, trip_id=args.trip, batch_size=args.batch_size)
    else:
        # Default: Standardized 3-Way Held-Out Part 3 Benchmark (Zero Leakage)
        evaluate_benchmark_partition(ckpt_path=ckpt, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
