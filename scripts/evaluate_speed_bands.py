"""
Comprehensive Speed Accuracy Evaluation across Velocity Bands.
Compares Old Non-Causal Model vs. New Causal Retrained Model against 10 Hz CAN ground truth
(and GPS Doppler on S-S4).
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

ROOT_DIR = os.path.abspath(".")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.data.can_sync import load_synchronized_can_speed, is_can_supervised_allowed
from sih.data.split import compute_trip_partition
from sih.calibration.mount import calibrate_stream
from sih.models.inference import load_ai_model
from sih.fusion.speed_smoother import CausalSpeedSmoother
from sih.features.streaming import StreamingFeatureExtractor


def run_model_inference(
    model: Any,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    feats: np.ndarray,
    device: torch.device,
    short_len: int = 20,
    long_len: int = 60,
) -> np.ndarray:
    norm_feats = (feats.T - norm_mean) / (norm_std + 1e-6)
    N = len(feats)
    pad_l = np.repeat(norm_feats[:, 0:1], long_len - 1, axis=1)
    padded_feats = np.hstack([pad_l, norm_feats]).astype(np.float32)

    from numpy.lib.stride_tricks import sliding_window_view
    windows_l = sliding_window_view(padded_feats, window_shape=long_len, axis=1)
    windows_l = np.ascontiguousarray(windows_l.transpose(1, 0, 2)).astype(np.float32)
    windows_s = np.ascontiguousarray(windows_l[:, :, -short_len:]).astype(np.float32)

    preds = []
    batch_size = 4096
    with torch.no_grad():
        for b in range(0, N, batch_size):
            b_s = torch.from_numpy(windows_s[b : b + batch_size]).to(device)
            b_l = torch.from_numpy(windows_l[b : b + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                vf, _, _ = model(b_s, b_l)
            preds.extend(vf.squeeze(-1).float().cpu().numpy().flatten())
    raw_preds = np.array(preds, dtype=np.float32)
    smoother = CausalSpeedSmoother(a_max_mps2=3.5, a_min_mps2=-5.0, tau_s=0.25)
    return smoother.filter_sequence(raw_preds, dt_s=0.1)


def compute_band_metrics(v_pred: np.ndarray, v_gt: np.ndarray) -> Dict[str, Dict[str, float]]:
    bands = {
        "Overall": np.ones(len(v_gt), dtype=bool),
        "< 20 km/h": v_gt < (20.0 / 3.6),
        "20-50 km/h": (v_gt >= (20.0 / 3.6)) & (v_gt <= (50.0 / 3.6)),
        "> 50 km/h": v_gt > (50.0 / 3.6),
    }

    results = {}
    for bname, mask in bands.items():
        if np.sum(mask) == 0:
            results[bname] = {"count": 0, "rmse": 0.0, "bias": 0.0, "scale": 1.0}
            continue
        vp = v_pred[mask]
        vg = v_gt[mask]
        diff = vp - vg
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        scale = float(np.sum(vp) / max(np.sum(vg), 1e-4))
        results[bname] = {
            "count": int(np.sum(mask)),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "scale": round(scale, 3),
        }
    return results


def evaluate_both_models(
    old_model_path: str = "models/checkpoints/best_moe_velocity_model_NONCAUSAL.pt",
    new_model_path: str = "models/checkpoints/causal_moe_v1.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Models onto Device: {device}...")
    old_model, old_mean, old_std, _ = load_ai_model(device, model_path=old_model_path)
    new_model, new_mean, new_std, _ = load_ai_model(device, model_path=new_model_path)

    loader = GenericDataLoader()
    data_dir = "data/raw/iovnbd_trips"
    trips_eval = [
        ("S-M", "Highway (Part 3)", True),
        ("S-S2", "Arterial (Part 3)", True),
        ("S-S1", "Urban (Part 3)", True),
        ("S-S3a", "Mixed (Unseen Drive)", True),
        ("S-S4", "Arterial (Unseen Drive, GPS)", False),
    ]

    all_old_vpred = []
    all_new_vpred = []
    all_vgt = []

    per_trip_results = {}

    for tid, desc, has_can in trips_eval:
        print(f"\nProcessing {tid} [{desc}]...")
        trip = loader.load_file(os.path.join(data_dir, f"{tid}.csv"))

        # 1. Ground truth
        if has_can:
            c_spd = load_synchronized_can_speed(tid, data_dir, strict=True)
            v_gt = c_spd
        else:
            # S-S4 Doppler GPS ground truth interpolated to IMU timestamps
            imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.float64)
            g_valid = [g for g in trip.gnss_samples if g.is_valid and g.speed_mps is not None]
            g_ts = np.array([g.timestamp_ns for g in g_valid], dtype=np.float64)
            g_spd = np.array([g.speed_mps for g in g_valid], dtype=np.float64)
            v_gt = np.interp(imu_ts, g_ts, g_spd).astype(np.float32)

        # 2. Slice test partition
        min_len = min(len(trip.imu_samples), len(v_gt))
        if tid in ("S-S3a", "S-S4"):
            p_start, p_end = 0, min_len
        else:
            part = compute_trip_partition(tid, min_len)
            p_start, p_end = part.get_range("bench")  # Part 3 (80-100%)

        # 3. Old non-causal features
        old_cache = os.path.join("data/cache", f"{tid}_features_12ch.npz")
        if os.path.exists(old_cache):
            old_feats = np.load(old_cache)["feats"][:min_len]
        else:
            from sih.features.streaming import load_or_compute_causal_features
            old_feats, _, _ = load_or_compute_causal_features(trip)

        # 4. New causal features
        from sih.features.streaming import load_or_compute_causal_features
        new_feats, _, _ = load_or_compute_causal_features(trip)
        new_feats = new_feats[:min_len]

        v_pred_old = run_model_inference(old_model, old_mean, old_std, old_feats, device)[p_start:p_end]
        v_pred_new = run_model_inference(new_model, new_mean, new_std, new_feats, device)[p_start:p_end]
        v_gt_part = v_gt[p_start:p_end]

        old_metrics = compute_band_metrics(v_pred_old, v_gt_part)
        new_metrics = compute_band_metrics(v_pred_new, v_gt_part)

        per_trip_results[tid] = {
            "desc": desc,
            "old": old_metrics,
            "new": new_metrics,
        }

        all_old_vpred.append(v_pred_old)
        all_new_vpred.append(v_pred_new)
        all_vgt.append(v_gt_part)

    # Aggregate cross-trip metrics
    concat_old = np.concatenate(all_old_vpred)
    concat_new = np.concatenate(all_new_vpred)
    concat_gt = np.concatenate(all_vgt)

    agg_old = compute_band_metrics(concat_old, concat_gt)
    agg_new = compute_band_metrics(concat_new, concat_gt)

    print("\n" + "=" * 105)
    print("      SPEED ACCURACY BENCHMARK VS GROUND TRUTH: OLD NON-CAUSAL vs NEW CAUSAL")
    print("=" * 105)
    print(f"{'Band / Evaluation Split':<28} | {'Old RMSE':<9} {'Old Bias':<9} {'Old Scale':<10} | {'New RMSE':<9} {'New Bias':<9} {'New Scale':<10}")
    print("-" * 105)

    for bname in ["Overall", "< 20 km/h", "20-50 km/h", "> 50 km/h"]:
        o = agg_old[bname]
        n = agg_new[bname]
        print(f"Aggregated {bname:<17} | {o['rmse']:<9.3f} {o['bias']:<+9.3f} {o['scale']:<10.3f} | {n['rmse']:<9.3f} {n['bias']:<+9.3f} {n['scale']:<10.3f}")

    print("-" * 105)
    for tid, res in per_trip_results.items():
        o = res["old"]["Overall"]
        n = res["new"]["Overall"]
        print(f"{tid:<6} ({res['desc']:<19}) | {o['rmse']:<9.3f} {o['bias']:<+9.3f} {o['scale']:<10.3f} | {n['rmse']:<9.3f} {n['bias']:<+9.3f} {n['scale']:<10.3f}")
    print("=" * 105 + "\n")

    return per_trip_results, agg_old, agg_new


if __name__ == "__main__":
    evaluate_both_models()
