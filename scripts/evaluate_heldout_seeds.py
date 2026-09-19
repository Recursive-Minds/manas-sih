"""
Fresh Held-Out Seed Evaluation (SIH PS 26168)
--------------------------------------------
Evaluates 3 freshly drawn random seeds (from os.urandom): [319976, 480577, 473995].
Evaluated ONCE using the canonical single-model configuration without tuning.
Saved to artifacts/heldout_seed_results.json and permanently locked against any future tuning.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.run_final_benchmark import load_precomputed_benchmark_data, evaluate_seed_scenarios

HELDOUT_SEEDS = [319976, 480577, 473995]

def run_heldout_evaluation():
    print("=" * 80)
    print("    FRESH HELDOUT SEED EVALUATION (SIH PS 26168)")
    print(f"    Evaluating {len(HELDOUT_SEEDS)} Seeds Drawn via os.urandom: {HELDOUT_SEEDS}")
    print("    Strictly Single-Pass Evaluation - Zero Hyperparameter Tuning")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")

    # Load canonical model and precompute benchmark data
    pre = load_precomputed_benchmark_data(device, model_path=None, map_source="osm")

    results = []
    for idx, s in enumerate(HELDOUT_SEEDS, 1):
        t0 = time.time()
        df, detailed, met = evaluate_seed_scenarios(s, pre, map_source="osm")
        dt = time.time() - t0
        results.append(met)
        print(f"  [{idx}/{len(HELDOUT_SEEDS)}] Seed {s} -> Map Drift: {met['med_drift']:5.2f}% | Pure: {met['pure_med_drift']:5.2f}% | Tier-1: {met['t1_count']:2d}/{met['tot_sc']} | Sub-30%: {met['t1_count'] + met['t2_count']:2d}/{met['tot_sc']} ({dt:.1f}s)")

    map_drifts = [m["med_drift"] for m in results]
    pure_drifts = [m["pure_med_drift"] for m in results]
    p90_drifts = [m["p90_drift"] for m in results]
    t1_counts = [m["t1_count"] for m in results]
    beats_cnts = [m.get("beats_pure_count", 0) for m in results]
    beats_rates = [m.get("beats_pure_rate", 0.0) for m in results]

    heldout_summary = {
        "evaluation_name": "Fresh Held-Out Seed Evaluation (Zero Tuning)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "Single Dual-Expert MoE (best_moe_velocity_model.pt)",
        "seeds": HELDOUT_SEEDS,
        "seed_count": len(HELDOUT_SEEDS),
        "osm_median_drift_mean": round(float(np.mean(map_drifts)), 2),
        "osm_median_drift_std": round(float(np.std(map_drifts)), 2),
        "osm_p90_drift_mean": round(float(np.mean(p90_drifts)), 2),
        "osm_p90_drift_std": round(float(np.std(p90_drifts)), 2),
        "pure_dr_median_drift_mean": round(float(np.mean(pure_drifts)), 2),
        "pure_dr_median_drift_std": round(float(np.std(pure_drifts)), 2),
        "tier1_passes_mean": round(float(np.mean(t1_counts)), 1),
        "tier1_passes_std": round(float(np.std(t1_counts)), 1),
        "beats_pure_count_mean": round(float(np.mean(beats_cnts)), 1),
        "beats_pure_rate_mean": round(float(np.mean(beats_rates)) * 100.0, 1),
        "per_seed_evaluations": [
            {
                "seed": m["seed"],
                "osm_median_drift_pct": round(m["med_drift"], 2),
                "osm_p90_drift_pct": round(m["p90_drift"], 2),
                "pure_dr_median_drift_pct": round(m["pure_med_drift"], 2),
                "tier1_count": m["t1_count"],
                "sub30_count": m["t1_count"] + m["t2_count"],
                "beats_pure_count": m.get("beats_pure_count", 0),
                "beats_pure_rate_pct": round(m.get("beats_pure_rate", 0.0) * 100.0, 1),
                "highway_drift_pct": round(m["hwy_dom_drift"], 2),
                "arterial_drift_pct": round(m["art_dom_drift"], 2),
                "urban_drift_pct": round(m["urb_dom_drift"], 2),
                "mixed_drift_pct": round(m["mix_dom_drift"], 2),
            }
            for m in results
        ],
    }

    out_path = os.path.join(ROOT_DIR, "artifacts", "heldout_seed_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(heldout_summary, f, indent=2)
    print(f"\n[Saved Heldout Results] {out_path}")

    print("\n" + "=" * 80)
    print("HELDOUT SEED EVALUATION SUMMARY:")
    print(f"  Mean Median Drift:    {heldout_summary['osm_median_drift_mean']}% ± {heldout_summary['osm_median_drift_std']}%")
    print(f"  Mean P90 Drift:       {heldout_summary['osm_p90_drift_mean']}% ± {heldout_summary['osm_p90_drift_std']}%")
    print(f"  Mean Pure DR Drift:   {heldout_summary['pure_dr_median_drift_mean']}% ± {heldout_summary['pure_dr_median_drift_std']}%")
    print(f"  Mean Tier 1 Count:    {heldout_summary['tier1_passes_mean']} / 40")
    print(f"  Mean Beats Pure Rate: {heldout_summary['beats_pure_rate_mean']}% ({heldout_summary['beats_pure_count_mean']} / 40)")
    print("=" * 80)

if __name__ == "__main__":
    run_heldout_evaluation()
