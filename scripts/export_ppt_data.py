"""
scripts/export_ppt_data.py

Processes the 6-seed benchmark results (results/final/six_seed/production_scenarios.csv)
and exports:
1. ppt_pack/data/runs_6seed.csv (per-run table)
2. ppt_pack/data/summary_6seed.json (overall stats, per-scenario medians, domain/regime splits,
   regime scorecard vs PS targets, and frozen held-out confirmation numbers)
3. ppt_pack/data/stages.json (stages drift: naive, EKF+NHC, pure-DR, full pipeline)
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def compute_speed_regime(dist_m: float, duration_s: float) -> str:
    if duration_s <= 0:
        return "city"
    kmh = (dist_m / duration_s) * 3.6
    if kmh < 20.0:
        return "crawl"
    elif kmh <= 50.0:
        return "city"
    else:
        return "highway"

def compute_stats(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}
    per_seed = df.groupby("seed")["final_drift_pct"].median()
    return {
        "n": int(len(df)),
        "median_drift_pct": round(float(df["final_drift_pct"].median()), 2),
        "mean_of_seed_medians": round(float(per_seed.mean()), 2),
        "std_of_seed_medians": round(float(per_seed.std(ddof=0)), 2),
        "p90_drift_pct": round(float(df["final_drift_pct"].quantile(0.90)), 2),
        "share_lt_10_pct": round(float((df["final_drift_pct"] < 10.0).mean() * 100.0), 2),
        "share_lt_30_pct": round(float((df["final_drift_pct"] < 30.0).mean() * 100.0), 2),
        "beats_pure_dr_pct": round(float((df["final_drift_pct"] < df["pure_dr_drift_pct"]).mean() * 100.0), 2),
        "pure_dr_median_drift_pct": round(float(df["pure_dr_drift_pct"].median()), 2),
    }

def main():
    csv_path = os.path.join(ROOT, "results", "final", "six_seed", "production_scenarios.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist yet.")
        return 1

    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} rows from {csv_path}")

    # Build runs_6seed dataframe
    rows = []
    for _, r in df_raw.iterrows():
        dur_s = float(r["duration_s"])
        dist_m = float(r["dist_m"])
        regime = compute_speed_regime(dist_m, dur_s)
        trip_str = str(r["trip"])
        is_unseen = trip_str.startswith("S-S3a") or trip_str.startswith("S-S4")

        rows.append({
            "seed": int(r["seed"]),
            "scenario_id": int(r["scenario_id"]),
            "trip": trip_str,
            "domain": str(r["domain"]),
            "speed_regime": regime,
            "unseen_trip": is_unseen,
            "blackout_s": round(dur_s, 2),
            "gt_dist_m": round(dist_m, 2),
            "pure_dr_err_m": round(float(r["pure_err_m"]), 2),
            "pure_dr_drift_pct": round(float(r["pure_drift_pct"]), 2),
            "final_err_m": round(float(r["map_err_m"]), 2),
            "final_drift_pct": round(float(r["map_drift_pct"]), 2),
        })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, "ppt_pack", "data")
    os.makedirs(out_dir, exist_ok=True)

    csv_out = os.path.join(out_dir, "runs_6seed.csv")
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out} ({len(df)} rows)")

    # Overall stats
    overall = compute_stats(df)

    # Per scenario (40 rows)
    per_scenario = []
    for sc_id, g in df.groupby("scenario_id"):
        pure_med = float(g["pure_dr_drift_pct"].median())
        final_med = float(g["final_drift_pct"].median())
        improved_cnt = int((g["final_drift_pct"] < g["pure_dr_drift_pct"]).sum())
        lt10_cnt = int((g["final_drift_pct"] < 10.0).sum())
        lt30_cnt = int((g["final_drift_pct"] < 30.0).sum())
        first = g.iloc[0]
        per_scenario.append({
            "scenario_id": int(sc_id),
            "trip": first["trip"],
            "domain": first["domain"],
            "speed_regime": first["speed_regime"],
            "blackout_s": first["blackout_s"],
            "gt_dist_m": first["gt_dist_m"],
            "pure_drift_median": round(pure_med, 2),
            "final_drift_median": round(final_med, 2),
            "seeds_evaluated": len(g),
            "improved_count": improved_cnt,
            "lt_10_count": lt10_cnt,
            "lt_30_count": lt30_cnt,
        })
    per_scenario.sort(key=lambda x: x["scenario_id"])

    # Splits
    by_trip = {t: compute_stats(g) for t, g in df.groupby("trip")}
    by_domain = {d: compute_stats(g) for d, g in df.groupby("domain")}
    by_speed_regime = {s: compute_stats(g) for s, g in df.groupby("speed_regime")}
    unseen_vs_seen = {
        "unseen": compute_stats(df[df["unseen_trip"] == True]),
        "seen": compute_stats(df[df["unseen_trip"] == False]),
    }

    # Regime scorecard vs PS targets
    # targets:
    # crawl: median error in m vs < 10 m
    # city: median drift vs < 15 %
    # highway: median drift vs < 10 %
    # all: median drift vs < 10 %
    def eval_status(val: float, target: float) -> str:
        if val <= target:
            return "Met"
        elif val <= target * 1.5:
            return "Near"
        else:
            return "Not met"

    crawl_sub = df[df["speed_regime"] == "crawl"]
    city_sub = df[df["speed_regime"] == "city"]
    hwy_sub = df[df["speed_regime"] == "highway"]

    crawl_val = round(float(crawl_sub["final_err_m"].median()), 2) if len(crawl_sub) > 0 else 0.0
    city_val = round(float(city_sub["final_drift_pct"].median()), 2) if len(city_sub) > 0 else 0.0
    hwy_val = round(float(hwy_sub["final_drift_pct"].median()), 2) if len(hwy_sub) > 0 else 0.0
    all_val = round(float(df["final_drift_pct"].median()), 2)

    scorecard = {
        "crawl": {
            "regime": "crawl (< 20 km/h)",
            "metric_name": "median_error_m",
            "value": crawl_val,
            "target": 10.0,
            "unit": "m",
            "n": len(crawl_sub),
            "status": eval_status(crawl_val, 10.0),
        },
        "city": {
            "regime": "city (20-50 km/h)",
            "metric_name": "median_drift_pct",
            "value": city_val,
            "target": 15.0,
            "unit": "%",
            "n": len(city_sub),
            "status": eval_status(city_val, 15.0),
        },
        "highway": {
            "regime": "highway (> 50 km/h)",
            "metric_name": "median_drift_pct",
            "value": hwy_val,
            "target": 10.0,
            "unit": "%",
            "n": len(hwy_sub),
            "status": eval_status(hwy_val, 10.0),
        },
        "all": {
            "regime": "all scenarios",
            "metric_name": "median_drift_pct",
            "value": all_val,
            "target": 10.0,
            "unit": "%",
            "n": len(df),
            "status": eval_status(all_val, 10.0),
        },
    }

    heldout_120 = {
        "label": "independent confirmation, not used for tuning",
        "n": 120,
        "seeds": [319976, 480577, 473995],
        "median_drift_pct": 11.15,
        "mean_of_seed_medians": 10.71,
        "std_of_seed_medians": 1.17,
        "p90_drift_pct": 32.91,
        "share_lt_10_pct": 48.33,
        "share_lt_30_pct": 85.83,
        "unseen_trip_median_pct": 9.66,
        "unseen_n": 60,
        "beats_pure_dr_pct": 86.67,
        "pre_round1_baseline": {
            "median_drift_pct": 11.48,
            "mean_of_seed_medians": 11.13,
            "std_of_seed_medians": 1.50,
            "p90_drift_pct": 37.25,
            "share_lt_10_pct": 42.50,
            "unseen_trip_median_pct": 11.89,
            "beats_pure_dr_pct": 83.33,
        }
    }

    summary_data = {
        "overall": overall,
        "regime_scorecard": scorecard,
        "per_scenario": per_scenario,
        "by_trip": by_trip,
        "by_domain": by_domain,
        "by_speed_regime": by_speed_regime,
        "unseen_vs_seen": unseen_vs_seen,
        "heldout_120": heldout_120,
    }

    json_out = os.path.join(out_dir, "summary_6seed.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Wrote {json_out}")

    # Step 2: stages.json
    stages_data = {
        "description": "Progressive Dead Reckoning Drift across pipeline architectural stages",
        "stages": [
            {
                "stage": "(a) Naive Double Integration",
                "drift_pct": 424.13,
                "error_m": 3452.9,
                "distance_m": 814.0,
                "scenario": "60s_blackout_at_300s",
                "trip": "S-S1 (Highway)",
                "source_file": "benchmarks/run_phase2_es_ekf.py",
                "label": "historical baseline (Phase 2 unconstrained open-loop IMU integration)"
            },
            {
                "stage": "(b) ES-EKF + NHC",
                "drift_pct": 178.79,
                "error_m": 1455.57,
                "distance_m": 814.0,
                "scenario": "60s_blackout_at_300s",
                "trip": "S-S1 (Highway)",
                "source_file": "benchmarks/run_phase2_es_ekf.py",
                "label": "historical baseline (Phase 2 15-state ES-EKF with non-holonomic velocity constraints)"
            },
            {
                "stage": "(c) AI Velocity Estimator + ES-EKF (No Map)",
                "drift_pct": overall["pure_dr_median_drift_pct"],
                "metric": "median_across_canonical_seeds",
                "source_file": "ppt_pack/data/runs_6seed.csv (pure_dr_drift_pct)",
                "label": "current production AI speed + ES-EKF (open-loop without topological map matching)"
            },
            {
                "stage": "(d) Full Production Pipeline (Smart IDR)",
                "drift_pct": overall["median_drift_pct"],
                "metric": "median_across_canonical_seeds",
                "source_file": "ppt_pack/data/runs_6seed.csv (final_drift_pct)",
                "label": "current full production pipeline with Bayesian MoE, ES-EKF, online speed calibration, and topological road map-matching"
            }
        ]
    }

    stages_out = os.path.join(out_dir, "stages.json")
    with open(stages_out, "w", encoding="utf-8") as f:
        json.dump(stages_data, f, indent=2)
    print(f"Wrote {stages_out}")

if __name__ == "__main__":
    main()
