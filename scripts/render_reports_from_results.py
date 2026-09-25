import os
import sys
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.run_final_benchmark import generate_markdown_report, detect_dynamic_spotlights

def main():
    json_path = os.path.join(ROOT_DIR, "benchmark_results.json")
    with open(json_path, "r", encoding="utf-8-sig") as f:
        bj = json.load(f)

    csv_path = os.path.join(ROOT_DIR, "artifacts", "phase4_unseen_sm_benchmark_results.csv")
    df = pd.read_csv(csv_path)

    detailed_results = []
    for r in df.to_dict(orient="records"):
        r["dist_m"] = r["distance_m"]
        r["trip_id"] = r.get("trip", r.get("trip_id"))
        r["hdg_diff"] = r.get("hdg_seed_err", 0.0)
        detailed_results.append(r)

    spotlights = detect_dynamic_spotlights(detailed_results)

    tot_sc = len(df)
    med_drift = float(df["map_drift_pct"].median())
    p90_drift = float(df["map_drift_pct"].quantile(0.90))
    t1_count = len(df[df["map_drift_pct"] < 10.0])
    t2_count = len(df[(df["map_drift_pct"] >= 10.0) & (df["map_drift_pct"] <= 30.0)])
    t3_count = len(df[df["map_drift_pct"] > 30.0])

    crawl_df = df[df["distance_m"] < 250.0]
    city_df = df[(df["distance_m"] >= 250.0) & (df["distance_m"] <= 550.0)]
    hwy_df = df[df["distance_m"] > 550.0]
    crawl_err_m = float(crawl_df["map_err_m"].median()) if len(crawl_df) > 0 else 0.0
    city_drift = float(city_df["map_drift_pct"].median()) if len(city_df) > 0 else 0.0
    hwy_drift = float(hwy_df["map_drift_pct"].median()) if len(hwy_df) > 0 else 0.0

    hwy_dom_drift = float(df[df["domain"] == "Highway"]["map_drift_pct"].median())
    art_dom_drift = float(df[df["domain"] == "Arterial"]["map_drift_pct"].median())
    urb_dom_drift = float(df[df["domain"] == "Urban"]["map_drift_pct"].median())
    mix_dom_drift = float(df[df["domain"] == "Mixed"]["map_drift_pct"].median())

    trip_configs = [
        ("S-M", 8, "Highway"),
        ("S-S2", 6, "Arterial"),
        ("S-S1", 6, "Urban"),
        ("S-S3a", 10, "Mixed"),
        ("S-S4", 10, "Arterial"),
    ]

    trip_stats = {}
    for tid, count, dom in trip_configs:
        sub = df[df["trip"].str.startswith(tid)]
        if len(sub) > 0:
            trip_stats[tid] = {
                "domain": dom,
                "count": len(sub),
                "map_med": float(sub["map_drift_pct"].median()),
                "pure_med": float(sub["pure_drift_pct"].median()),
                "mean_dist": float(sub["distance_m"].mean()),
            }

    multi_seed_results = []
    fixed_eval = bj.get("canonical_6_seed_fixed_evaluation")
    if fixed_eval and "per_seed_evaluations" in fixed_eval:
        for pse in fixed_eval["per_seed_evaluations"]:
            multi_seed_results.append({
                "seed": pse["seed"],
                "med_drift": pse["osm_median_drift_pct"],
                "pure_med_drift": pse["pure_dr_median_drift_pct"],
                "t1_count": pse["tier1_count"],
                "t2_count": pse["sub30_count"] - pse["tier1_count"],
                "tot_sc": 40,
                "hwy_dom_drift": pse["highway_drift_pct"],
                "art_dom_drift": pse["arterial_drift_pct"],
                "urb_dom_drift": pse["urban_drift_pct"],
                "mix_dom_drift": pse.get("mixed_drift_pct", 0.0),
                "p90_drift": pse.get("osm_p90_drift_pct", 0.0),
                "beats_pure_count": pse.get("beats_pure_count", 0),
                "beats_pure_rate": pse.get("beats_pure_rate_pct", 0.0) / 100.0,
            })

    hdg_seed_csv = os.path.join(ROOT_DIR, "results", "round1", "hdg_seed", "production_scenarios.csv")
    if os.path.exists(hdg_seed_csv):
        hdf = pd.read_csv(hdg_seed_csv)
        mean_hdg_seed_err = float(hdf["hdg_seed_err"].mean())
    elif "hdg_seed_err" in df.columns:
        mean_hdg_seed_err = float(df["hdg_seed_err"].mean())
    else:
        mean_hdg_seed_err = 18.18

    generate_markdown_report(
        df, detailed_results, spotlights, med_drift, p90_drift, t1_count, t2_count, t3_count, tot_sc,
        crawl_err_m, city_drift, hwy_drift, hwy_dom_drift, art_dom_drift, urb_dom_drift,
        mix_dom_drift=mix_dom_drift, trip_stats=trip_stats, trip_configs=trip_configs,
        mean_hdg_seed_err=mean_hdg_seed_err, multi_seed_results=multi_seed_results,
    )
    print("Reports re-rendered successfully!")

if __name__ == "__main__":
    main()
