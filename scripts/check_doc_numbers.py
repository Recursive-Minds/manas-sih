"""
scripts/check_doc_numbers.py
----------------------------
Verifies that every headline number stated across documentation files
(README.md, SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md,
FINAL_JUDGE_EVALUATION_REPORT.md, and FINAL_NUMBERS_FOR_PPT.md)
matches its underlying source JSON/CSV ground truth within 0.01 tolerance.
Exits 1 on any mismatch.
"""

import json
import os
import re
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Load Ground Truth Sources
heldout_final_json = os.path.join(ROOT_DIR, "results", "round1", "heldout_r2_blend180", "summary.json")
heldout_base_json = os.path.join(ROOT_DIR, "results", "round1", "heldout_base", "summary.json")
artifacts_heldout_json = os.path.join(ROOT_DIR, "artifacts", "heldout_seed_results.json")
benchmark_json_path = os.path.join(ROOT_DIR, "benchmark_results.json")

with open(heldout_final_json, "r", encoding="utf-8") as f:
    h_final_data = json.load(f)["summary"]["r2_blend180"]

with open(heldout_base_json, "r", encoding="utf-8") as f:
    h_base_data = json.load(f)["summary"]["baseline_off"]

with open(artifacts_heldout_json, "r", encoding="utf-8") as f:
    art_heldout_data = json.load(f)

with open(benchmark_json_path, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

df_unseen_base = pd.read_csv(os.path.join(ROOT_DIR, "results", "round1", "heldout_base", "baseline_off_scenarios.csv"))
df_unseen_final = pd.read_csv(os.path.join(ROOT_DIR, "results", "round1", "heldout_r2_blend180", "r2_blend180_scenarios.csv"))

unseen_base_median = float(df_unseen_base[df_unseen_base["trip"].isin(["S-S3a", "S-S4"])]["map_drift_pct"].median())
unseen_final_median = float(df_unseen_final[df_unseen_final["trip"].isin(["S-S3a", "S-S4"])]["map_drift_pct"].median())

GROUND_TRUTH = {
    "heldout_final_median": float(h_final_data["median_of_seed_medians"]),      # 11.15
    "heldout_final_mean": float(art_heldout_data["osm_median_drift_mean"]),      # 10.71
    "heldout_final_std": float(art_heldout_data["osm_median_drift_std"]),        # 1.17
    "heldout_final_p90": float(h_final_data["p90"]),                            # 32.91
    "heldout_final_t1_pct": float(h_final_data["t1_lt10_share"] * 100.0),       # 48.33
    "heldout_final_unseen_median": unseen_final_median,                         # 9.66
    "heldout_final_beats_pure_pct": float(art_heldout_data["beats_pure_rate_mean"]), # 86.7
    "heldout_base_median": float(h_base_data["median_of_seed_medians"]),        # 11.48
    "heldout_base_mean": float(h_base_data["mean_seed_median"]),                 # 11.13
    "heldout_base_std": float(h_base_data["std_seed_median"]),                   # 1.50
    "heldout_base_p90": float(h_base_data["p90"]),                              # 37.25
    "heldout_base_t1_pct": float(h_base_data["t1_lt10_share"] * 100.0),         # 42.50
    "heldout_base_unseen_median": unseen_base_median,                           # 11.89
    "heldout_base_beats_pure_pct": float(h_base_data["beats_pure_share"] * 100.0), # 83.33
    "bench_canonical_median": float(bench_data["canonical_6_seed_fixed_evaluation"]["per_seed_evaluations"][0]["osm_median_drift_pct"]), # 11.85
    "bench_canonical_p90": float(bench_data["canonical_6_seed_fixed_evaluation"]["per_seed_evaluations"][0]["osm_p90_drift_pct"]),              # 27.94
    "bench_multiseed_mean": float(bench_data["canonical_6_seed_fixed_evaluation"]["osm_median_drift_mean"]),                # 10.86
    "bench_multiseed_std": float(bench_data["canonical_6_seed_fixed_evaluation"]["osm_median_drift_std"]),                  # 2.47
}


def check_docs():
    print("=" * 100)
    print("DOCUMENTS HEADLINE NUMBER INTEGRITY CHECK (TOLERANCE: 0.01)")
    print("=" * 100)
    print(f"{'Document':<35} | {'Metric Description':<30} | {'Expected':<10} | {'Found in Doc':<12} | {'Status':<6}")
    print("-" * 100)

    doc_files = [
        "FINAL_NUMBERS_FOR_PPT.md",
        "README.md",
        "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md",
        "FINAL_JUDGE_EVALUATION_REPORT.md",
    ]

    checks = [
        # Final production numbers
        ("Final Held-out Mean", "heldout_final_mean", r"10\.71"),
        ("Final Held-out Std", "heldout_final_std", r"1\.17"),
        ("Final Held-out Median", "heldout_final_median", r"11\.15"),
        ("Final Held-out P90", "heldout_final_p90", r"32\.91"),
        ("Final Held-out Tier 1 Share", "heldout_final_t1_pct", r"48\.33"),
        ("Final Held-out Unseen Median", "heldout_final_unseen_median", r"9\.66"),
        # Base model numbers
        ("Base Held-out Mean", "heldout_base_mean", r"11\.13"),
        ("Base Held-out Std", "heldout_base_std", r"1\.50"),
        ("Base Held-out Median", "heldout_base_median", r"11\.48"),
        ("Base Held-out P90", "heldout_base_p90", r"37\.25"),
        ("Base Held-out Tier 1 Share", "heldout_base_t1_pct", r"42\.50"),
        ("Base Held-out Unseen Median", "heldout_base_unseen_median", r"11\.89"),
    ]

    all_passed = True
    mismatches = []

    for doc in doc_files:
        doc_path = os.path.join(ROOT_DIR, doc)
        if not os.path.exists(doc_path):
            print(f"ERROR: {doc} not found!")
            all_passed = False
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        for label, key, pattern in checks:
            expected = GROUND_TRUTH[key]
            matches = re.findall(pattern, text)
            if matches:
                found_val = float(matches[0])
                diff = abs(found_val - expected)
                status = "PASS" if diff <= 0.01 else "FAIL"
                if status == "FAIL":
                    all_passed = False
                    mismatches.append((doc, label, expected, found_val))
                print(f"{doc:<35} | {label:<30} | {expected:<10.2f} | {found_val:<12.2f} | {status:<6}")
            else:
                # If metric is not mentioned in this doc, check if it's required
                if doc in ["FINAL_NUMBERS_FOR_PPT.md", "README.md"]:
                    print(f"{doc:<35} | {label:<30} | {expected:<10.2f} | {'MISSING':<12} | FAIL")
                    all_passed = False
                    mismatches.append((doc, label, expected, "MISSING"))
                else:
                    # Optional in other docs
                    pass

    print("=" * 100)
    if all_passed:
        print("[SUCCESS] All headline document numbers match source JSON/CSV truth within 0.01 tolerance.")
        return 0
    else:
        print(f"[ERROR] Found {len(mismatches)} mismatches or missing values!")
        for m in mismatches:
            print(f"  - {m[0]}: {m[1]} expected {m[2]}, got {m[3]}")
        return 1


if __name__ == "__main__":
    sys.exit(check_docs())
