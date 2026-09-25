"""
scripts/build_number_registry.py
--------------------------------
Generates docs/NUMBER_SOURCES.json with exact provenance, computation selectors,
and models for all result numbers cited across documentation files.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

registry = {
    # ---------------------------------------------------------
    # 1. Held-Out Primary Results (3 Seeds, 120 Scenarios)
    # ---------------------------------------------------------
    "heldout_osm_median_drift_mean": {
        "metric": "Held-Out Mean of Seed Medians",
        "value": 10.71,
        "unit": "%",
        "source_file": "artifacts/heldout_seed_results.json",
        "selector": "osm_median_drift_mean",
        "computation": "mean of seed medians across 3 held-out seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_osm_median_drift_std": {
        "metric": "Held-Out Std of Seed Medians",
        "value": 1.17,
        "unit": "%",
        "source_file": "artifacts/heldout_seed_results.json",
        "selector": "osm_median_drift_std",
        "computation": "std of seed medians across 3 held-out seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_overall_median": {
        "metric": "Held-Out Overall Median Drift",
        "value": 11.15,
        "unit": "%",
        "source_file": "results/round1/heldout_r2_blend180/summary.json",
        "selector": "summary.r2_blend180.median_of_seed_medians",
        "computation": "median of per-seed medians (11.148% -> 11.15%)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_p90_drift": {
        "metric": "Held-Out P90 (90th Percentile) Drift",
        "value": 32.91,
        "unit": "%",
        "source_file": "results/round1/heldout_r2_blend180/summary.json",
        "selector": "summary.r2_blend180.p90",
        "computation": "quantile 0.90 of map_drift_pct across 120 held-out scenarios",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_tier1_share": {
        "metric": "Held-Out Tier 1 Share (< 10% Drift)",
        "value": 48.33,
        "unit": "%",
        "source_file": "results/round1/heldout_r2_blend180/summary.json",
        "selector": "summary.r2_blend180.t1_lt10_share",
        "computation": "58 / 120 = 48.333% -> 48.33%",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_unseen_trips_median": {
        "metric": "Held-Out Unseen Trips Median Drift (S-S3a, S-S4)",
        "value": 9.66,
        "unit": "%",
        "source_file": "results/round1/heldout_r2_blend180/r2_blend180_scenarios.csv",
        "selector": "trip in ['S-S3a', 'S-S4']",
        "computation": "median of map_drift_pct over 60 unseen trip scenarios",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_beats_pure_rate": {
        "metric": "Held-Out Beats Pure DR Rate",
        "value": 86.67,
        "unit": "%",
        "source_file": "results/round1/heldout_r2_blend180/summary.json",
        "selector": "summary.r2_blend180.beats_pure_share",
        "computation": "104 / 120 = 86.667% -> 86.67%",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "heldout_pure_dr_mean": {
        "metric": "Held-Out Pure DR Mean Drift",
        "value": 22.93,
        "unit": "%",
        "source_file": "artifacts/heldout_seed_results.json",
        "selector": "pure_dr_median_drift_mean",
        "computation": "mean of pure DR medians across 3 held-out seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md", "README.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"
        ]
    },
    "heldout_pure_dr_std": {
        "metric": "Held-Out Pure DR Std Drift",
        "value": 0.69,
        "unit": "%",
        "source_file": "artifacts/heldout_seed_results.json",
        "selector": "pure_dr_median_drift_std",
        "computation": "std of pure DR medians across 3 held-out seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": [
            "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md", "README.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"
        ]
    },
    "heldout_min_median": {
        "metric": "Held-Out Minimum Seed Median",
        "value": 9.11,
        "unit": "%",
        "source_file": "artifacts/heldout_seed_results.json",
        "selector": "per_seed_evaluations min osm_median_drift_pct",
        "computation": "min of seed medians across 3 held-out seeds (9.11%)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_JUDGE_EVALUATION_REPORT.md", "README.md"]
    },
    "heldout_max_median": {
        "metric": "Held-Out Maximum Seed Median",
        "value": 11.86,
        "unit": "%",
        "source_file": "artifacts/heldout_seed_results.json",
        "selector": "per_seed_evaluations max osm_median_drift_pct",
        "computation": "max of seed medians across 3 held-out seeds (11.86%)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_JUDGE_EVALUATION_REPORT.md", "README.md"]
    },

    # ---------------------------------------------------------
    # 2. Held-Out Baseline Pre-Round-1 Results
    # ---------------------------------------------------------
    "heldout_base_mean": {
        "metric": "Base Model Held-Out Mean Drift",
        "value": 11.13,
        "unit": "%",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "summary.baseline_off.mean_seed_median",
        "computation": "mean of baseline seed medians across 3 held-out seeds",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_base_std": {
        "metric": "Base Model Held-Out Std Drift",
        "value": 1.50,
        "unit": "%",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "summary.baseline_off.std_seed_median",
        "computation": "std of baseline seed medians across 3 held-out seeds",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_base_median": {
        "metric": "Base Model Held-Out Median Drift",
        "value": 11.48,
        "unit": "%",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "summary.baseline_off.median_of_seed_medians",
        "computation": "median of baseline seed medians across 3 held-out seeds",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_base_p90": {
        "metric": "Base Model Held-Out P90 Drift",
        "value": 37.25,
        "unit": "%",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "summary.baseline_off.p90",
        "computation": "quantile 0.90 of baseline map_drift_pct across 120 held-out scenarios",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_base_tier1_share": {
        "metric": "Base Model Held-Out Tier 1 Share",
        "value": 42.50,
        "unit": "%",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "summary.baseline_off.t1_lt10_share",
        "computation": "51 / 120 = 42.50%",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_base_unseen_median": {
        "metric": "Base Model Held-Out Unseen Trips Median",
        "value": 11.89,
        "unit": "%",
        "source_file": "results/round1/heldout_base/baseline_off_scenarios.csv",
        "selector": "trip in ['S-S3a', 'S-S4']",
        "computation": "median of baseline map_drift_pct over 60 unseen trip scenarios",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_base_beats_pure_rate": {
        "metric": "Base Model Held-Out Beats Pure DR Rate",
        "value": 83.33,
        "unit": "%",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "summary.baseline_off.beats_pure_share",
        "computation": "100 / 120 = 83.333% -> 83.33%",
        "model": "pre-round-1 (round1_interval_lam0.5_s42.pt with all flags OFF)",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md"]
    },
    "heldout_p90_tightening_pp": {
        "metric": "Held-Out P90 Tail Tightening Delta",
        "value": 4.34,
        "unit": "pp",
        "source_file": "results/round1/heldout_base/summary.json",
        "selector": "37.25% - 32.91%",
        "computation": "difference in P90 between baseline_off and r2_blend180 (4.34 pp)",
        "model": "production vs base comparison",
        "seed_set": "held-out 3 seeds, 120 sc.",
        "tolerance": 0.01,
        "docs": ["FINAL_NUMBERS_FOR_PPT.md", "README.md"]
    },

    # ---------------------------------------------------------
    # 3. Dev Multi-Seed Results (6 Seeds, 236 Scenarios)
    # ---------------------------------------------------------
    "dev_multiseed_osm_mean": {
        "metric": "Dev Multi-Seed Mean of Seed Medians",
        "value": 10.86,
        "unit": "%",
        "source_file": "benchmark_results.json",
        "selector": "canonical_6_seed_fixed_evaluation.osm_median_drift_mean",
        "computation": "mean of seed medians across 6 dev seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "dev 6 seeds, 236 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "dev_multiseed_osm_std": {
        "metric": "Dev Multi-Seed Std of Seed Medians",
        "value": 2.47,
        "unit": "%",
        "source_file": "benchmark_results.json",
        "selector": "canonical_6_seed_fixed_evaluation.osm_median_drift_std",
        "computation": "std of seed medians across 6 dev seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "dev 6 seeds, 236 sc.",
        "tolerance": 0.02,
        "docs": [
            "README.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "dev_multiseed_osm_median": {
        "metric": "Dev Multi-Seed Median of Seed Medians",
        "value": 11.76,
        "unit": "%",
        "source_file": "benchmark_results.json",
        "selector": "canonical_6_seed_fixed_evaluation.per_seed_evaluations",
        "computation": "median of seed medians [11.85, 13.54, 11.66, 12.75, 9.12, 6.53] = 11.755% -> 11.76%",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "dev 6 seeds, 236 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md",
            "CLAUDE.md", "GEMINI.md"
        ]
    },
    "dev_multiseed_osm_p90": {
        "metric": "Dev Multi-Seed Mean P90 Drift",
        "value": 36.60,
        "unit": "%",
        "source_file": "benchmark_results.json",
        "selector": "canonical_6_seed_fixed_evaluation.osm_p90_drift_mean",
        "computation": "mean of P90s across 6 dev seeds",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "dev 6 seeds, 236 sc.",
        "tolerance": 0.01,
        "docs": ["README.md", "FINAL_JUDGE_EVALUATION_REPORT.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]
    },

    # ---------------------------------------------------------
    # 4. Canonical Dev Seed 541098 Overall Results
    # ---------------------------------------------------------
    "canonical_seed_541098_median": {
        "metric": "Canonical Dev Seed 541098 Median Drift",
        "value": 11.85,
        "unit": "%",
        "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv",
        "selector": "map_drift_pct",
        "computation": "median of map_drift_pct across 40 scenarios on Seed 541098",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "canonical_seed_541098_p90": {
        "metric": "Canonical Dev Seed 541098 P90 Drift",
        "value": 27.94,
        "unit": "%",
        "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv",
        "selector": "map_drift_pct",
        "computation": "quantile 0.90 of map_drift_pct across 40 scenarios on Seed 541098",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "canonical_seed_541098_tier1_count": {
        "metric": "Canonical Dev Seed 541098 Tier 1 Count",
        "value": 18,
        "unit": "count",
        "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv",
        "selector": "map_drift_pct < 10.0",
        "computation": "count of scenarios with map_drift_pct < 10.0",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.0,
        "docs": [
            "README.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "canonical_seed_541098_tier1_pct": {
        "metric": "Canonical Dev Seed 541098 Tier 1 Percentage",
        "value": 45.0,
        "unit": "%",
        "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv",
        "selector": "map_drift_pct < 10.0",
        "computation": "18 / 40 = 45.0%",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "canonical_seed_541098_sub30_count": {
        "metric": "Canonical Dev Seed 541098 Sub-30 Count",
        "value": 37,
        "unit": "count",
        "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv",
        "selector": "map_drift_pct <= 30.0",
        "computation": "count of scenarios with map_drift_pct <= 30.0",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.0,
        "docs": ["README.md", "FINAL_JUDGE_EVALUATION_REPORT.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]
    },
    "canonical_seed_541098_sub30_pct": {
        "metric": "Canonical Dev Seed 541098 Sub-30 Percentage",
        "value": 92.5,
        "unit": "%",
        "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv",
        "selector": "map_drift_pct <= 30.0",
        "computation": "37 / 40 = 92.5%",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.01,
        "docs": ["README.md", "FINAL_JUDGE_EVALUATION_REPORT.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]
    },

    # ---------------------------------------------------------
    # 5. Heading Seeding Diagnostics (Blackout Start Value)
    # ---------------------------------------------------------
    "hdg_seed_err_dev_mean": {
        "metric": "Dev Seeds Initial Heading Seeding Error Mean",
        "value": 18.18,
        "unit": "°",
        "source_file": "results/round1/hdg_seed/production_scenarios.csv",
        "selector": "hdg_seed_err",
        "computation": "mean of hdg_seed_err across 236 dev scenarios (18.1843° -> 18.18°)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "dev 6 seeds, 236 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "PROBLEM_STATEMENT_AND_INITIAL_PLAN.md",
            "ROUND1_README.md"
        ]
    },
    "hdg_seed_err_dev_median": {
        "metric": "Dev Seeds Initial Heading Seeding Error Median",
        "value": 7.05,
        "unit": "°",
        "source_file": "results/round1/hdg_seed/production_scenarios.csv",
        "selector": "hdg_seed_err",
        "computation": "median of hdg_seed_err across 236 dev scenarios (7.0515° -> 7.05°)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "dev 6 seeds, 236 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md",
            "PROBLEM_STATEMENT_AND_INITIAL_PLAN.md", "ROUND1_README.md"
        ]
    },
    "hdg_seed_err_seed541098_mean": {
        "metric": "Seed 541098 Initial Heading Seeding Error Mean",
        "value": 17.15,
        "unit": "°",
        "source_file": "results/round1/hdg_seed/production_scenarios.csv",
        "selector": "seed == 541098, hdg_seed_err",
        "computation": "mean of hdg_seed_err on Seed 541098 across 40 scenarios (17.1469° -> 17.15°)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md",
            "PROBLEM_STATEMENT_AND_INITIAL_PLAN.md", "ROUND1_README.md"
        ]
    },
    "hdg_seed_err_seed541098_median": {
        "metric": "Seed 541098 Initial Heading Seeding Error Median",
        "value": 8.30,
        "unit": "°",
        "source_file": "results/round1/hdg_seed/production_scenarios.csv",
        "selector": "seed == 541098, hdg_seed_err",
        "computation": "median of hdg_seed_err on Seed 541098 across 40 scenarios (8.2952° -> 8.30°)",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 40 sc.",
        "tolerance": 0.01,
        "docs": [
            "README.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md",
            "PROBLEM_STATEMENT_AND_INITIAL_PLAN.md", "ROUND1_README.md"
        ]
    },

    # ---------------------------------------------------------
    # 6. Streaming Parity Verification
    # ---------------------------------------------------------
    "parity_endpoint_diff": {
        "metric": "Batch vs Streaming Endpoint Parity Difference",
        "value": 0.0000,
        "unit": "m",
        "source_file": "scripts/quick_parity.py",
        "selector": "endpoint_diff across 5 canonical scenarios (#22, #23, #25, #26, #30)",
        "computation": "exact difference = 0.0000 m",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, 5 canonical scenarios",
        "tolerance": 0.0001,
        "docs": [
            "README.md", "FINAL_NUMBERS_FOR_PPT.md", "FINAL_JUDGE_EVALUATION_REPORT.md",
            "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md", "APP_REPORT.md", "APP_STATUS_REPORT.md",
            "ROUND1_README.md", "CLAUDE.md", "GEMINI.md"
        ]
    },
    "parity_sc22_err_m": {
        "metric": "Parity Scenario 22 Final Error",
        "value": 16.25,
        "unit": "m",
        "source_file": "scripts/quick_parity.py",
        "selector": "Scenario #22 endpoint error",
        "computation": "16.25 m in batch and streaming",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, S-S3a",
        "tolerance": 0.01,
        "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]
    },
    "parity_sc23_err_m": {
        "metric": "Parity Scenario 23 Final Error",
        "value": 67.77,
        "unit": "m",
        "source_file": "scripts/quick_parity.py",
        "selector": "Scenario #23 endpoint error",
        "computation": "67.77 m in batch and streaming",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, S-S3a",
        "tolerance": 0.01,
        "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]
    },
    "parity_sc25_err_m": {
        "metric": "Parity Scenario 25 Final Error",
        "value": 77.30,
        "unit": "m",
        "source_file": "scripts/quick_parity.py",
        "selector": "Scenario #25 endpoint error",
        "computation": "77.30 m in batch and streaming",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, S-S3a",
        "tolerance": 0.01,
        "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]
    },
    "parity_sc26_err_m": {
        "metric": "Parity Scenario 26 Final Error",
        "value": 122.77,
        "unit": "m",
        "source_file": "scripts/quick_parity.py",
        "selector": "Scenario #26 endpoint error",
        "computation": "122.77 m in batch and streaming",
        "model": "production (round1_interval_lam0.5_s42.pt + production.json)",
        "seed_set": "seed 541098, S-S3a",
        "tolerance": 0.01,
        "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]
    },

    # ---------------------------------------------------------
    # 7. Model Footprint
    # ---------------------------------------------------------
    "model_torchscript_size": {
        "metric": "TorchScript Model File Size",
        "value": 2.66,
        "unit": "MB",
        "source_file": "models/exported/moe_velocity_model.torchscript.pt",
        "selector": "os.path.getsize / (1024 * 1024)",
        "computation": "2789139 / (1024*1024) = 2.6599 MB -> 2.66 MB",
        "model": "production (round1_interval_lam0.5_s42.pt exported)",
        "seed_set": "N/A",
        "tolerance": 0.01,
        "docs": [
            "README.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md",
            "PROBLEM_STATEMENT_AND_INITIAL_PLAN.md", "CLAUDE.md", "GEMINI.md"
        ]
    },

    # ---------------------------------------------------------
    # 8. Scenario Spotlights (Seed 541098, phase4_unseen_sm_benchmark_results.csv)
    # ---------------------------------------------------------
    "sc03_distance_m": {"metric": "Sc 03 Outage Distance", "value": 1174.60, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 03 distance_m", "computation": "1174.60m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc03_pure_err_m": {"metric": "Sc 03 Pure Error", "value": 357.85, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 03 pure_err_m", "computation": "357.85m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc03_pure_drift_pct": {"metric": "Sc 03 Pure Drift", "value": 30.47, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 03 pure_drift_pct", "computation": "30.47%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc03_map_err_m": {"metric": "Sc 03 Map Error", "value": 100.52, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 03 map_err_m", "computation": "100.52m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc03_map_drift_pct": {"metric": "Sc 03 Map Drift", "value": 8.56, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 03 map_drift_pct", "computation": "8.56%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc10_distance_m": {"metric": "Sc 10 Outage Distance", "value": 245.71, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 10 distance_m", "computation": "245.71m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc10_pure_err_m": {"metric": "Sc 10 Pure Error", "value": 113.68, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 10 pure_err_m", "computation": "113.68m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc10_pure_drift_pct": {"metric": "Sc 10 Pure Drift", "value": 46.27, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 10 pure_drift_pct", "computation": "46.27%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc10_map_err_m": {"metric": "Sc 10 Map Error", "value": 32.02, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 10 map_err_m", "computation": "32.02m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc10_map_drift_pct": {"metric": "Sc 10 Map Drift", "value": 13.03, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 10 map_drift_pct", "computation": "13.03%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc14_distance_m": {"metric": "Sc 14 Outage Distance", "value": 202.58, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 14 distance_m", "computation": "202.58m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc14_pure_err_m": {"metric": "Sc 14 Pure Error", "value": 32.27, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 14 pure_err_m", "computation": "32.27m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc14_pure_drift_pct": {"metric": "Sc 14 Pure Drift", "value": 15.93, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 14 pure_drift_pct", "computation": "15.93%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc14_map_err_m": {"metric": "Sc 14 Map Error", "value": 32.27, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 14 map_err_m", "computation": "32.27m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc14_map_drift_pct": {"metric": "Sc 14 Map Drift", "value": 15.93, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 14 map_drift_pct", "computation": "15.93%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc15_distance_m": {"metric": "Sc 15 Outage Distance", "value": 399.74, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 15 distance_m", "computation": "399.74m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc15_pure_err_m": {"metric": "Sc 15 Pure Error", "value": 78.24, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 15 pure_err_m", "computation": "78.24m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc15_pure_drift_pct": {"metric": "Sc 15 Pure Drift", "value": 19.57, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 15 pure_drift_pct", "computation": "19.57%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc15_map_err_m": {"metric": "Sc 15 Map Error", "value": 71.89, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 15 map_err_m", "computation": "71.89m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc15_map_drift_pct": {"metric": "Sc 15 Map Drift", "value": 17.98, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 15 map_drift_pct", "computation": "17.98%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc18_distance_m": {"metric": "Sc 18 Outage Distance", "value": 98.90, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 18 distance_m", "computation": "98.90m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc18_pure_err_m": {"metric": "Sc 18 Pure Error", "value": 52.11, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 18 pure_err_m", "computation": "52.11m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc18_pure_drift_pct": {"metric": "Sc 18 Pure Drift", "value": 52.69, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 18 pure_drift_pct", "computation": "52.69%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc18_map_err_m": {"metric": "Sc 18 Map Error", "value": 10.56, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 18 map_err_m", "computation": "10.56m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc18_map_drift_pct": {"metric": "Sc 18 Map Drift", "value": 10.68, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 18 map_drift_pct", "computation": "10.68%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc19_distance_m": {"metric": "Sc 19 Outage Distance", "value": 361.75, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 19 distance_m", "computation": "361.75m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc19_pure_err_m": {"metric": "Sc 19 Pure Error", "value": 37.92, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 19 pure_err_m", "computation": "37.92m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc19_pure_drift_pct": {"metric": "Sc 19 Pure Drift", "value": 10.48, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 19 pure_drift_pct", "computation": "10.48%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc19_map_err_m": {"metric": "Sc 19 Map Error", "value": 13.64, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 19 map_err_m", "computation": "13.64m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc19_map_drift_pct": {"metric": "Sc 19 Map Drift", "value": 3.77, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 19 map_drift_pct", "computation": "3.77%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc27_distance_m": {"metric": "Sc 27 Outage Distance", "value": 591.93, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 27 distance_m", "computation": "591.93m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc27_pure_err_m": {"metric": "Sc 27 Pure Error", "value": 26.55, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 27 pure_err_m", "computation": "26.55m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc27_pure_drift_pct": {"metric": "Sc 27 Pure Drift", "value": 4.48, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 27 pure_drift_pct", "computation": "4.48%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc27_map_err_m": {"metric": "Sc 27 Map Error", "value": 3.48, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 27 map_err_m", "computation": "3.48m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc27_map_drift_pct": {"metric": "Sc 27 Map Drift", "value": 0.59, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 27 map_drift_pct", "computation": "0.59%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc28_distance_m": {"metric": "Sc 28 Outage Distance", "value": 374.52, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 28 distance_m", "computation": "374.52m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc28_pure_err_m": {"metric": "Sc 28 Pure Error", "value": 102.87, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 28 pure_err_m", "computation": "102.87m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc28_pure_drift_pct": {"metric": "Sc 28 Pure Drift", "value": 27.47, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 28 pure_drift_pct", "computation": "27.47%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc28_map_err_m": {"metric": "Sc 28 Map Error", "value": 12.93, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 28 map_err_m", "computation": "12.93m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc28_map_drift_pct": {"metric": "Sc 28 Map Drift", "value": 3.45, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 28 map_drift_pct", "computation": "3.45%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc30_distance_m": {"metric": "Sc 30 Outage Distance", "value": 244.21, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 30 distance_m", "computation": "244.21m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md", "DEMO.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc30_pure_err_m": {"metric": "Sc 30 Pure Error", "value": 19.39, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 30 pure_err_m", "computation": "19.39m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc30_pure_drift_pct": {"metric": "Sc 30 Pure Drift", "value": 7.94, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 30 pure_drift_pct", "computation": "7.94%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc30_map_err_m": {"metric": "Sc 30 Map Error", "value": 7.04, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 30 map_err_m", "computation": "7.04m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md", "DEMO.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc30_map_drift_pct": {"metric": "Sc 30 Map Drift", "value": 2.88, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 30 map_drift_pct", "computation": "2.88%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md", "DEMO.md", "SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc31_distance_m": {"metric": "Sc 31 Outage Distance", "value": 490.87, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 31 distance_m", "computation": "490.87m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc31_pure_err_m": {"metric": "Sc 31 Pure Error", "value": 21.13, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 31 pure_err_m", "computation": "21.13m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc31_pure_drift_pct": {"metric": "Sc 31 Pure Drift", "value": 4.30, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 31 pure_drift_pct", "computation": "4.30%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc31_map_err_m": {"metric": "Sc 31 Map Error", "value": 15.07, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 31 map_err_m", "computation": "15.07m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc31_map_drift_pct": {"metric": "Sc 31 Map Drift", "value": 3.07, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 31 map_drift_pct", "computation": "3.07%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc34_distance_m": {"metric": "Sc 34 Outage Distance", "value": 328.29, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 34 distance_m", "computation": "328.29m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc34_pure_err_m": {"metric": "Sc 34 Pure Error", "value": 191.14, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 34 pure_err_m", "computation": "191.14m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc34_pure_drift_pct": {"metric": "Sc 34 Pure Drift", "value": 58.22, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 34 pure_drift_pct", "computation": "58.22%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc34_map_err_m": {"metric": "Sc 34 Map Error", "value": 3.05, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 34 map_err_m", "computation": "3.05m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc34_map_drift_pct": {"metric": "Sc 34 Map Drift", "value": 0.93, "unit": "%", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 34 map_drift_pct", "computation": "0.93%", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.01, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},

    "sc02_distance_m": {"metric": "Sc 02 Outage Distance", "value": 600.18, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 02 distance_m", "computation": "600.18m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md"]},
    "sc22_distance_m": {"metric": "Sc 22 Outage Distance", "value": 475.17, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 22 distance_m", "computation": "475.17m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]},
    "sc23_distance_m": {"metric": "Sc 23 Outage Distance", "value": 1128.39, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 23 distance_m", "computation": "1128.39m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]},
    "sc25_distance_m": {"metric": "Sc 25 Outage Distance", "value": 614.33, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 25 distance_m", "computation": "614.33m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]},
    "sc26_distance_m": {"metric": "Sc 26 Outage Distance", "value": 892.82, "unit": "m", "source_file": "artifacts/phase4_unseen_sm_benchmark_results.csv", "selector": "sc 26 distance_m", "computation": "892.82m", "model": "production", "seed_set": "seed 541098, 40 sc.", "tolerance": 0.1, "docs": ["APP_REPORT.md", "APP_STATUS_REPORT.md"]}
}

target_path = os.path.join(DOCS_DIR, "NUMBER_SOURCES.json")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print(f"Wrote {len(registry)} registered numbers to {target_path}")
