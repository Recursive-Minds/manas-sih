"""
Test Suite: Report Integrity & Non-Hardcoded Metric Verification.

Verifies that:
1. FINAL_JUDGE_EVALUATION_REPORT.md contains 100% live computed values matching the benchmark CSV.
2. No stale hardcoded literals (e.g. 32.77%, 89.32%, / 35) exist in the summary tables.
3. Every metric reported in the executive summary is dynamically derived from the evaluated scenarios.
"""
import os
import re
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_PATH = os.path.join(ROOT_DIR, "FINAL_JUDGE_EVALUATION_REPORT.md")
CSV_PATH = os.path.join(ROOT_DIR, "artifacts", "phase4_unseen_sm_benchmark_results.csv")
BENCH_SCRIPT = os.path.join(ROOT_DIR, "benchmarks", "run_final_benchmark.py")


def test_report_and_csv_exist():
    assert os.path.exists(REPORT_PATH), f"Missing {REPORT_PATH}"
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    assert os.path.exists(BENCH_SCRIPT), f"Missing {BENCH_SCRIPT}"


def test_benchmark_script_has_no_stale_literals():
    with open(BENCH_SCRIPT, "r", encoding="utf-8") as f:
        code = f.read()

    # Verify no stale literals exist in the markdown generation section
    assert "32.77%" not in code, "Stale baseline literal 32.77% found in run_final_benchmark.py"
    assert "89.32%" not in code, "Stale P90 literal 89.32% found in run_final_benchmark.py"
    assert "/ 35" not in code, "Stale scenario count denominator / 35 found in run_final_benchmark.py"
    assert "80.0% sub-30%" not in code, "Stale 80.0% sub-30% literal found in run_final_benchmark.py"
    assert "43.8% relative error reduction" not in code, "Stale 43.8% reduction literal found in run_final_benchmark.py"
    assert "0.66°" not in code, "Stale heading literal 0.66° found in run_final_benchmark.py"


def test_executive_summary_matches_csv_exactly():
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 40, f"Expected exactly 40 scenarios, found {len(df)}"

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        doc = f.read()

    # Compute ground truth live metrics from CSV
    med_drift = float(df["map_drift_pct"].median())
    p90_drift = float(df["map_drift_pct"].quantile(0.90))
    pure_med = float(df["pure_drift_pct"].median())
    pure_p90 = float(df["pure_drift_pct"].quantile(0.90))

    t1_count = len(df[df["map_drift_pct"] < 10.0])
    pure_t1_count = len(df[df["pure_drift_pct"] < 10.0])
    sub30_count = len(df[df["map_drift_pct"] <= 30.0])
    pure_sub30_count = len(df[df["pure_drift_pct"] <= 30.0])

    tot_sc = len(df)

    # Verify each value appears in the Executive Summary table
    assert f"**{med_drift:.2f}%**" in doc, f"Expected Overall Median Drift **{med_drift:.2f}%** in report"
    assert f"**{pure_med:.2f}%**" in doc, f"Expected Baseline Pure Median **{pure_med:.2f}%** in report"
    assert f"**{p90_drift:.2f}%**" in doc, f"Expected P90 Drift **{p90_drift:.2f}%** in report"
    assert f"**{pure_p90:.2f}%**" in doc, f"Expected Baseline P90 **{pure_p90:.2f}%** in report"

    assert f"{t1_count/tot_sc*100:.1f}% ({t1_count} / {tot_sc})" in doc
    assert f"{pure_t1_count/tot_sc*100:.1f}% ({pure_t1_count} / {tot_sc})" in doc
    assert f"{(sub30_count)/tot_sc*100:.1f}% ({sub30_count} / {tot_sc})" in doc
    assert f"{pure_sub30_count/tot_sc*100:.1f}% ({pure_sub30_count} / {tot_sc})" in doc

    if "init_heading_err_deg" in df.columns:
        med_hdg = float(df["init_heading_err_deg"].median())
        assert f"**{med_hdg:.2f}°**" in doc, f"Expected Initial Heading Error **{med_hdg:.2f}°** in report"


def test_domain_scorecard_matches_csv():
    df = pd.read_csv(CSV_PATH)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        doc = f.read()

    for tid, dom in [("S-M", "Highway"), ("S-S2", "Arterial"), ("S-S1", "Urban"), ("S-S3a", "Mixed"), ("S-S4", "Arterial")]:
        sub = df[df["trip"].str.startswith(tid)]
        m_drift = float(sub["map_drift_pct"].median())
        assert f"**{m_drift:.2f}%**" in doc, f"Expected {tid} median drift **{m_drift:.2f}%** in scorecard"
