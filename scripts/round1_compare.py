"""
Paired comparison of two round1_eval scenario CSVs (same seeds -> same scenarios,
scenario selection depends only on GNSS, not on the model or flags).

    python scripts/round1_compare.py A.csv B.csv [--name-a base --name-b cand]

Reports: better/worse counts (|delta| > 0.5 pp), median delta, sign-test p-value,
split by seen-trip benchmark part (S-M, S-S1, S-S2 Part 3) vs unseen trips (S-S3a, S-S4),
and, if present, how often the pre-blackout speed scale hits its clip ceiling.
"""

from __future__ import annotations

import argparse
import math

import pandas as pd


def sign_test_p(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def block(j: pd.DataFrame, label: str) -> str:
    d = j["map_drift_pct_b"] - j["map_drift_pct_a"]
    b, w = int((d < -0.5).sum()), int((d > 0.5).sum())
    return (f"{label:10s} n={len(j):4d} | median A {j['map_drift_pct_a'].median():6.2f} B {j['map_drift_pct_b'].median():6.2f} | "
            f"better {b:3d} worse {w:3d} | median delta {d.median():+.2f} pp | mean delta {d.mean():+.2f} pp | "
            f"worst regression {d.max():+.1f} pp | sign-test p={sign_test_p(b, w):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    args = ap.parse_args()
    A = pd.read_csv(args.a)
    B = pd.read_csv(args.b)
    j = A.merge(B, on=["seed", "scenario_id"], suffixes=("_a", "_b"))
    if len(j) != len(A) or len(j) != len(B):
        print(f"WARNING: scenario sets differ (A {len(A)}, B {len(B)}, matched {len(j)})")
    print(f"A = {args.name_a} ({args.a})\nB = {args.name_b} ({args.b})")
    print(block(j, "ALL"))
    unseen = j["trip_a"].isin(["S-S3a", "S-S4"])
    print(block(j[~unseen], "seen-trip"))
    print(block(j[unseen], "unseen"))
    for dom, g in j.groupby("domain_a"):
        print(block(g, dom))
    for side, df in (("A", A), ("B", B)):
        if "r1_scale_engine" in df.columns:
            s = df["r1_scale_engine"]
            ceil = df["domain"].map(lambda d: 1.35 if d == "Highway" else 1.25)
            print(f"{side}: speed scale median {s.median():.3f} | at clip ceiling {(s >= ceil - 1e-3).mean():.1%} "
                  f"| at floor 0.85 {(s <= 0.851).mean():.1%}")
        if "r1_scale_raw" in df.columns and df["r1_scale_raw"].notna().any():
            r = df["r1_scale_raw"].dropna()
            print(f"{side}: RAW (unclipped) scale p10 {r.quantile(0.1):.3f} | median {r.median():.3f} | p90 {r.quantile(0.9):.3f} "
                  f"| outside [0.85, 1.25] {((r < 0.85) | (r > 1.25)).mean():.1%}")
        if "r1_scale_ekf" in df.columns:
            print(f"{side}: EKF internal speed scale median {df['r1_scale_ekf'].median():.3f} (should be ~1.0)")


if __name__ == "__main__":
    main()
