"""
Round-1 experiment runner. Orchestration only (Rule 13): it reuses the benchmark's
own data loading and scenario selection, flips round-1 flags, and writes results
to results/round1/<tag>/ ONLY. It never touches README / reports / benchmark_results.json.

Examples (repo root):
  # baseline + parity reference, canonical seeds
  python scripts/round1_eval.py --tag pre_patch --configs config/round1/baseline_off.json
  # after applying the engine edits: all-off must reproduce pre_patch exactly
  python scripts/round1_eval.py --tag parity --configs config/round1/baseline_off.json config/round1/diagnostics.json \
      --assert-parity results/round1/pre_patch/baseline_off_scenarios.csv
  # ablation (first config is the comparison baseline)
  python scripts/round1_eval.py --tag ablation --configs config/round1/baseline_off.json config/round1/t9_engine.json ...
  # held-out seeds need explicit user approval
  python scripts/round1_eval.py --tag heldout --seeds heldout --i-have-user-approval --configs ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sih.round1.config import Round1Config, set_active_config  # noqa: E402

CANONICAL_SEEDS = [541098, 75496, 45736, 12345, 987654, 314159]
HELDOUT_SEEDS = [319976, 480577, 473995]
SCALAR_KEYS = ["dist_m", "duration_s", "map_err_m", "map_drift_pct", "pure_err_m", "pure_drift_pct",
               "final_at_m", "final_ct_m", "hdg_seed_err", "t_start_s"]


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_final_benchmark", os.path.join(ROOT, "benchmarks", "run_final_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scenario_rows(seed: int, cfg_name: str, detailed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in detailed:
        row = {"config": cfg_name, "seed": seed, "scenario_id": r["scenario_id"], "trip": r["trip_id"], "domain": r["domain"]}
        for k in SCALAR_KEYS:
            row[k] = float(r.get(k, np.nan))
        gt = np.asarray(r.get("gt_speeds", []), dtype=np.float64)
        ms = np.asarray(r.get("map_speeds", []), dtype=np.float64)
        if len(gt) and len(gt) == len(ms):
            row["speed_ratio"] = float(ms.sum() / max(gt.sum(), 1e-6))
            row["creep_m"] = float(ms[gt < 0.3].sum() * 0.1)
            row["gt_stop_s"] = float((gt < 0.3).sum() * 0.1)
        for k, v in r.items():
            if k.startswith("r1_") and isinstance(v, (int, float, np.floating, np.integer)):
                row[k] = float(v)
        rows.append(row)
    return rows


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    per_seed = df.groupby("seed")["map_drift_pct"].median()
    return {
        "n_scenarios": int(len(df)),
        "median_of_seed_medians": float(per_seed.median()),
        "mean_seed_median": float(per_seed.mean()),
        "std_seed_median": float(per_seed.std(ddof=0)),
        "per_seed_median": {int(k): round(float(v), 3) for k, v in per_seed.items()},
        "p90": float(df["map_drift_pct"].quantile(0.9)),
        "t1_lt10_share": float((df["map_drift_pct"] < 10).mean()),
        "beats_pure_share": float((df["map_drift_pct"] < df["pure_drift_pct"]).mean()),
        "worst": float(df["map_drift_pct"].max()),
        "median_abs_at_m": float(df["final_at_m"].abs().median()),
        "median_abs_ct_m": float(df["final_ct_m"].abs().median()),
        "by_domain": {d: round(float(g["map_drift_pct"].median()), 3) for d, g in df.groupby("domain")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="canonical", help="canonical | heldout | comma list")
    ap.add_argument("--i-have-user-approval", action="store_true", help="required for held-out seeds")
    ap.add_argument("--model-path", default=None, help="checkpoint, or comma-separated list = mean ensemble")
    ap.add_argument("--map-source", default="osm", choices=["osm", "masked", "trip"])
    ap.add_argument("--assert-parity", default=None, help="reference *_scenarios.csv; first config must match exactly")
    args = ap.parse_args()

    if args.seeds == "canonical":
        seeds = CANONICAL_SEEDS
    elif args.seeds == "heldout":
        if not args.i_have_user_approval:
            print("Held-out seeds are locked. Ask the user for 'go held-out', then pass --i-have-user-approval.")
            return 2
        seeds = HELDOUT_SEEDS
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
        if set(seeds) & set(HELDOUT_SEEDS) and not args.i_have_user_approval:
            print("List contains held-out seeds; approval flag required.")
            return 2

    out_dir = os.path.join(ROOT, "results", "round1", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    cfgs = [Round1Config.from_json(p) for p in args.configs]

    import torch
    bm = load_benchmark_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    # the production profile (if promoted) must not silently change the "base" model here
    set_active_config(Round1Config(name="precompute"))
    pre = bm.load_precomputed_benchmark_data(device, model_path=args.model_path, map_source=args.map_source)
    print(f"[round1_eval] precompute {time.time() - t0:.0f}s")

    all_summ = {}
    frames = []
    for cfg in cfgs:
        set_active_config(cfg)
        rows = []
        for seed in seeds:
            _, detailed, _ = bm.evaluate_seed_scenarios(seed, pre, map_source=args.map_source)
            rows += scenario_rows(seed, cfg.name, detailed)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"{cfg.name}_scenarios.csv"), index=False)
        all_summ[cfg.name] = summarise(df)
        frames.append(df)
        s = all_summ[cfg.name]
        print(f"[round1_eval] {cfg.name:28s} median {s['median_of_seed_medians']:.2f} | mean {s['mean_seed_median']:.2f} "
              f"+- {s['std_seed_median']:.2f} | p90 {s['p90']:.1f} | T1 {s['t1_lt10_share']:.2f} | worst {s['worst']:.1f}")
    set_active_config(None)

    # paired comparison against the first config
    base = frames[0].set_index(["seed", "scenario_id"])
    for df in frames[1:]:
        cur = df.set_index(["seed", "scenario_id"])
        j = base[["map_drift_pct"]].join(cur[["map_drift_pct"]], rsuffix="_new", how="inner")
        d = j["map_drift_pct_new"] - j["map_drift_pct"]
        all_summ[df["config"].iloc[0]]["vs_first"] = {
            "better": int((d < -0.5).sum()), "worse": int((d > 0.5).sum()),
            "median_delta_pp": float(d.median()), "worst_regression_pp": float(d.max()),
        }

    parity = None
    if args.assert_parity:
        ref = pd.read_csv(args.assert_parity).set_index(["seed", "scenario_id"])
        cur = frames[0].set_index(["seed", "scenario_id"])
        same_index = ref.index.equals(cur.index)
        max_diff = float(np.max(np.abs(ref["map_err_m"].values - cur["map_err_m"].values))) if same_index else float("inf")
        parity = {"reference": args.assert_parity, "same_scenarios": bool(same_index), "max_abs_diff_map_err_m": max_diff,
                  "pass": bool(same_index and max_diff <= 1e-9)}
        # every diagnostics-only / all-off config must also match
        for df in frames[1:]:
            c = next(c for c in cfgs if c.name == df["config"].iloc[0])
            if c.is_all_off() or (c.diagnostics and c.to_dict() == Round1Config(name=c.name, diagnostics=True).to_dict()):
                cc = df.set_index(["seed", "scenario_id"])
                md = float(np.max(np.abs(ref["map_err_m"].values - cc["map_err_m"].values)))
                parity[f"{c.name}_max_abs_diff"] = md
                parity["pass"] = parity["pass"] and md <= 1e-9
        print(f"[round1_eval] PARITY {'PASS' if parity['pass'] else 'FAIL'}: {parity}")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "model_path": args.model_path, "map_source": args.map_source,
                   "configs": {c.name: c.to_dict() for c in cfgs}, "summary": all_summ, "parity": parity,
                   "elapsed_s": time.time() - t0}, f, indent=2, default=str)
    print(f"[round1_eval] wrote {out_dir}/summary.json")
    return 1 if (parity is not None and not parity["pass"]) else 0


if __name__ == "__main__":
    sys.exit(main())
