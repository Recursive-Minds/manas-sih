"""
T2 - Worst-scenario autopsy (diagnosis only, changes nothing).

Runs the benchmark scenarios for one seed with round-1 DIAGNOSTICS on (behaviour
identical to baseline) and explains, per scenario, where the end-point error
comes from:
  speed_err_m   = sum(v_est - v_can) * dt      distance error from speed (along-track)
  creep_m       = distance driven while CAN says stopped (< 0.3 m/s)
  |final_ct_m|  = heading / map-snap error (cross-track)
  scale_eff     = engine.speed_scale * ekf._speed_scale  (T9 double-scale check)
Writes results/round1/autopsy/autopsy.md + one PNG per scenario.

    python scripts/round1_autopsy.py --seed 541098 --ids 9 40 35 --worst 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from sih.round1.config import Round1Config, set_active_config  # noqa: E402
from round1_eval import load_benchmark_module  # noqa: E402


def turn_deg(gt_pts: np.ndarray) -> float:
    d = np.diff(gt_pts, axis=0)
    keep = np.linalg.norm(d, axis=1) > 2.0
    if keep.sum() < 2:
        return 0.0
    b = np.degrees(np.arctan2(d[keep, 0], d[keep, 1]))
    db = (np.diff(b) + 180.0) % 360.0 - 180.0
    return float(np.sum(np.abs(db)))


def verdict(row) -> str:
    parts = {
        "SPEED/SCALE": abs(row["speed_err_m"] - row["creep_m"]),
        "STOP CREEP": abs(row["creep_m"]),
        "HEADING/CROSS-TRACK": abs(row["final_ct_m"]),
    }
    k = max(parts, key=parts.get)
    return f"{k} ({parts[k]:.0f} m)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=541098)
    ap.add_argument("--ids", type=int, nargs="*", default=[9, 40, 35])
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(ROOT, "results", "round1", "autopsy")
    os.makedirs(out, exist_ok=True)
    bm = load_benchmark_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pre = bm.load_precomputed_benchmark_data(device, model_path=args.model_path)
    set_active_config(Round1Config(name="diagnostics", diagnostics=True))
    _, detailed, metrics = bm.evaluate_seed_scenarios(args.seed, pre)
    set_active_config(None)

    rows = []
    for r in detailed:
        gt = np.asarray(r["gt_speeds"], float)
        ms = np.asarray(r["map_speeds"], float)
        rows.append({
            "id": r["scenario_id"], "trip": r["trip_id"], "domain": r["domain"], "dur": r["duration_s"],
            "dist": r["dist_m"], "map": r["map_drift_pct"], "pure": r["pure_drift_pct"],
            "final_at_m": r["final_at_m"], "final_ct_m": r["final_ct_m"],
            "speed_err_m": float((ms - gt).sum() * 0.1), "creep_m": float(ms[gt < 0.3].sum() * 0.1),
            "stop_s": float((gt < 0.3).sum() * 0.1), "speed_ratio": float(ms.sum() / max(gt.sum(), 1e-6)),
            "turn_deg": turn_deg(np.asarray(r["gt_pts"])), "hdg_seed_err": r.get("hdg_seed_err", np.nan),
            "scale_engine": r.get("r1_scale_engine", np.nan), "scale_ekf": r.get("r1_scale_ekf", np.nan),
            "scale_eff": r.get("r1_scale_effective", np.nan), "entry": r.get("entry_acq_info", ""),
            "_r": r,
        })
    worst = sorted(rows, key=lambda x: -x["map"])[: args.worst]
    focus = {x["id"] for x in worst} | set(args.ids)
    sel = [x for x in rows if x["id"] in focus]

    lines = [f"# Round-1 autopsy - seed {args.seed}", "",
             f"Seed median drift {metrics['med_drift']:.2f} % over {len(rows)} scenarios.", "",
             "| # | trip | dom | dur s | dist m | drift % | pure % | AT m | CT m | speed err m | creep m | stop s | v ratio | turn deg | seed hdg err | scale eng x ekf = eff | main cause |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(sel, key=lambda x: -x["map"]):
        lines.append(
            f"| {x['id']} | {x['trip']} | {x['domain']} | {x['dur']:.0f} | {x['dist']:.0f} | {x['map']:.1f} | {x['pure']:.1f} | "
            f"{x['final_at_m']:.1f} | {x['final_ct_m']:.1f} | {x['speed_err_m']:.1f} | {x['creep_m']:.1f} | {x['stop_s']:.0f} | "
            f"{x['speed_ratio']:.3f} | {x['turn_deg']:.0f} | {x['hdg_seed_err']:.1f} | "
            f"{x['scale_engine']:.3f} x {x['scale_ekf']:.3f} = {x['scale_eff']:.3f} | {verdict(x)} |")
    eff = np.array([x["scale_eff"] for x in rows], float)
    eng = np.array([x["scale_engine"] for x in rows], float)
    lines += ["", "## Double speed-scale check (T9)",
              f"Across all {len(rows)} scenarios: median engine scale {np.nanmedian(eng):.3f}, "
              f"median effective (engine x ekf) {np.nanmedian(eff):.3f}. "
              "If effective differs from engine by more than ~2 %, the correction is being applied twice.",
              "", "Plain-text maths: speed_err_m = sum(v_est - v_can) * 0.1 ; creep_m = sum(v_est where v_can < 0.3) * 0.1"]
    with open(os.path.join(out, "autopsy.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for x in sel:
        r = x["_r"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        g, m, p = np.asarray(r["gt_pts"]), np.asarray(r["map_pts"]), np.asarray(r["pure_pts"])
        ax[0].plot(g[:, 0], g[:, 1], "k-", label="GNSS truth")
        ax[0].plot(m[:, 0], m[:, 1], "c-", label="map DR")
        ax[0].plot(p[:, 0], p[:, 1], "m--", label="pure DR", alpha=0.6)
        ax[0].axis("equal"); ax[0].legend(); ax[0].set_title(f"#{x['id']} {x['trip']} {x['map']:.1f} %")
        t = np.asarray(r["time_rel_s"])
        ax[1].plot(t, r["gt_speeds"], "k-", label="CAN"); ax[1].plot(t, r["map_speeds"], "c-", label="DR")
        ax[1].set_title("speed m/s"); ax[1].legend()
        ax[2].plot(t, r["along_track_series"], label="along-track"); ax[2].plot(t, r["cross_track_series"], label="cross-track")
        ax[2].axhline(0, color="k", lw=0.5); ax[2].set_title("error m"); ax[2].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, f"scenario_{x['id']}.png"), dpi=110); plt.close(fig)
    print(f"[autopsy] wrote {out}/autopsy.md and {len(sel)} PNGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
