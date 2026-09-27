"""
scripts/generate_ppt_trajectory_plots.py

Generates 1600px wide trajectory plots on white background for scenarios #06, #18, #30, #33
from canonical seed 541098 using the production system.
Saves to ppt_pack/images/ and writes ppt_pack/images/README.txt.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sih.round1.config import Round1Config, set_active_config
import benchmarks.run_final_benchmark as bm

TARGET_SCENARIOS = [6, 18, 30, 33]
SEED = 541098

def main():
    out_dir = os.path.join(ROOT, "ppt_pack", "images")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading precomputed benchmark data...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_active_config(Round1Config(name="precompute"))
    pre = bm.load_precomputed_benchmark_data(device, map_source="osm")

    print(f"Evaluating canonical seed {SEED} with production profile...")
    prod_cfg = Round1Config.from_json("config/round1/production.json")
    set_active_config(prod_cfg)

    df_seed, detailed, metrics = bm.evaluate_seed_scenarios(SEED, pre, map_source="osm")
    set_active_config(None)

    sc_map = {r["scenario_id"]: r for r in detailed if r["scenario_id"] in TARGET_SCENARIOS}
    readme_lines = [
        "SIH 26168 - Production Scenario Trajectory Images (Canonical Dev Seed 541098)\n",
        "Generated with current production system (round1_interval_lam0.5_s42.pt, T7 speed calib blend, T8 junction snapping)\n",
        "=" * 80 + "\n\n"
    ]

    for sc_id in TARGET_SCENARIOS:
        if sc_id not in sc_map:
            print(f"Warning: Scenario #{sc_id} not found in detailed results!")
            continue

        row = sc_map[sc_id]
        trip_id = row["trip_id"]
        domain = row["domain"]
        dur = float(row["duration_s"])
        dist = float(row["dist_m"])
        final_err = float(row["map_err_m"])
        final_drift = float(row["map_drift_pct"])
        pure_err = float(row["pure_err_m"])
        pure_drift = float(row["pure_drift_pct"])

        gt_pts = row["gt_pts"]
        pure_pts = row["pure_pts"]
        map_pts = row["map_pts"]
        rnet = row["road_net"]

        p_start = gt_pts[0]
        max_r = dist + 150.0

        # Create 1600px wide figure: 16 inches at 100 dpi = 1600 px wide
        fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Draw road network corridor segments
        drawn_road = False
        if rnet is not None:
            for s in rnet.segments:
                d = min(np.linalg.norm(s.start_enu_m - p_start), np.linalg.norm(s.end_enu_m - p_start))
                if d < max_r:
                    lbl = "OSM Road Corridor" if not drawn_road else None
                    ax.plot([s.start_enu_m[0], s.end_enu_m[0]], [s.start_enu_m[1], s.end_enu_m[1]],
                            color="#e2e8f0", linewidth=14, solid_capstyle="round", zorder=1, label=lbl)
                    ax.plot([s.start_enu_m[0], s.end_enu_m[0]], [s.start_enu_m[1], s.end_enu_m[1]],
                            color="#f8fafc", linewidth=8, solid_capstyle="round", zorder=2)
                    drawn_road = True

        # Plot trajectories
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], "k--", linewidth=3.2, alpha=0.9, label="Ground Truth (GNSS)", zorder=3)
        ax.plot(pure_pts[:, 0], pure_pts[:, 1], color="#ef4444", linestyle=":", linewidth=3.2,
                label=f"Pure DR (No Map) - {pure_err:.1f}m ({pure_drift:.2f}% drift)", zorder=4)
        ax.plot(map_pts[:, 0], map_pts[:, 1], color="#0284c7", linestyle="-", linewidth=3.8,
                label=f"Smart IDR (Production) - {final_err:.1f}m ({final_drift:.2f}% drift)", zorder=5)

        # Markers
        ax.plot(p_start[0], p_start[1], "ko", markersize=12, zorder=6, label="Blackout Entry")
        ax.plot(gt_pts[-1, 0], gt_pts[-1, 1], "kx", markersize=14, markeredgewidth=3.5, zorder=6, label="Ground Truth Exit")
        ax.plot(pure_pts[-1, 0], pure_pts[-1, 1], "o", color="#ef4444", markeredgecolor="black", markersize=11, zorder=6)
        ax.plot(map_pts[-1, 0], map_pts[-1, 1], "s", color="#0284c7", markeredgecolor="white", markersize=11, zorder=6)

        # Labels, title, styling
        title_str = (f"Scenario #{sc_id:02d}: {domain} Blackout Outage ({trip_id}, {dur:.0f}s)\n"
                     f"Distance: {dist:.1f}m | Smart IDR Drift: {final_drift:.2f}% ({final_err:.1f}m error) vs Pure DR: {pure_drift:.2f}% ({pure_err:.1f}m)")
        ax.set_title(title_str, fontsize=15, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xlabel("East Coordinate (meters)", fontsize=13, fontweight="bold", labelpad=8)
        ax.set_ylabel("North Coordinate (meters)", fontsize=13, fontweight="bold", labelpad=8)
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.95, fontsize=11, edgecolor="#cbd5e1")

        plt.tight_layout()
        img_name = f"trajectory_scenario_{sc_id:02d}.png"
        img_path = os.path.join(out_dir, img_name)
        plt.savefig(img_path, dpi=100, facecolor="white", edgecolor="none")
        plt.close()
        print(f"Rendered {img_path} ({sc_id:02d}: drift={final_drift:.2f}%)")

        readme_lines.append(
            f"- File: {img_name}\n"
            f"  Scenario: #{sc_id:02d}\n"
            f"  Seed: {SEED}\n"
            f"  Trip: {trip_id}\n"
            f"  Domain: {domain}\n"
            f"  Blackout Duration: {dur:.1f} s\n"
            f"  Ground Truth Distance: {dist:.1f} m\n"
            f"  Pure DR Drift: {pure_drift:.2f}% ({pure_err:.2f} m)\n"
            f"  Smart IDR Drift: {final_drift:.2f}% ({final_err:.2f} m)\n"
            f"  Resolution: 1600 x 1000 px, 100 DPI, white background\n\n"
        )

    readme_path = os.path.join(out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(readme_lines)
    print(f"Wrote {readme_path}")

if __name__ == "__main__":
    main()
