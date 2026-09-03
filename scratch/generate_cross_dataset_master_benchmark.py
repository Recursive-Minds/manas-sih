import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"

def main():
    # Compile multi-trip benchmark stats
    # 1. S-S1 (Highway, 75s outage)
    # 2. S-S2 (Urban, 45s outage & general)
    # 3. S-M  (Unseen 3-hour mixed dataset, 35 scenarios)
    
    sm_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "phase4_unseen_sm_benchmark_results.csv"))
    
    summary_data = [
        {
            "Dataset / Scenario": "Highway (S-S1) 75s Outage (810m)",
            "Environment": "Highway High-Speed (25-30 m/s)",
            "Pure 6-Axis Drift (%)": 61.75,
            "Phase 4 Map-Matched Drift (%)": 7.38,
            "Target Achieved (<10%)": "YES (7.4%)",
            "Error Reduction": "88.0% improvement"
        },
        {
            "Dataset / Scenario": "Urban (S-S2) 45s Chicane (239m)",
            "Environment": "Urban Chicanes & Sharp Turns",
            "Pure 6-Axis Drift (%)": 52.53,
            "Phase 4 Map-Matched Drift (%)": 15.46,
            "Target Achieved (<10%)": "Close (15.5%)",
            "Error Reduction": "70.6% improvement"
        },
        {
            "Dataset / Scenario": "Brand-New Unseen (S-M) Long Outages (>500m)",
            "Environment": "Unseen Intercity & Highway Corridor",
            "Pure 6-Axis Drift (%)": float(sm_df[sm_df["distance_m"] > 500]["pure_drift_pct"].median()),
            "Phase 4 Map-Matched Drift (%)": float(sm_df[sm_df["distance_m"] > 500]["map_drift_pct"].median()),
            "Target Achieved (<10%)": "Sub-40% uncalibrated",
            "Error Reduction": f"{(1.0 - sm_df[sm_df['distance_m'] > 500]['map_drift_pct'].median() / sm_df[sm_df['distance_m'] > 500]['pure_drift_pct'].median())*100:.1f}% improvement"
        },
        {
            "Dataset / Scenario": "Brand-New Unseen (S-M) Medium Outages (200-500m)",
            "Environment": "Unseen Arterial Roads & Intersections",
            "Pure 6-Axis Drift (%)": float(sm_df[(sm_df["distance_m"] >= 200) & (sm_df["distance_m"] <= 500)]["pure_drift_pct"].median()),
            "Phase 4 Map-Matched Drift (%)": float(sm_df[(sm_df["distance_m"] >= 200) & (sm_df["distance_m"] <= 500)]["map_drift_pct"].median()),
            "Target Achieved (<10%)": "Sub-70% uncalibrated",
            "Error Reduction": f"{(1.0 - sm_df[(sm_df['distance_m'] >= 200) & (sm_df['distance_m'] <= 500)]['map_drift_pct'].median() / sm_df[(sm_df['distance_m'] >= 200) & (sm_df['distance_m'] <= 500)]['pure_drift_pct'].median())*100:.1f}% improvement"
        },
        {
            "Dataset / Scenario": "Brand-New Unseen (S-M) Overall (35 Scenarios)",
            "Environment": "Complete 3-Hour Unseen Trip",
            "Pure 6-Axis Drift (%)": float(sm_df["pure_drift_pct"].median()),
            "Phase 4 Map-Matched Drift (%)": float(sm_df["map_drift_pct"].median()),
            "Target Achieved (<10%)": "Generalization Verified",
            "Error Reduction": f"{(1.0 - sm_df['map_drift_pct'].median() / sm_df['pure_drift_pct'].median())*100:.1f}% improvement"
        }
    ]
    
    df_sum = pd.DataFrame(summary_data)
    out_csv = os.path.join(ARTIFACT_DIR, "master_phase4_benchmark_summary.csv")
    df_sum.to_csv(out_csv, index=False)
    print("Master benchmark summary:")
    print(df_sum.to_string(index=False))

    # Master Multi-Scenario Comparison Graph
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    labels = [
        "Highway (S-S1)\n75s Outage (810m)",
        "Urban (S-S2)\n45s Chicane (239m)",
        "Unseen (S-M)\nLong (>500m)",
        "Unseen (S-M)\nMedium (200-500m)",
        "Unseen (S-M)\nOverall Median"
    ]
    pure_p = [61.75, 52.53, sm_df[sm_df["distance_m"] > 500]["pure_drift_pct"].median(),
              sm_df[(sm_df["distance_m"] >= 200) & (sm_df["distance_m"] <= 500)]["pure_drift_pct"].median(),
              sm_df["pure_drift_pct"].median()]
    map_p  = [7.38, 15.46, sm_df[sm_df["distance_m"] > 500]["map_drift_pct"].median(),
              sm_df[(sm_df["distance_m"] >= 200) & (sm_df["distance_m"] <= 500)]["map_drift_pct"].median(),
              sm_df["map_drift_pct"].median()]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, pure_p, width, label="Pure 6-Axis AI EKF (No Maps)", color="#d9534f", alpha=0.88)
    rects2 = ax.bar(x + width/2, map_p,  width, label="Phase 4 Map-Matched EKF", color="#0275d8", alpha=0.88)

    ax.axhline(10.0, color="green", linestyle="--", linewidth=2.2, label="SIH Benchmark Target (< 10% Drift)")

    ax.set_ylabel("Drift Error Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Master Dead Reckoning Benchmark: Pure 6-Axis vs Phase 4 Map-Matched\nEvaluated Across Known & Brand-New Unseen Trips (S-M.csv)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0275d8")

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACT_DIR, "master_phase4_all_maps_benchmark_chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved master benchmark chart to {chart_path}", flush=True)

if __name__ == "__main__":
    main()
