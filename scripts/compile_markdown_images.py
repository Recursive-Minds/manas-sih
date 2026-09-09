"""
compile_markdown_images.py
--------------------------
Compiles all high-resolution benchmark plots, multi-tier galleries, and trajectory
spotlight maps directly into Markdown documents as self-contained base64 data URIs.
"""

import base64
import os
import re

def b64_img(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def update_system_implementation_record():
    target_path = "docs/SYSTEM_IMPLEMENTATION_AND_TECHNIQUES_RECORD.md"
    if not os.path.exists(target_path):
        print(f"Target not found: {target_path}")
        return

    with open(target_path, "r", encoding="utf-8") as f:
        doc = f.read()

    print("Encoding images to base64...")
    moe_b64 = b64_img("artifacts/moe_training_curves.png")
    drift_b64 = b64_img("artifacts/phase4_unseen_sm_drift_comparison_chart.png")
    gallery_b64 = b64_img("artifacts/unseen_sm_all_tiers_gallery.png")
    sc30_b64 = b64_img("artifacts/map_scenario_30_highway_off_ramp_fork_split.png")
    sc02_b64 = b64_img("artifacts/map_scenario_02_90_degree_sharp_highway_turn.png")
    sc17_b64 = b64_img("artifacts/map_scenario_17_ultra_precision_highway_outage.png")
    sc10_b64 = b64_img("artifacts/map_scenario_10_high_speed_curve_outage.png")
    sc14_b64 = b64_img("artifacts/map_scenario_14_urban_chicane_navigation.png")
    sc31_b64 = b64_img("artifacts/map_scenario_31_acute_highway_branch_fork.png")

    # 1. Insert MoE training curves into Section 6.3 if not already present
    if "Bayesian MoE Dual-Expert Training Dynamics" not in doc:
        target_6_3 = "* **Checkpoint Metrics** (`models/checkpoints/best_moe_velocity_model.pt`): Validation RMSE **3.57 m/s**, scale ratio **1.039**."
        repl_6_3 = target_6_3 + f"""

<p align="center">
  <img src="data:image/png;base64,{moe_b64}" width="850" alt="Bayesian MoE Dual-Expert Training Dynamics" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>"""
        if target_6_3 in doc:
            doc = doc.replace(target_6_3, repl_6_3)
            print("Embedded MoE curves into Section 6.3.")
        else:
            print("Warning: target_6_3 string not found.")

    # 2. Insert Drift Distribution & Gallery into Section 9 if not already present
    if "35-Scenario Drift Distribution Comparison Chart" not in doc:
        target_sec9 = "## 9. Full 35-Scenario Benchmark Performance Record\n\nEvaluated on the held-out Part 3 partition across all 3 real-world driving sequences:"
        repl_sec9 = f"""## 9. Full 35-Scenario Benchmark Performance Record

### 9.1 Drift Distribution on Unseen Test Sequences

<p align="center">
  <img src="data:image/png;base64,{drift_b64}" width="850" alt="35-Scenario Drift Distribution Comparison Chart" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

### 9.2 Master Trajectory Visualizations: All-Tiers Multi-Domain Gallery

<p align="center">
  <img src="data:image/png;base64,{gallery_b64}" width="1050" alt="Master 9-Panel Trajectory Gallery" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

### 9.3 Scenario-by-Scenario Evaluation Table
Evaluated on the held-out Part 3 partition across all 3 real-world driving sequences:"""
        if target_sec9 in doc:
            doc = doc.replace(target_sec9, repl_sec9)
            print("Embedded Drift Chart and Gallery into Section 9.")
        else:
            print("Warning: target_sec9 string not found.")

    # 3. Insert Trajectory Spotlights into Section 11 if not already present
    if "### 11.4 Key Scenario Trajectory Spotlights" not in doc:
        target_sec11_end = "* **Sub-30% Consistency Rate**: **88.6% (31 / 35 scenarios)**."
        repl_sec11_spotlights = target_sec11_end + f"""

### 11.4 Key Scenario Trajectory Spotlights

#### Scenario #30: Highway Off-Ramp Fork Split (403m Outage)
* **Pure 6-Axis Baseline**: Diverged to **88.77% drift** (Red Dotted Line).
* **Phase 4 Map-Matched EKF**: Snapped cleanly to the exiting branch corridor, achieving **1.42% drift (5.7m error)** (Blue Solid Line).

<p align="center">
  <img src="data:image/png;base64,{sc30_b64}" width="750" alt="Scenario 30 Map" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

#### Scenario #02: 90-Degree Sharp Highway Turn (401m Outage)
* **Pure 6-Axis Baseline**: Experienced severe gyro scale loss, drifting to **32.40% error**.
* **Phase 4 Map-Matched EKF**: Topological successor gating tracked the sharp 90-degree right turn, achieving **3.73% drift**.

<p align="center">
  <img src="data:image/png;base64,{sc02_b64}" width="750" alt="Scenario 02 Map" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

#### Scenario #17: Ultra-Precision Highway Outage (555m Outage)
* **Pure 6-Axis Baseline**: Drifted by 49.71% over half a kilometer.
* **Phase 4 Map-Matched EKF**: Perfect corridor adherence yielding **0.00% endpoint drift (4.2m along-track error over 555m)**.

<p align="center">
  <img src="data:image/png;base64,{sc17_b64}" width="750" alt="Scenario 17 Map" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

#### Scenario #10: High-Speed Curve Outage (544m Outage)
* **Pure 6-Axis Baseline**: High-speed highway turn with centripetal force.
* **Phase 4 Map-Matched EKF**: Curvature kinematics governor bounded velocity, tracking the arc with **11.30% drift**.

<p align="center">
  <img src="data:image/png;base64,{sc10_b64}" width="750" alt="Scenario 10 Map" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

#### Scenario #14: Urban Chicane Navigation (968m Outage)
* **Pure 6-Axis Baseline**: Navigating repeated serpentine curves over nearly 1 kilometer.
* **Phase 4 Map-Matched EKF**: Maintained lane-level ribbon attachment across all chicanes, achieving **1.45% drift (14.0m error over 968m)**.

<p align="center">
  <img src="data:image/png;base64,{sc14_b64}" width="750" alt="Scenario 14 Map" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>

#### Scenario #31: Acute Highway Branch Fork (350m Outage)
* **Pure 6-Axis Baseline**: Acute divergence angle caused unguided EKF to bifurcate off-road.
* **Phase 4 Map-Matched EKF**: Directed successor transition prior correctly identified the route branch, achieving **9.27% drift**.

<p align="center">
  <img src="data:image/png;base64,{sc31_b64}" width="750" alt="Scenario 31 Map" style="max-width:100%; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15);" />
</p>"""
        if target_sec11_end in doc:
            doc = doc.replace(target_sec11_end, repl_sec11_spotlights)
            print("Embedded Spotlights into Section 11.")
        else:
            print("Warning: target_sec11_end string not found.")

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Successfully compiled images into {target_path} (File size: {os.path.getsize(target_path) / 1024:.1f} KB)")


def update_final_judge_report():
    target_path = "FINAL_JUDGE_EVALUATION_REPORT.md"
    if not os.path.exists(target_path):
        print(f"Target not found: {target_path}")
        return

    with open(target_path, "r", encoding="utf-8") as f:
        doc = f.read()

    drift_b64 = b64_img("artifacts/phase4_unseen_sm_drift_comparison_chart.png")
    gallery_b64 = b64_img("artifacts/unseen_sm_all_tiers_gallery.png")
    sc30_b64 = b64_img("artifacts/map_scenario_30_highway_off_ramp_fork_split.png")
    sc02_b64 = b64_img("artifacts/map_scenario_02_90_degree_sharp_highway_turn.png")
    sc17_b64 = b64_img("artifacts/map_scenario_17_ultra_precision_highway_outage.png")

    doc = doc.replace('src="artifacts/phase4_unseen_sm_drift_comparison_chart.png"', f'src="data:image/png;base64,{drift_b64}"')
    doc = doc.replace('src="artifacts/unseen_sm_all_tiers_gallery.png"', f'src="data:image/png;base64,{gallery_b64}"')
    doc = doc.replace('src="artifacts/map_scenario_30_highway_off_ramp_fork_split.png"', f'src="data:image/png;base64,{sc30_b64}"')
    doc = doc.replace('src="artifacts/map_scenario_02_90_degree_sharp_highway_turn.png"', f'src="data:image/png;base64,{sc02_b64}"')
    doc = doc.replace('src="artifacts/map_scenario_17_ultra_precision_highway_outage.png"', f'src="data:image/png;base64,{sc17_b64}"')

def verify_documents():
    docs = [
        "docs/SYSTEM_IMPLEMENTATION_AND_TECHNIQUES_RECORD.md",
        "FINAL_JUDGE_EVALUATION_REPORT.md"
    ]
    for d in docs:
        if not os.path.exists(d):
            continue
        with open(d, "r", encoding="utf-8") as f:
            content = f.read()
        imgs = re.findall(r'<img [^>]+>', content)
        b64_imgs = [img for img in imgs if "data:image/png;base64," in img]
        print(f"[{d}] Total images: {len(imgs)} | Compiled base64 images: {len(b64_imgs)}")
        for idx, img in enumerate(imgs):
            alt_m = re.search(r'alt="([^"]+)"', img)
            alt = alt_m.group(1) if alt_m else "no-alt"
            is_b64 = "data:image/png;base64," in img
            print(f"   Image {idx+1}: {alt} (Base64: {is_b64})")


if __name__ == "__main__":
    update_system_implementation_record()
    update_final_judge_report()
    verify_documents()

