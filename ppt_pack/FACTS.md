# SIH PS 26168: Definitive Presentation Facts & Provable Metrics Sheet

Every number in this sheet is directly extracted from verified execution runs, benchmark logs, and binary bundle assets. All mathematical expressions use clean plain-text formatting.

---

## 1. Primary Benchmark Scorecards

### Canonical Development Seeds (6 Seeds x 40 Scenarios = 236 Evaluated Runs)
*Source: `ppt_pack/data/summary_6seed.json` & `results/final/six_seed/production_scenarios.csv` (config: `config/round1/production.json`)*
- **Overall Median Drift**: **12.03%**
- **Cross-Seed Mean of Seed Medians**: **12.32% +- 1.13%** (Std: 1.13%)
- **90th Percentile (P90) Drift**: **37.30%**
- **Share < 10% Drift**: **42.37%** (100 / 236 runs)
- **Share < 30% Drift**: **83.90%** (198 / 236 runs)
- **Beats Pure Dead-Reckoning Rate**: **82.20%** (194 / 236 runs)
- **Pure Dead-Reckoning Median Drift**: **22.46%**
- **Per-Scenario Medians (40 Scenarios)**:
  - Scenarios improved vs Pure DR: **38 / 40 (95.0%)**
  - Scenarios with median drift < 10%: **15 / 40 (37.5%)**
  - Scenarios with median drift < 30%: **38 / 40 (95.0%)**

### Speed Regime Scorecard vs Problem Statement Targets
*Source: `ppt_pack/data/summary_6seed.json` (Computed Status Rule: Met if value <= target, Near if value <= 1.5x target, else Not met)*
- **Crawl (< 20 km/h)**: Median Error = **15.82 m** (Target: < 10.0 m, n = 43) -> **Status: Not met**
- **City (20-50 km/h)**: Median Drift = **12.16%** (Target: < 15.0%, n = 152) -> **Status: Met**
- **Highway (> 50 km/h)**: Median Drift = **13.98%** (Target: < 10.0%, n = 41) -> **Status: Near**
- **All Scenarios**: Median Drift = **12.03%** (Target: < 10.0%, n = 236) -> **Status: Near**

### Strictly Held-Out Seeds (3 Seeds x 40 Scenarios = 120 Evaluated Runs)
*Source: `FINAL_NUMBERS_FOR_PPT.md` & `results/round1/heldout_r2_blend180/summary.json` (Independent confirmation, never used during tuning)*
- **Median Drift**: **11.15%**
- **Cross-Seed Mean of Seed Medians**: **10.71% +- 1.17%** (Std: 1.17%)
- **90th Percentile (P90) Drift**: **32.91%** (error tail tightened by 4.34 pp from 37.25%)
- **Share < 10% Drift**: **48.33%** (58 / 120 runs)
- **Share < 30% Drift**: **85.83%** (103 / 120 runs)
- **Unseen Trips Median (S-S3a, S-S4)**: **9.66%** (n = 60, breaks the 10% barrier)
- **Beats Pure Dead-Reckoning Rate**: **86.67%** (104 / 120 runs)
- **Pre-Round-1 Baseline Comparison**:
  - Baseline Mean: 11.13% +- 1.50% -> Production Mean: 10.71% +- 1.17% (-0.42 pp, std down 22%)
  - Baseline P90: 37.25% -> Production P90: 32.91% (-4.34 pp)
  - Baseline Share < 10%: 42.50% -> Production Share < 10%: 48.33% (+5.83 pp)
  - Baseline Unseen Trips: 11.89% -> Production Unseen Trips: 9.66% (-2.23 pp)

---

## 2. Progressive Architecture Stage Drift (S-S1 60s Outage & Multi-Seed)
*Source: `ppt_pack/data/stages.json`*
- **(a) Naive Double Integration**: **424.13% drift** (3,452.9 m error over 814.0 m drive on S-S1; historical open-loop benchmark, `benchmarks/run_phase2_es_ekf.py`, commit `519a202`).
- **(b) ES-EKF + NHC**: **178.79% drift** (1,455.57 m error over 814.0 m drive on S-S1; historical 15-state ES-EKF with non-holonomic tire velocity constraints, `benchmarks/run_phase2_es_ekf.py`, commit `519a202`).
- **(c) AI Velocity Estimator + ES-EKF (No Map)**: **22.46% median drift** (current production dual-brain MoE speed estimation with 15-state ES-EKF, open-loop without map).
- **(d) Full Smart IDR Pipeline**: **12.03% median drift** (current complete system: MoE speed estimator, T7 per-band speed calibration blend, T8 post-turn junction snapping, and topological OSM road governor).

---

## 3. Key Scenario Trajectory Spotlights (Canonical Dev Seed 541098)
*Source: `ppt_pack/images/README.txt` & `results/final/six_seed/production_scenarios.csv`*
- **Scenario #06 (Highway Cruising)**: Trip `S-M`, 30.0s blackout, 427.1 m distance.
  - Smart IDR: **1.88% drift** (8.04 m final error) vs Pure DR: 7.10% (30.31 m error).
- **Scenario #18 (Dense Urban Grid)**: Trip `S-S1`, 45.0s blackout, 98.9 m distance.
  - Smart IDR: **6.85% drift** (6.77 m final error) vs Pure DR: 58.13% (57.49 m error).
- **Scenario #30 (Complex Mixed Route)**: Trip `S-S3a`, 60.0s blackout, 244.2 m distance.
  - Smart IDR: **9.55% drift** (23.31 m final error) vs Pure DR: 6.43% (15.71 m error).
- **Scenario #33 (Arterial Corridor)**: Trip `S-S4`, 60.0s blackout, 443.5 m distance.
  - Smart IDR: **4.46% drift** (19.76 m final error) vs Pure DR: 11.50% (51.02 m error).

---

## 4. On-Device Implementation Facts (Samsung Galaxy F12)
*Source: `ppt_pack/data/ondevice.json`*
- **Test Device**: Samsung Galaxy F12 (`SM-F127G`), Exynos 850 (8-core ARM Cortex-A55 @ 2.0 GHz), Android 13 (API 33, One UI Core 5.1).
- **Package Footprint**:
  - Full Standalone APK (`app-ondevice-debug.apk`): **50.12 MB** (includes Chaquopy Python 3.8 runtime, PyTorch CPU wheels, OSMDroid tile engine, assets).
  - TFLite Model (`moe_velocity_model.tflite`): **2.50 MB** (2,620,284 bytes).
  - ONNX Model (`moe_velocity_model.onnx`): **2.48 MB** (2,597,048 bytes).
  - PyTorch TorchScript Model (`moe_velocity_model.torchscript.pt`): **2.66 MB** (2,787,310 bytes).
- **Model Export Numerical Parity (vs PyTorch CPU over 1,000 Real Feature Windows)**:
  - TFLite max abs difference: **4.77e-6 m/s** (4.77 um/s, pass < 1.0e-5 m/s threshold).
  - ONNX max abs difference: **5.72e-6 m/s** (5.72 um/s, pass < 1.0e-5 m/s threshold).
- **Per-100ms Batch Execution Latency**:
  - Total pipeline per-batch latency: **5.06 ms mean** (P50: 4.92 ms, P90: 6.30 ms, P95: 7.18 ms, Max: 8.94 ms).
  - Processing budget headroom: **94.94% headroom** under 100 ms update window.
  - Subsystem split: Feature extraction: 0.32 ms | MoE Model: 3.37 ms | Speed smoother: 0.04 ms | EKF + Map: 1.29 ms.
- **Hardware Resources**:
  - Memory Footprint (PSS): **320.3 MB** (`dumpsys meminfo`).
  - Active CPU Load: **9.2% normalized across 8 cores** (73.6% single-core during 2.0x demo replay; ~1.5% at idle).
- **Drawer Setup Acceleration**:
  - Cold setup: **2.8 s** (warmup via cached pre-roll).
  - Instant snapshot restore: **0.4 s** (7.0x speedup).
- **On-Device Bundle Verification Table (Exact 0.000 m Parity vs Laptop)**:
  - Scenario #01: Trip S-M, Highway, 30.0s, 301.5m, Expected: 80.59m, Phone: 80.59m, Drift: 26.73%, Diff: 0.000m.
  - Scenario #22: Trip S-S3a, Mixed, 45.0s, 475.1m, Expected: 16.77m, Phone: 16.77m, Drift: 3.53%, Diff: 0.000m.
  - Scenario #23: Trip S-S3a, Mixed, 75.0s, 1128.4m, Expected: 67.76m, Phone: 67.76m, Drift: 6.00%, Diff: 0.000m.
  - Scenario #25: Trip S-S3a, Mixed, 45.0s, 614.3m, Expected: 77.30m, Phone: 77.30m, Drift: 12.58%, Diff: 0.000m.
  - Scenario #26: Trip S-S3a, Mixed, 75.0s, 892.8m, Expected: 122.80m, Phone: 122.80m, Drift: 13.75%, Diff: 0.000m.
  - Scenario #30: Trip S-S3a, Mixed, 60.0s, 244.2m, Expected: 7.04m, Phone: 7.04m, Drift: 2.88%, Diff: 0.000m.
- **Regression Test Suite**:
  - Total automated repository tests: **160 passed, 2 skipped, 0 failed** (162 total tests).

---

## 5. Strict "DO NOT CLAIM" Rules for Presentation
1. **DO NOT claim < 10% drift is achieved on all regimes**: The crawl regime (< 20 km/h) achieves 15.82 m median error against a < 10.0 m target (Status: "Not met").
2. **DO NOT claim all 40 scenarios are bundled in the phone APK**: Only 6 scenarios are packaged in ondevice assets due to APK size constraints. The remaining 34 can be fetched dynamically.
3. **DO NOT claim battery consumption is formally benchmarked**: Battery drain status is **PENDING** field driving validation.
4. **DO NOT claim live physical vehicle driving test in tunnels has been completed**: Real physical driving status is **PENDING**; results are evaluated on real-world vehicle CAN/IMU datasets (IO-VNBD).
5. **DO NOT claim mount calibration persists across app restarts**: Auto-calibration runs per session or reuses an in-memory session snapshot.
6. **DO NOT claim Barometer, Lean-aware NHC, NDK sensor capture, or INT8 are in production**: These are architectural designs, not implemented in the current production code.
7. **DO NOT claim the C++ engine runs the Android app**: The C++ engine is a standalone telematics reference prototype; the Android app runs via Chaquopy.
8. **DO NOT quote stale pre-round-1 numbers**: Never use 11.13% +- 1.50% as the production headline, 37/40, 17/40, or 40/40 test suite.
