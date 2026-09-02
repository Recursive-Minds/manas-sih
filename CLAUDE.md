# Project Memory & Architecture Guide: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion (SIH)

## 1. Problem Statement & Context
- **Target Problem**: Smartphones lose GNSS/GPS signal in tunnels, underground parking, dense tree cover/forests, and urban canyons. Most Indian vehicles (two-wheelers, commercial trucks, older cars) lack factory-fitted INS and rely on dashboard/mount-attached phones without OBD-II or speedometer wiring. Standard navigation apps freeze or jump erratically during GNSS loss.
- **Solution**: A lightweight, edge-deployable software engine AND a mobile application that turns a standalone smartphone into an Intelligent Dead Reckoning (IDR) system with GNSS fusion, operating strictly on built-in phone IMU (accelerometer, gyroscope, magnetometer) without external vehicle wiring.
- **Key Required Capabilities**:
  1. *In-vehicle Alignment / Calibration Engine*: Auto-detects pitch/roll/yaw relative to vehicle driving direction across arbitrary mounts, dynamically re-detects on bump/remount.
  2. *AI-based Speed / Vibration Filter*: Estimates forward velocity directly from noisy IMU, filtering engine idle vibrations, potholes, mount jostling, and vehicle dynamics.
  3. *Map-Matching with Kinematic Constraints*: Binds trajectory to real road geometry using offline OpenStreetMap (OSM), applying non-holonomic constraints (no sideways/vertical motion).
  4. *GNSS+INS Fusion Engine*: AI-enhanced fusion (not fixed-parameter classical filtering alone) to reduce drift.
  5. *Seamless GNSS-Deficit Handler*: Millisecond-level transition between GNSS-aided and INS-only modes in both directions with zero visible jumps.
  6. *Real-time Navigation UI*: Smooth, uninterrupted vehicle tracking.
- **Sensor Agnostic / Dual-Deliverable Constraint**: The engine and ML models must NOT be locked solely to smartphone sensors. The core engine must accept external IMU data streams (e.g., FOG-grade sensors at up to ~200Hz). Core engine and mobile app are two separate deliverables sharing one core.

---

## 2. Benchmark Targets
- **Dead Reckoning Drift**: < 10% of total distance travelled during GNSS blackout (e.g., < 5m drift over 50m in < 1 min, OR < 100m drift over 1km at ~60 km/h in tunnel conditions).
- **Update Rate**:
  - Smartphone: 10 Hz position updates.
  - Edge Engine: Up to ~200 Hz with high-rate external IMU input.
- **Evaluation Criteria**: Every phase must be evaluated against this concrete metric, not merely "runs without crashing".

---

## 3. Prior Art & Adaptation Nuances
- **AI-IMU Dead-Reckoning (Brossard et al., arXiv:1904.06064)**: EKF + neural-network noise adaptation for wheeled vehicles (~1.1% error on KITTI). *Limitation*: KITTI uses research-grade IMU rigidly bolted and pre-aligned; does not solve arbitrary loose smartphone mounts or phone noise profiles.
- **RoNIN (Herath et al., arXiv:2005.10063)**: Deep-learning IMU-to-velocity regression (ResNet/LSTM/TCN). *Limitation*: Designed for pedestrian walking, not vehicular dynamics / two-wheelers / Indian roads.
- **Our Task**: Integrate and adapt these ideas into consumer smartphone IMUs, moving vehicles (cars & two-wheelers), Indian road dynamics, packaged for real-time edge execution. Treat paper numbers as reference baselines to test against, not assumed outcomes.

---

## 4. Datasets & Ingestion Guidelines
- **Primary Training/Benchmark**: IO-VNBD (`onyekpeu/IO-VNBD`) — smartphone subset (10Hz IMU + 1Hz GPS) and vehicle-ECU subset (UK, Nigeria, France).
- **Field Data**: In-house real-world recordings on Indian roads (cars & two-wheelers) for potholes, mixed traffic, vibration, and lean dynamics.
- **Flexible Schema Rule**: Never hard-code column names or file structure assumptions specific to IO-VNBD. The data-loading layer must be schema-flexible to accommodate screening round datasets and custom logs.
- **Data Leakage Rule**: Train/Validation/Test splits MUST ALWAYS be partitioned strictly **by trip / sequence**, NEVER by row / random slice. Row-level splitting is considered a critical bug.

---

## 5. Architectural Principles (Non-Negotiable)
- **Strictly Modular Pipeline with Fixed Data Contracts**:
  ```
  IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition
  ```
- **Interface Decoupling**: Each stage (`Calibration`, `VelocityEstimator`, `FusionFilter`, `MapMatcher`, `GNSSHandoffPolicy`) must be an independent class behind an abstract interface.
- **Single Point of Assembly**: Concrete implementations are bound in exactly ONE configuration / assembly point. No scattered stage-selection logic.
- **Single-File Swappability**: Swapping any stage implementation must only touch the single file implementing that stage + one line in the configuration.
- **Standalone Core Engine**: The core library must have zero Android/UI dependencies so it can run as an edge C++/Python/Rust library and be embedded into the Android app.

---

## 6. Working Protocol & Rules of Engagement
1. **One narrow task at a time**: Never expand scope silently to adjacent modules unless explicitly requested.
2. **Real Data Verification**: Never declare a step done on code syntax or clean run alone; always show real data outputs (drift metric, plot, printout).
3. **Flag Suspicious Results**: Immediately flag suspiciously good metrics (often caused by data leakage or overfitting) and suggest causes.
4. **Never Fabricate Benchmarks**: Only quote numbers verified from real runs or direct paper citation.
5. **Clear Explanations & Single-Hypothesis Troubleshooting**: When an issue arises, explain in plain terms and propose ONE concrete testable hypothesis.
6. **No Unilateral Architectural Decisions**: Present 2–3 concrete options with trade-offs on non-trivial architectural or model choices.
7. **Flag Native/JNI & Performance Risks**: Explicitly call out on-device battery, thermal, and C++/JNI integration constraints.
8. **Definition of Done**: A specific, testable claim, inspectable metric, or real-data plot.

---

---

## 7. Actual Verified Current State (Phases 1-3 Built)

The pipeline is implemented through Phase 3 and adheres strictly to the contract:
`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`

- **Calibration** (`sih/calibration/mount.py`): 3D gravity leveling + centripetal-correlation yaw-axis lock with polarity detection.
- **AI Velocity Estimator** (`sih/models/tcn_attention.py`, `sih/velocity/ai_estimator.py`): Multi-scale dilated TCN (dilations 1/2/4/8/16) + 4-head self-attention over a 100-step (10s) rolling window of 8 input channels, with a dual head predicting forward speed and log-variance, trained with speed-stratified importance weighting.
- **Fusion Filter** (`sih/fusion/es_ekf.py`): 15-state error-state EKF (position, velocity, attitude, accel bias, gyro bias) with non-holonomic constraints, ZUPT/ZARU, and an online Doppler-vs-AI speed scale factor (`s_v`) adapting during GNSS availability and freezing during blackout.
- **Evaluation Harness** (`sih/eval/metrics.py`, `sih/eval/benchmark.py`): Position RMSE, max error, drift %, along/cross-track decomposition, simulated blackout injection.

---

## 8. Current Benchmark Results (Real Measured Ground Truth)

| Trip | Scenario | Pipeline | Drift % | Final Pos Error | Result |
|---|---|---|---|---|---|
| **S-S1** | 60s blackout @ 300s ($814.1\text{ m}$) | Phase 3 (ES-EKF + AI) | **8.69%** | $70.77\text{ m}$ | **PASSED (< 10%)** |
| **S-S1** | 30s blackout @ 120s ($510.3\text{ m}$) | Phase 3 (ES-EKF + AI) | **13.72%** | $70.02\text{ m}$ | **FAILED** |
| **S-S2 (Unseen)** | 30s blackout @ 120s ($99.4\text{ m}$) | Phase 3 (ES-EKF + AI) | **25.59%** | $25.44\text{ m}$ | **FAILED** |

*Note: Naive Baseline ($229\% - 2145\%$) and Phase 2 Classical EKF ($94\% - 105\%$) verify that AI velocity integration provides massive ($8\times - 83\times$) error reduction, but consistency and generalization across regimes require resolution before map matching.*

---

## 9. Prioritized Investigation Tasks (Strictly Before Phase 4 Map Matching)

1. **Diagnose the S-S1 Anomaly**: The 30s@120s blackout (13.72% drift) performed worse than 60s@300s (8.69%) despite being shorter. Plot calibration convergence and online `s_v` scale factor convergence over the first 120s. Test the hypothesis of insufficient pre-blackout convergence time with plots before implementing fixes.
2. **Run Randomized Blackout Sweep**: Inject 20–30 varied blackouts per trip across S-S1 and S-S2; report drift % distribution (mean, median, 90th percentile, worst-case).
3. **Wire Predicted Uncertainty into EKF**: Consume `log(sigma^2)` from the TCN velocity model into the EKF velocity measurement covariance $R_v$.
4. **Break Down S-S2 Generalization Gap by Regime**: Segment unseen trip error by speed band and turn density.
5. **Sanity-Check S-S2 Baseline (2145%)**: Verify coordinate frames, units, and data parsing on S-S2.
6. **Re-Verify Trip-Level Data Split**: Formally demonstrate zero row/window overlap between train and validation sequences.
7. **Log Scale-Factor (`s_v`) Clipping Frequency**: Measure how often `s_v` hits the $[0.7, 1.6]$ bounds to check for systematic model bias.

