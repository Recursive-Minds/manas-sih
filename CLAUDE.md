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
- **Strict Benchmark / Core Logic Separation**: The benchmark script (`benchmarks/run_final_benchmark.py`) must ONLY contain benchmark orchestration code (scenario selection, metric computation, plotting, report generation). ALL algorithmic logic — including GPS interpolation, heading seeding preparation, speed scaling, road network construction, and EKF configuration — MUST live in dedicated modules under `sih/` (e.g. `sih/fusion/`, `sih/data/`, `sih/map/`, `sih/calibration/`). The benchmark script calls into these modules; it never re-implements or inlines core logic. Any new algorithmic feature must be implemented in `sih/` first, then invoked from the benchmark. Violating this rule is a critical architectural bug.

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

## 7. Actual Verified Current State (Phases 1-6 Built & Passing)

The complete algorithmic pipeline is implemented through Phase 6 and adheres strictly to the contract:
`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`

- **Calibration (Phase 4)** (`sih/calibration/mount.py`): 3D gravity leveling (Rodrigues rotation) + dual-metric centripetal acceleration correlation (`|r_a| * E_a`) for yaw-axis selection with dynamic least-squares sign lock (`Cov(omega_z, psi_dot) / Var(omega_z)`).
- **Initial Heading Seeding (Phase 4)** (`sih/fusion/es_ekf.py`): Speed-regime 2-point GNSS displacement vector seeder with pre-blackout heading consistency gating (cross-checks against moving GNSS Doppler `v >= 2.0 m/s`, overriding if discrepancy > 50°) and decisive straight-line cruise innovation (`gain = 0.85`), bypassing phone cabin magnetic distortions of +28° to +76°.
- **AI Velocity Estimator (Phase 3)** (`sih/models/moe_fusion.py`, `sih/models/inference.py`, `sih/velocity/ai_estimator.py`): Dual-Brain Bayesian Mixture-of-Experts (`BayesianMoEFusion`) combining ResNet-1D micro-window (2.0s / 20 steps) + dilated TCN-Attention macro-window (6.0s / 60 steps) with GRU over 12 input features. Supervised by 10 Hz physical vehicle CAN-bus wheel speeds (`V-M.csv`, `V-S1.csv`, `V-S2.csv`), resolving the 9-second phone GPS stair-step optical illusion. Clean decoupled inference pipeline in `sih/models/inference.py`.
- **Causal Kinematic Speed Smoother (Phase 2/3)** (`sih/fusion/speed_smoother.py`): Physical acceleration slew rate limiting (`-5.0 m/s^2 <= a <= +3.5 m/s^2`) and causal EMA smoothing (`tau = 0.25s`) eliminating 89% of high-frequency speed variance without phase lag.
- **Fusion Filter (Phase 2 & 3)** (`sih/fusion/es_ekf.py`): 15-state error-state EKF on SO(3) quaternion manifold (position, velocity, attitude, accel bias, gyro bias).
- **Physical Hardening & Invariant Constraints**:
  - *Rate-Adaptive Closed-Loop NHC*: Enforces `v_lat = 0, v_up = 0` with dynamic covariance `R_lat(omega_z)` for tire slip during turns.
  - *Lorentzian Turn-Damped Gyro Bias*: Damps bias updates with `1.0 / (1.0 + (|omega_z| / omega_0)^2)` to prevent centripetal turn dynamics from corrupting gyro bias.
  - *Physical Rest ZUPT & ZARU*: Accel variance (`Var(a) < 0.04 m^2/s^4`) and gyro norm (`||omega|| < 0.05 rad/s`) clamp velocity to zero and freeze integration during stops.
  - *Low-Speed Crawl Clamping*: Enforces `v_fwd <= max(v_entry + 1.2 m/s, 3.5 m/s)` during crawl entries (`v_entry < 4.0 m/s`), preventing engine idle vibrations from simulating cruising.
  - *Pre-Blackout Dynamic Speed Scaling*: Adapts pavement vibration scale (`s_v = mean(v_GPS) / mean(v_AI)`) over the 20s prior to blackout.
  - *ZARU Highway Straight-Line Lock*: Freezes yaw gyro bias when `v > 15 m/s` and `|omega_z| < 0.005 rad/s` for > 2.0s, eliminating phantom highway curvature.
  - *Hybrid Speed Blending*: Blends accelerometer forward velocity integration with neural MoE speed using 3-8 Hz frequency vibration power.
- **Map-Matching & Gating (Phase 5)** (`sih/map/network.py`, `sih/map/matcher.py`, `sih/map/governor.py`): Spatial polyline indexing with turn-inflated Gaussian emission likelihood (`sigma_eff >= 45°`), curvature kinematics governor (`v <= sqrt(a_lat_max / kappa)`), branch multi-hypothesis fork gating (`diff_theta > 15 deg, L2 > 0.20 * L1`), expanded 110° successor turn gates (`sigma_h = 60°`), anti-boundary clamping watchdog suppressing junction stalls, and prompt corridor heading steering (`0.50 * diff_rad`).
- **Standalone 200 Hz C++ Core** (`engine/cpp/`): Zero-dependency modern C++ implementation compiled into `idr_core.dll` for dual-deliverable embedded telematics.

---

## 8. Master Benchmark Results (Empirical Single Source of Truth)

All benchmark scores, multi-seed statistical validations (6 random seeds x 40 scenarios = 240 evaluation runs), domain breakdowns, and trajectory maps are maintained exclusively in:
👉 [FINAL_JUDGE_EVALUATION_REPORT.md](file:///c:/Users/carpe/SIH/FINAL_JUDGE_EVALUATION_REPORT.md)

**Official SIH Benchmark Criteria**:
- **Grand Target**: Dead Reckoning Drift < 10% of total distance travelled during GNSS blackout (< 5m over 50m, or < 100m over 1km).
- **Tier 1 (Traffic Crawl, < 20 km/h, < 200m)**: Stopping drift arrested via Physical Rest ZUPT.
- **Tier 2 (City Maneuvers, 20-50 km/h, 200-500m)**: Heading drift < 10% through dynamic multi-source heading and road governing.
- **Tier 3 (Highway Cruising, > 50 km/h, > 500m-1.2km)**: Speed scale fidelity sum(v_hat)/sum(v_GT) approx 1.00 and high-speed gyro drift suppression.

*(See [FINAL_JUDGE_EVALUATION_REPORT.md](file:///c:/Users/carpe/SIH/FINAL_JUDGE_EVALUATION_REPORT.md) for current verified scorecards passing all SIH criteria).*

---

## 9. Active & Completed Phases for Final SIH Submission

1. **[COMPLETED] Phase 6: Seamless GNSS <-> INS Handoff State Machine**:
   - Production 6-state FSM (`sih/handoff/manager.py`): `INITIALIZING` -> `GNSS_HEALTHY` -> `GNSS_DEGRADED` -> `INS_DEAD_RECKONING` -> `REACQUISITION_VERIFY` -> `REACQUISITION_BLENDING`.
   - Chi-Square Normalized Innovation Squared (NIS) and multi-sample kinematic plausibility gating (`sih/handoff/integrity.py`).
   - C^2 cubic Hermite smoothstep zero-jump reconciliation (`sih/handoff/reconciliation.py`), verified on real sequence `S-M.csv` with **0.0000 m exit jump** and **100.0% parameter freeze** during portal multipath.
   - Comprehensive test suite in `tests/test_handoff.py` (7/7 passed, 40/40 repo-wide).

2. **[COMPLETED] Live Indian Road Vector Ingestion & Speed-Adaptive Predictive Corridor Caching Engine**:
   - Dynamic Overpass OSM road geometry client with fallback to local Indian GIS (PMGSY / Bhuvan) (`sih/map/osm_client.py`, `sih/map/local_gis.py`, `sih/map/hybrid_provider.py`).
   - Deterministic 0.05 degree (~5.5 km) spatial disk cache with LRU eviction and negative caching (`sih/map/cache.py`).
   - Speed-adaptive predictive lookahead (`R = clamp(v * 180s, 800m, 6000m)`) with asynchronous thread worker and atomic pointer swap (`sih/map/corridor_manager.py`).
   - Verified on Mumbai-Pune Expressway Bhatan Tunnel: 3,142 road segments ingested, 14.19 ms subsequent offline cache retrieval, and 0.42 ms P99 IMU loop latency during live background prefetching.
   - Comprehensive unit test suite in `tests/test_map_ingestion.py` (6/6 passed, 40/40 repo-wide).

3. **[COMPLETED] Phase 7: Mobile App Deployment Readiness & Edge Causal Runtime**:
   - Exported PyTorch Mobile TorchScript graph `models/exported/moe_velocity_model.torchscript.pt` (**2.66 MB**, 0.000000 m/s numerical parity, **2.68 ms latency** on CPU / 373 Hz throughput).
   - Exported 12-channel normalization vectors `models/exported/normalization_params.npz`.
   - Production streaming causal interface `sih/mobile/causal_stream.py` (`MobileDeadReckoningStream`) ingesting 10-50 Hz IMU and 1 Hz GNSS with zero lookahead.
   - Comprehensive unit test suite `tests/test_mobile_stream.py` (3/3 passed, 40/40 repo-wide).

4. **Phase 8: Final Presentation & Jury Demonstration**:
   - Standalone evaluation executable and interactive web dashboard (`FINAL_JUDGE_EVALUATION_REPORT.html`).
   - Slide deck highlighting 93.6% drift reduction, 0.0000m exit jump, and 2.66 MB edge model footprint.

---

## 10. The 3 Authoritative Documentation Files (Single Source of Truth)

To eliminate contradictory metrics across disparate files, all project documentation is strictly consolidated into **3 authoritative master files** at the root of the repository:

1. **`SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md`**: The living technical reference for mathematical formulations, dynamic coordinate frames, SO(3) leveling, Delta-v speed observer, dynamic heading fusion, repaired road governor, active parameters, edge C++ NDK engine, and complete codebase inventory.
   - **MANDATORY RULE**: MUST be updated ANYTIME parameters are tweaked or new features/algorithms are added.
2. **`PROBLEM_STATEMENT_AND_INITIAL_PLAN.md`**: The definitive record of the SIH 26168 Problem Statement, Indian road challenges, 3 operational tiers, initial 5-phase roadmap, key scientific discoveries (why neural heading failed, 9s GPS illusion), and the 20 physical failure modes.
3. **`FINAL_JUDGE_EVALUATION_REPORT.md` (and `.html`)**: The SINGLE SOURCE OF TRUTH for all empirical figures, 6-seed 240-scenario benchmark matrix, domain scorecards, error decompositions, and scenario plots.
   - **MANDATORY RULE**: MUST be re-generated whenever benchmarks are executed.
   - **NO DIVERGENT METRICS RULE**: Never hardcode or duplicate benchmark numbers into other markdown files. All other documents link directly to `FINAL_JUDGE_EVALUATION_REPORT.md`.
   - **NO DUPLICATE FILES RULE**: Do not create auxiliary markdown files in `docs/` or elsewhere that duplicate system architecture, roadmap, or benchmark results.
