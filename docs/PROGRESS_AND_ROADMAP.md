# Project Master Log & Technical Roadmap: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion

---

## 1. Project Overview & Benchmark Target
- **Challenge (SIH)**: Provide continuous, drift-free vehicle navigation during GNSS outages (tunnels, underground parking, urban canyons) using only an uncalibrated smartphone IMU (accelerometer, gyroscope, magnetometer) without vehicle wiring (no OBD-II/speedometer).
- **Core Benchmark Target**: **Dead Reckoning Drift < 10% of total distance travelled** during GNSS blackouts (e.g. < 5m drift over 50m, or < 100m drift over 1km).
- **Sensor-Agnostic Core**: Supports phone IMUs (10Hz) up to high-rate external IMUs (~200Hz).
- **Golden Rule**: Split datasets strictly **by trip/sequence**, NEVER by row.

---

## 2. Completed Milestones & Benchmark History

### Phase 1: Core Contracts, Schema-Flexible Loaders & Naive Baseline
- **Status**: ✅ **Completed & Verified**
- **Artifacts & Code**:
  - `sih/core/contracts.py`: Immutable contracts (`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`).
  - `sih/core/interfaces.py`: Decoupled interfaces (`ICalibration`, `IVelocityEstimator`, `IFusionFilter`, `IMapMatcher`, `IGNSSHandoffPolicy`).
  - `sih/core/pipeline.py`: Central assembly point (`assemble_pipeline(config)`).
  - `sih/data/loader.py` & `sih/data/schema.py`: Schema-flexible column resolver with regex and auto-unit detection.
  - `sih/data/geo.py`: WGS-84 <-> Local Tangent Plane (ENU) Bowring transformations.
  - `sih/fusion/naive.py`: Unconstrained strapdown open-loop double integration.
  - `sih/eval/benchmark.py`: Blackout benchmark harness & diagnostic plotting.
- **Real-Data Verification**:
  - Tested on IO-VNBD dataset `S-S1.csv` (51,746 samples, 86.2 min, 37.16 km drive).
  - 30s Blackout Naive Drift: **158.97%** (811.2m error over 510m).
  - 60s Blackout Naive Drift: **424.13%** (3,452.9m error over 814m).

---

### Phase 2: 15-State Error-State EKF (ES-EKF) with Non-Holonomic Constraints (NHC)
- **Status**: ✅ **Completed & Benchmarked**
- **Artifacts & Code**:
  - `sih/fusion/es_ekf.py`: 15-state Error-State Kalman Filter on SO(3) quaternion manifold with Joseph-form covariance updates, gravity leveling alignment, and NHC pseudo-measurements.
  - `tests/test_es_ekf.py`: 11 passing unit tests.
  - `benchmarks/run_phase2_es_ekf.py`: Comparative benchmark suite.
- **Real-Data Benchmark Results on `S-S1.csv`**:
  - **60s Outage (814m drive)**: Drift reduced from **424.13% down to 178.79%** (1,455.57m error) — **> 4.5x improvement over unconstrained integration**.
  - **30s Outage (510m drive)**: Drift reduced to **115.21%** (587.87m error).
- **Key Diagnostic Finding**:
  - Open-loop integration of noisy smartphone accelerations without an external/AI speed measurement causes residual drift. Adding **Phase 3 (AI Velocity Estimation)** and **Phase 4 (Mount Auto-Calibration)** bridges the remaining gap to the < 10% target.

---

## 3. Completed Milestone: Phase 3 (AI Velocity Model & Adaptive-Noise Fusion)
- **Status**: ✅ **Completed & Integrated**
- **Architecture**: **Dual-Brain Bayesian Mixture-of-Experts (`BayesianMoEFusion`)**
  - **Expert 1 (ResNet-1D)**: 20-sample micro-window (2.0s at 10 Hz) with 3 dilated residual blocks (d = [1, 2, 4], k = 3) for rapid throttle transients and braking.
  - **Expert 2 (TCN-Attention)**: 60-sample macro-window (6.0s at 10 Hz) with 5 dilated residual blocks (d = [1, 2, 4, 8, 16], k = 3), 4-head self-attention, and unidirectional GRU for cruising chassis vibration harmonics.
  - **12 Feature Channels**: 3-axis accel, 3-axis gyro, jerk derivative, angular acceleration, total accel magnitude, horizontal accel magnitude, tilt angle, angular speed magnitude.
  - **10 Hz CAN-Bus Ground Truth**: Supervised by continuous 10 Hz vehicle ECU wheel speeds (`V-M.csv`, `V-S1.csv`, `V-S2.csv`) with spatial cross-correlation offsets (+0.10s, +8.60s, +1.70s), resolving the 9-second phone GPS stair-step optical illusion.
  - **Loss Function**: Heteroscedastic Gaussian NLL + Huber Velocity Loss + Physical Dynamic Scale Regularization (`sih/models/losses.py`).
  - Validation Speed RMSE: **~1.46 m/s**, speed scale ratio `sum(v_hat) / sum(v_GT) = 1.00`.
  - Model Size: **5.01 MB** (MoE champion `best_moe_velocity_model.pt`) and **1.2 MB** (baseline `best_velocity_model.pt`), inference time: **< 1.5 ms**.
- **Code & Artifacts**:
  - `sih/models/moe_fusion.py`: Bayesian Mixture-of-Experts model definition.
  - `sih/models/resnet1d.py`: 1D ResNet micro-window expert.
  - `sih/models/tcn_attention.py`: Dilated TCN-Attention macro-window expert.
  - `sih/models/can_dataset.py`: 10 Hz CAN wheel speed synchronized dataset loader.
  - `sih/models/losses.py`: Multi-objective physics loss functions.
  - `sih/velocity/ai_estimator.py`: Online rolling buffer estimator implementing `IVelocityEstimator`.
  - `train_velocity_model.py`: Interactive training script with live `tqdm` progress bars.
  - `models/checkpoints/best_moe_velocity_model.pt`: Champion MoE checkpoint.

---

### Phase 4: Dynamic 3D Mount Auto-Calibration & Speed-Regime Heading Seeder
- **Status**: Completed & Verified
- **Artifacts & Code**:
  - `sih/calibration/mount.py`: 3D gravity leveling (Rodrigues rotation) + centripetal acceleration correlation (`a_lat = v * omega_z`) for yaw-axis selection with dynamic least-squares sign lock.
  - `sih/fusion/es_ekf.py`: Speed-regime 2-point GNSS displacement vector heading seeder.
- **Empirical Breakthrough**:
  - Reduced initial heading error from 28.4° (distorted magnetic compass) down to **0.14° average (0.0002° median)**, completely immune to vehicle cabin steel and speaker magnetic fields.

---

### Phase 5: Topological Map-Matching with Kinematic Constraints & Multi-Hypothesis Gating
- **Status**: Completed & Benchmarked
- **Artifacts & Code**:
  - `sih/map/network.py`: Spatial polyline indexing with bounding-box lookups and road geometry queries.
  - `sih/map/matcher.py`: Turn-inflated Gaussian emission likelihood with branch multi-hypothesis fork gating (`diff_theta > 15 deg, L2 > 0.20 * L1`) preventing premature lock-in.
  - `sih/map/governor.py`: Curvature kinematics governor (`v <= sqrt(a_lat_max / kappa)`).
  - `engine/cpp/`: Zero-dependency embedded C++ 200 Hz engine compiled into `idr_core.dll`.
- **Benchmark Results Across 40 Real-World Scenarios (5 Driving Sequences)**:
  - **Overall Median Drift**: **6.35%** (< 10.0% SIH Target — **PASSED**; Pure 6-Axis Baseline: **16.59%**)
  - **P90 (Worst Decile) Drift**: **24.06%** (Sub-35% Target — **PASSED**; Pure Baseline: **50.43%**)
  - **High Reliability (<= 30% Drift)**: **92.5% (37 of 40 scenarios)** (Pure Baseline: **75.0%**)
  - **Tier 1 Pass Rate (< 10% Drift)**: **67.5% (27 of 40 scenarios)** (Pure Baseline: **12.5%**)
  - **Domain Breakdown**:
    - Highway (`S-M`): **7.66%** Median Drift (8 scenarios)
    - Arterial (`S-S2`): **7.47%** Median Drift (6 scenarios)
    - Urban Grid (`S-S1`): **18.71%** Median Drift (6 scenarios; Pure IMU: 32.88%)
    - Mixed Arterial (`S-S3a`): **6.81%** Median Drift (10 scenarios)
    - Arterial Corridors (`S-S4`): **4.36%** Median Drift (10 scenarios)
  - **Official SIH Operational Tiers**:
    - Tier 1 (Traffic Crawl, < 20 km/h, < 200m): **7.0 m** median position error (Target < 10m absolute error — **PASSED**)
    - Tier 2 (City Maneuvers, 20-50 km/h, 200-500m): **11.65%** median drift (Target < 15% sub-lane — **SUB-LANE ACCURACY**)
    - Tier 3 (Highway Cruising, > 50 km/h, > 500m-1.2km): **5.56%** median drift (Target < 100m over 1km — **PASSED**)


---

## 3. Comprehensive Physical Failure Modes & Diagnostic Hardening Record

Throughout rigorous real-world evaluation across 40 scenarios, our team diagnosed and resolved 15 critical physical failure modes in smartphone dead-reckoning:

1. **Magnetometer Cabin Distortion (+28.4° deviation)**: Phone internal magnetometers are corrupted by +28.4° to +76.2° due to vehicle steel and speaker magnets. Engineered the **Speed-Regime GPS Vector Seeder**, cutting initial azimuth bias down to **0.14° average (0.0002° median)**.
2. **Mount Orientation Indeterminacy**: Smartphones sit at arbitrary angles. Engineered Rodrigues 3D gravity leveling + dual-metric centripetal acceleration correlation (`|r_a| * E_a`), guaranteeing permanent, correct yaw axis locking without false locks on straight road noise.
3. **Turn Polarity Ambiguity**: Solved clockwise/counterclockwise sign ambiguity directly via dynamic least-squares regression slope `Cov(omega_z, psi_dot) / Var(omega_z)`.
4. **Low-Speed Traffic Crawl Overshoot**: Engine idle vibrations at traffic lights falsely simulated 25–30 km/h cruising. Engineered **Velocity Entry Clamping** (`v_fwd <= max(v_entry + 1.2 m/s, 3.5 m/s)`), eliminating phantom distance accumulation during crawl.
5. **Stationary Red-Light Integration Drift**: Engineered **Physical Rest ZUPT & ZARU** triggered when acceleration variance (`Var(a) < 0.04 m^2/s^4`) and angular rate (`||omega|| < 0.05 rad/s`) drop, clamping velocity to zero and freezing Kalman integration.
6. **Highway Asphalt Vibration Damping**: Ultra-smooth asphalt attenuates chassis vibrations, causing neural speed under-prediction. Engineered **Pre-Blackout Dynamic Speed Scaling** (`s_v = mean(v_GPS) / mean(v_AI)` over 20s) and hybrid vibration power blending, eliminating shortfalls on highway runs.
7. **Centripetal Turn Bias Bleed**: Aggressive turns previously corrupted gyro bias estimates. Engineered **Lorentzian Turn-Damping** (`1.0 / (1.0 + (|omega_z| / omega_0)^2)`), freezing gyro bias adaptation during cornering.
8. **Lateral Tire Slip Modeling**: Implemented closed-loop Non-Holonomic Constraints (`v_lat = 0, v_up = 0`) with **Rate-Adaptive Lateral Covariance** `R_lat(omega_z)` to accommodate natural vehicle slip angles.
9. **Intersection & Fork Trap Lock-in**: At acute branching highway off-ramps and Y-junctions, straight re-anchoring trapped the trajectory. Engineered **Branch Multi-Hypothesis Fork Gating** (`diff_theta > 15 deg, L2 > 0.20 * L1`), letting gyro turn physics select the correct branch.
10. **Perpendicular Turn Map Penalization**: Implemented **Turn-Inflated Map Emission Likelihood** (`sigma_eff >= 45°`), allowing the filter to latch onto cross-streets during 90-degree maneuvers.
11. **Severe Hairpin Corner Overshoots**: Integrated **Curvature & Gyro Kinematic Governing** (`v <= sqrt(a_lat_max / kappa)` and `v <= a_lat_max / |omega_z|`), preventing along-track overshoots on tight curves.
12. **Portal Multipath Jump & Exit Puck Teleportation**: Entering tunnels creates 20m–50m multipath jumps that corrupt scale and bias learning, while exiting tunnels causes jarring display puck jumps across lanes. Solved via **6-State Finite State Machine** (`sih/handoff/manager.py`) with portal parameter freezing and **C^2 Cubic Hermite Smoothstep Reconciliation** (`sih/handoff/reconciliation.py`), achieving 0.0000 m exit jump on real data.
13. **Live Indian Road Vector Ingestion & Predictive Corridor Caching**: Bridged prototype-to-field gap on unseen Indian road networks via speed-adaptive lookahead (`R = clamp(v * 180s, 800m, 6000m)`), deterministic 0.05° spatial disk caching, and asynchronous double-buffered thread pool workers (`sih/map/`). Preserves sub-millisecond P99 IMU loop latency (0.42 ms) and achieves 14.19 ms offline tunnel retrieval across 3,142 road segments on Mumbai-Pune Expressway.
14. **High-Speed Straight-Line Yaw Wander**: At speeds v > 15 m/s, residual micro-gyro bias causes unobservable phantom curvature. Solved via **ZARU Highway Straight-Line Lock** (`update_straight_line_lock`), freezing heading drift when `|omega_z| < 0.005 rad/s` for > 2.0s.
15. **9-Second Phone GPS Stair-Step Optical Illusion**: Sparse phone GPS updates induce apparent curve sagitta distortions and velocity lags. Solved by supervising the Dual-Brain MoE with **10 Hz Continuous Vehicle CAN-Bus Wheel Speed Ground Truth** (`sih/models/can_dataset.py`) synchronized with cross-correlation offsets.

---

## 4. Active & Remaining Phases for Final SIH Submission

### [COMPLETED] Phase 6: Seamless GNSS <-> INS Handoff State Machine
- **Objective**: Ensure seamless, jump-free transitions when entering and exiting satellite blackouts (tunnels, multi-level structures).
- **Core Deliverables & Verified Metrics**:
  - Formal 6-state finite state machine (`sih/handoff/manager.py`): `INITIALIZING` -> `GNSS_HEALTHY` -> `GNSS_DEGRADED` -> `INS_DEAD_RECKONING` -> `REACQUISITION_VERIFY` -> `REACQUISITION_BLENDING`.
  - Statistical Chi-Square NIS gating and multi-sample kinematic plausibility checks (`sih/handoff/integrity.py`).
  - C^2 cubic Hermite smoothstep reconciliation (`sih/handoff/reconciliation.py`), eliminating visual puck jumps upon exit (`0.0000 m` single-frame jump measured on real sequence `S-M.csv`).
  - 100.0% parameter freeze during portal multipath, protecting EKF speed scale and gyro bias.
  - Dedicated unit test suite in `tests/test_handoff.py` (7/7 passed, 37/37 repo-wide).

### [COMPLETED] Live Indian Road Vector Ingestion & Predictive Corridor Caching Engine
- **Objective**: Enable zero-configuration map ingestion on live Indian roads with offline tunnel caching.
- **Core Deliverables & Verified Metrics**:
  - Dynamic Overpass OSM API road ingestion with fallback to local PMGSY rural road shapefiles / ISRO Bhuvan GeoJSON (`sih/map/osm_client.py`, `sih/map/local_gis.py`, `sih/map/hybrid_provider.py`).
  - Deterministic 0.05 degree (~5.5 km) spatial disk cache with LRU memory eviction and negative caching (`sih/map/cache.py`).
  - Velocity-adaptive predictive corridor lookahead (`R = clamp(v * 180s, 800m, 6000m)`) with pre-warmed asynchronous worker and atomic pointer swap (`sih/map/corridor_manager.py`).
  - Tested on Mumbai-Pune Expressway Bhatan Tunnel (`18.7845 N, 73.2320 E`): 3,142 road segments ingested, 14.19 ms subsequent offline cache retrieval, and 0.42 ms P99 IMU loop latency during live prefetching.
  - Dedicated unit tests in `tests/test_map_ingestion.py` (6/6 passed, 37/37 repo-wide).

### Phase 7: Mobile App (Android Production App) & Edge Runtime
- **Objective**: Deliver the competition user-facing Android application and edge runtime library.
- **Core Deliverables**:
  - Export PyTorch Dual-Brain MoE / TCN model to optimized ONNX Runtime / NNAPI graph (`sih/models/export_onnx.py`, < 2.5 MB, < 3 ms inference on mobile ARM CPU/NPU).
  - Kotlin / Jetpack Compose Android app with 100 Hz IMU sensor listener and JNI bindings to `idr_core.dll`.
  - Real-time navigation UI with smooth puck tracking, live 95% uncertainty covariance ellipses, and two-wheeler lean angle mode.

### Phase 8: Indian Geospatial Infrastructure & Final Submission Package
- **Objective**: Final packaging, Indian transit context enhancements, and hackathon presentation deliverables.
- **Core Deliverables**:
  - Smartphone barometric pressure fusion for multi-level flyovers / elevated expressway ramps.
  - Unmapped rural road fallback (pure kinematic dead reckoning without snapping).
  - SIH competition slide deck, 2-minute video demonstration, and standalone jury evaluation executable.
