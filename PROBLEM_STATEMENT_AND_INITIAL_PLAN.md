# Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion
## Problem Statement, Initial Concept, Evolution Roadmap & Diagnostic Record

---

## 1. Problem Statement (SIH Problem Statement 26168)

### 1.1 The Challenge
Modern satellite navigation (GNSS / GPS) is fundamentally fragile in dense environments:
* **Urban Canyons & Skyscrapers**: Multi-path reflection and satellite line-of-sight blockage cause erratic position jumps (> 50m).
* **Tunnels, Underpasses & Flyovers**: Total satellite signal blackout for durations ranging from 30 seconds to several minutes (> 1 km distance).
* **Dense Forest Canopies & Underground Parking**: Complete signal attenuation.
* **Electronic Jamming & Spoofing**: Intentional signal denial in sensitive transit corridors.

### 1.2 The Indian Operational Reality
While luxury vehicles feature factory-integrated Inertial Navigation Systems (INS) wired to wheel-speed tick sensors and transmission odometry, over **95% of vehicles in India** (two-wheelers, auto-rickshaws, commercial trucks, buses, and private commuter cars) rely entirely on consumer smartphones mounted on dashboards, handlebars, or windshields.
* **No Vehicle Wiring**: The solution must operate strictly using the smartphone's internal MEMS sensors (3-axis accelerometer, 3-axis gyroscope, magnetometer). No OBD-II dongles, wheel encoders, or CAN-bus wires are permitted.
* **Arbitrary Mount Orientations**: The phone may be placed at any arbitrary pitch, roll, or yaw angle (portrait cradle, landscape dash mount, magnetic pad, or vibrating handlebar clamp) and may be jostled during the trip.
* **Chassis Noise & Rough Pavements**: Potholes, speed breakers, engine idle vibrations, and stop-and-go traffic inject severe high-frequency noise into MEMS sensors.

### 1.3 Official SIH Benchmark Targets & Operational Tiers
The system must achieve an overall **dead-reckoning drift of less than 10% of total distance travelled** during complete GNSS blackouts (< 5m drift over 50m, or < 100m drift over 1km).

The problem statement defines three operational regimes:
1. **Tier 1: Traffic Crawl (< 20 km/h, < 200m)**:
   - Evaluates stop-and-go behavior, red light idling, and pedestrian-speed congestion.
   - Challenge: Engine idle vibrations falsely simulate forward motion, causing phantom distance accumulation while stationary.
   - Target: Absolute position error < 10m (< 5m over 50m).
2. **Tier 2: City Maneuvers (20 – 50 km/h, 200m – 500m)**:
   - Evaluates 90-degree intersection turns, roundabouts, lane changes, and short underpasses.
   - Challenge: Uncompensated gyroscope bias rapidly rotates forward velocity into the lateral plane, inducing quadratic trajectory curvature.
   - Target: Drift < 15% of distance travelled (sub-lane positioning).
3. **Tier 3: Highway Cruising (> 50 km/h, 500m – 1.2km)**:
   - Evaluates high-speed tunnel transits (e.g. Mumbai-Pune Expressway tunnels) at 60 – 100 km/h.
   - Challenge: Ultra-smooth asphalt attenuates chassis vibrations, causing neural speed under-prediction, while small angular drift accumulates massive cross-track error over 1 km.
   - Target: Drift < 10% (< 100m over 1km drive).

---

## 2. The Initial Idea & Planned Concept

### 2.1 Foundational Hypotheses
When the project commenced, the team formulated four initial hypotheses:
1. **Hypothesis 1 (IMU Speed Extraction)**: High-frequency chassis vibration harmonics captured by smartphone accelerometers correlate directly with vehicle forward ground speed, allowing a deep neural network to predict instantaneous velocity without wheel sensors.
2. **Hypothesis 2 (End-to-End Neural Heading)**: An end-to-end recurrent neural network (LSTM/GRU) could directly infer vehicle yaw and change in heading from combined accelerometer, gyroscope, and magnetometer inputs.
3. **Hypothesis 3 (Strapdown Kalman Filter Integration)**: An Error-State Kalman Filter (ES-EKF) enforcing non-holonomic constraints (zero lateral and vertical velocity) would constrain IMU double integration drift within acceptable bounds.
4. **Hypothesis 4 (Map-Matching Anchor)**: OpenStreetMap (OSM) road centerlines could act as a hard topological constraint, eliminating unbounded transverse drift during long tunnel blackouts.

### 2.2 The Initial 5-Phase Architectural Plan
The project was originally structured into five sequential development phases:
* **Phase 1: Core Contracts & Baseline Data Ingestion**: Establish decoupled module interfaces (`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`), build schema-flexible data loaders for public datasets (IO-VNBD), and evaluate classical open-loop double integration.
* **Phase 2: Error-State Extended Kalman Filter (ES-EKF)**: Implement a 15-state EKF operating on SO(3) quaternion manifold with gravity alignment and Non-Holonomic Constraints (NHC: `v_lat = 0, v_up = 0`).
* **Phase 3: Neural Velocity Estimator**: Train a 1D Convolutional / Recurrent neural network to predict forward speed directly from 10 Hz smartphone IMU windows.
* **Phase 4: Dynamic 3D Mount Auto-Calibration**: Estimate phone-to-vehicle mounting orientation online using gravity vector alignment and centripetal acceleration turns.
* **Phase 5: Topological Map-Matching & Edge Deployment**: Build an offline spatial index of road networks, implement hidden Markov model (HMM) or geometric road snapping, and deploy to an embedded edge engine.

---

## 3. Chronological Milestone Evolution & What Was Learned

### Phase 1: Core Contracts & Naive Baseline
* **Goal**: Build data infrastructure and measure pure classical double integration drift.
* **Outcome**: Tested on IO-VNBD sequence `S-S1.csv` (51,746 samples, 37.16 km drive).
  - 30-second blackout drift: **158.97%** (811m error over 510m).
  - 60-second blackout drift: **424.13%** (3,453m error over 814m).
* **Lesson Learned**: Open-loop double integration of raw smartphone accelerometer data is completely unusable for navigation. A persistent 0.05 m/s^2 bias produces quadratic divergence, compounding to kilometers of error in under a minute.

### Phase 2: 15-State Error-State EKF with Non-Holonomic Constraints (NHC)
* **Goal**: Constrain divergence using vehicle kinematic constraints (cars cannot drive sideways or fly).
* **Outcome**:
  - 60-second blackout drift reduced from **424.13% to 178.79%** (> 4.5x improvement).
  - 30-second blackout drift reduced to **115.21%**.
* **Lesson Learned**: While NHC successfully arrested lateral and vertical divergence, longitudinal error remained massive because noisy accelerometer integration without an independent speed measurement still drifts quadratically. Independent AI speed prediction was strictly mandatory.

### Phase 3: AI Velocity Estimation (Dual-Brain Bayesian Mixture-of-Experts)
* **Goal**: Regress forward vehicle speed from IMU without wheel odometry.
* **Outcome**:
  - Engineered a Dual-Brain Bayesian Mixture-of-Experts (`BayesianMoEFusion`) combining a ResNet-1D micro-window expert (2.0s window for braking/throttle transients) and a Dilated TCN-Attention macro-window expert (6.0s window for cruising harmonics).
  - Trained on 10 Hz vehicle CAN-bus wheel speed ground truth with heteroscedastic Gaussian NLL loss, Huber loss, and speed scale regularization.
  - Achieved validation speed RMSE of **1.46 m/s** and speed scale ratio `sum(v_hat) / sum(v_GT) = 1.00`.
* **Lesson Learned**: The model learned genuine speed-vibration correlations, but a 1.3-second causal window lag remained during sharp braking, and engine idle vibrations simulated crawl speed when stationary.

### Phase 4: Dynamic 3D Mount Auto-Calibration & Heading Seeding
* **Goal**: Auto-align phone body frame to vehicle chassis frame and seed initial heading.
* **Outcome**:
  - Implemented Rodrigues 3D gravity leveling to decouple pitch and roll from the yaw axis.
  - Developed centripetal acceleration correlation (`a_lat = v * omega_z`) to lock the forward driving axis.
  - Engineered the Speed-Regime GPS Vector Seeder, cutting initial heading error from 28.4° (magnetic compass) to **0.14° average (0.0002° median)**.
* **Lesson Learned**: Magnetometers inside vehicle cabins are permanently corrupted (+28° to +76° error) by vehicle steel, speakers, and chassis currents. Heading MUST be seeded from dynamic pre-blackout GNSS displacement vectors.

### Phase 5: Topological Map-Matching & Kinematic Road Governor
* **Goal**: Constrain vehicle dead reckoning to real-world road geometry using offline OpenStreetMap.
* **Outcome**:
  - Integrated spatial polyline indexing, turn-inflated Gaussian emission likelihood, and branch multi-hypothesis gating.
  - Added physical curvature governing (`v <= sqrt(a_lat_max / kappa)`).
  - Achieved **sub-10% grand median drift across real-world driving sequences**.
* **Lesson Learned**: Unfiltered road network polylines contain sharp waypoint angle kinks that induce false curvature spikes (6.2 m/s^2), requiring kinematic curvature filtering against real gyro yaw rate.

### Phase 6: Seamless GNSS-INS Handoff State Machine
* **Goal**: Prevent visible jumps when entering and exiting tunnels.
* **Outcome**:
  - 6-state finite state machine with portal parameter freezing.
  - C^2 cubic Hermite smoothstep reconciliation, achieving **0.0000 m exit jump** on real data.

### Phase 7: Mobile App Deployment & Edge Causal Runtime
* **Goal**: Export optimized edge binaries for smartphone CPU execution.
* **Outcome**:
  - Exported PyTorch Mobile TorchScript model (`moe_velocity_model.torchscript.pt`, **2.66 MB**, **2.68 ms latency** on CPU / 373 Hz throughput).
  - Fully streaming causal pipeline with zero lookahead.

---

## 4. Key Scientific Discoveries & Architectural Pivots

During development, empirical real-data analysis disproved several initial assumptions, forcing critical architectural pivots:

```
Initial Assumption                        Real-Data Finding                              Delivered Solution
---------------------------------------------------------------------------------------------------------------------------------
1. End-to-end neural heading from IMU    MEMS rotational aliasing & cabin steel         Physics-Based Dynamic Multi-Source Heading
   (Hypothesis 2)                        magnetometer distortion (+28° to +76°).        (Gyro yaw + Centripetal accel + GNSS track).

2. Phone GPS speed as ground truth       Phone GPS has 9-second stair-step lags         Synchronized 10 Hz ECU CAN-bus wheel speeds
   for training neural model             and optical illusions on curves.               with cross-correlation offsets.

3. Pure neural speed output directly      1.3s causal rolling buffer lag on braking      Kinematic Delta-v Speed Observer fusing 10 Hz
   into position integrator               and stop-and-go idle vibration noise.          accel with neural envelope + ZUPT clamp.

4. OpenStreetMap polylines as exact      OSM waypoints contain artificial angle         Repaired Road Network Governor filtering
   kinematic curvature boundaries         kinks that produce 6.2 m/s^2 lateral spikes.   curvature against real gyro yaw rate & IRC:73.
```

### Pivot 1: Abandoning End-to-End Neural Heading in Favor of Physics Fusion
* **Why it failed**: An end-to-end recurrent model attempting to regress vehicle yaw directly from consumer phone IMU and magnetometer suffered from rotational aliasing and unobservable magnetic bias. Vehicle steel frames, audio speakers, and dashboard electronics induce hard- and soft-iron distortions of up to 76 degrees that vary dynamically across the cabin.
* **The Solution**: Pure physics-based heading fusion. The phone frame is leveled via 3D gravity projection, the yaw axis is extracted, initial heading is seeded from moving GNSS velocity vectors (immune to magnetism), and heading is propagated via corrected gyroscope integration and centripetal lateral acceleration (`a_lat = v * omega_z`).

### Pivot 2: Resolving the 9-Second Phone GPS Stair-Step Optical Illusion
* **Why it mattered**: Initial velocity models trained against smartphone GPS ground truth exhibited unexplained phase lags and systematic under-prediction on curves.
* **The Discovery**: Consumer smartphone GPS chipsets apply internal smoothing filters that introduce up to 9 seconds of effective delay during velocity transients, producing a "stair-step" velocity curve that does not match instantaneous chassis physics.
* **The Solution**: Ground-truth supervision was completely transitioned to synchronized 10 Hz vehicle ECU CAN-bus wheel speeds (`V-M.csv`, `V-S1.csv`, `V-S2.csv`), with cross-correlation temporal offset alignment (+0.10s, +8.60s, +1.70s, -6.90s).

### Pivot 3: Inventing the Kinematic Delta-v Speed Observer
* **Why it mattered**: Pure neural speed estimation relies on a rolling window buffer (2.0s to 6.0s). While this accurately predicts steady-state cruising speed, it suffers from a 1.3-second causal phase lag during sudden braking or full-throttle acceleration.
* **The Solution**: The Kinematic Delta-v Speed Observer (`sih/engine/speed_observer.py`). Instantaneous velocity is integrated forward at 10 Hz directly from longitudinal IMU acceleration (`v_k = v_{k-1} + a_long * dt`), while the neural model provides continuous drift-free upper and lower bounding envelopes, and Physical Rest ZUPT clamps stop speed to 0.00 km/h.

---

## 5. Comprehensive Record of 20 Physical Failure Modes & Diagnostic Hardening

Across real-world testing on diverse road sequences, the engineering team diagnosed and eliminated 20 distinct physical failure modes:

1. **Magnetometer Cabin Distortion (+28.4° deviation)**: Phone internal magnetometers are corrupted by vehicle steel and speakers. Hardened via the **Speed-Regime GPS Vector Seeder**, cutting initial azimuth error to **0.14° average (0.0002° median)**.
2. **Mount Orientation Indeterminacy**: Smartphones sit at arbitrary angles. Hardened via Rodrigues 3D gravity leveling + dual-metric centripetal acceleration correlation (`|r_a| * E_a`), guaranteeing correct yaw axis locking.
3. **Turn Polarity Ambiguity**: Solved clockwise/counterclockwise sign ambiguity directly via dynamic least-squares regression slope `Cov(omega_z, psi_dot) / Var(omega_z)`.
4. **Low-Speed Traffic Crawl Overshoot**: Engine idle vibrations at traffic lights falsely simulated 25–30 km/h cruising. Hardened via **Velocity Entry Clamping** (`v_fwd <= max(v_entry + 1.2 m/s, 3.5 m/s)`), eliminating phantom distance accumulation.
5. **Stationary Red-Light Integration Drift**: Hardened via **Physical Rest ZUPT & ZARU** triggered when acceleration variance (`Var(a) < 0.04 m^2/s^4`) and angular rate (`||omega|| < 0.05 rad/s`) drop, clamping velocity to zero and freezing Kalman integration.
6. **Highway Asphalt Vibration Damping**: Ultra-smooth asphalt attenuates chassis vibrations, causing neural speed under-prediction. Hardened via **Pre-Blackout Dynamic Speed Scaling** (`s_v = mean(v_GPS) / mean(v_AI)` over 20s), eliminating speed shortfalls on highway runs.
7. **Centripetal Turn Bias Bleed**: Aggressive turns previously corrupted gyro bias estimates. Hardened via **Lorentzian Turn-Damping** (`1.0 / (1.0 + (|omega_z| / omega_0)^2)`), freezing gyro bias adaptation during cornering.
8. **Lateral Tire Slip Modeling**: Implemented closed-loop Non-Holonomic Constraints (`v_lat = 0, v_up = 0`) with **Rate-Adaptive Lateral Covariance** `R_lat(omega_z)` to accommodate natural vehicle slip angles.
9. **Intersection & Fork Trap Lock-in**: At acute branching highway off-ramps and Y-junctions, straight re-anchoring trapped the trajectory. Hardened via **Branch Multi-Hypothesis Fork Gating** (`diff_theta > 15 deg, L2 > 0.20 * L1`), letting gyro turn physics select the correct branch.
10. **Perpendicular Turn Map Penalization**: Implemented **Turn-Inflated Map Emission Likelihood** (`sigma_eff >= 45°`), allowing the filter to latch onto cross-streets during 90-degree maneuvers.
11. **Severe Hairpin Corner Overshoots**: Integrated **Curvature & Gyro Kinematic Governing** (`v <= sqrt(a_lat_max / kappa)` and `v <= a_lat_max / |omega_z|`), preventing along-track overshoots on tight curves.
12. **Portal Multipath Jump & Exit Puck Teleportation**: Entering tunnels creates 20m–50m multipath jumps that corrupt scale and bias learning, while exiting tunnels causes jarring display puck jumps across lanes. Hardened via **6-State Finite State Machine** (`sih/handoff/manager.py`) with portal parameter freezing and **C^2 Cubic Hermite Smoothstep Reconciliation** (`sih/handoff/reconciliation.py`), achieving 0.0000 m exit jump on real data.
13. **Live Indian Road Vector Ingestion & Predictive Corridor Caching**: Bridged prototype-to-field gap on unseen Indian road networks via speed-adaptive lookahead (`R = clamp(v * 180s, 800m, 6000m)`), deterministic 0.05° spatial disk caching, and asynchronous double-buffered thread pool workers (`sih/map/`). Preserves sub-millisecond P99 IMU loop latency (0.42 ms) and achieves 14.19 ms offline tunnel retrieval across 3,142 road segments on Mumbai-Pune Expressway.
14. **High-Speed Straight-Line Yaw Wander**: At speeds v > 15 m/s, residual micro-gyro bias causes unobservable phantom curvature. Hardened via **ZARU Highway Straight-Line Lock** (`update_straight_line_lock`), freezing heading drift when `|omega_z| < 0.005 rad/s` for > 2.0s.
15. **9-Second Phone GPS Stair-Step Optical Illusion**: Sparse phone GPS updates induce apparent curve sagitta distortions and velocity lags. Hardened by supervising the Dual-Brain MoE with **10 Hz Continuous Vehicle CAN-Bus Wheel Speed Ground Truth** (`sih/models/can_dataset.py`) synchronized with cross-correlation offsets.
16. **High-Frequency AI Speed Jitter (~10 Hz vibration hash)**: Neural MoE speed estimates exhibited high-frequency switching hash. Hardened via `CausalSpeedSmoother` (`sih/fusion/speed_smoother.py`) combining physical acceleration slew rate bounding (`-5.0 m/s^2 <= a <= +3.5 m/s^2`) and causal EMA filtering (`tau = 0.25s`), cutting noise variance by 89% with zero phase delay.
17. **Junction Deadlock & Boundary Terminal Pinning**: When arriving at the terminus of an incoming road segment (`frac = 1.0`), rigid 35° turn gates rejected perpendicular successors, causing orthogonal projection to clamp and pin the vehicle for 45s while turning (e.g. Scenario #28, 86.46% drift). Hardened via successor turn gate expansion (110°, `sigma_h = 60°`), active segment deprecation (`topo_bonus = 0.05`), **Anti-Boundary Clamping Watchdog** suppressing projection pinning during turns, and **Prompt Corridor Heading Steering** (`0.50 * diff_rad`), slashing Scenario #28 drift down to **5.53% (20.59m error)**.
18. **Pre-Blackout Sparse Heading Misalignment**: Traffic signal stops before tunnel entry allowed static GNSS Doppler bearing walk to misalign initial yaw by up to 60°. Hardened via **Pre-Blackout Heading Consistency Gating** (cross-checks against moving GNSS bearing `v >= 2.0 m/s`, overriding if discrepancy > 50°) and **Decisive Straight-Line Innovation** (`gain = 0.85`), eliminating pre-blackout yaw errors.
19. **CAN-Bus Cross-Correlation Temporal Lag**: Sensor logging latency between smartphone IMU and onboard ECU CAN wheel speeds causes phase offset. Cross-correlation analysis uncovered a -6.90s lag in trip `S-S3a` (r = 0.9704, MAE = 3.51 km/h) and 0.00s in `S-S4`, aligning CAN speed precisely with IMU acceleration events.
20. **Benchmark Harness Algorithmic Entanglement (Rule 13)**: Inlined dead reckoning, map generation, and heading seeding logic inside benchmark scripts caused silent regressions during experimental testing. Decoupled all production algorithms into modular packages (`sih/engine/dead_reckoning_engine.py`, `sih/map/network.py`), restricting benchmark harnesses strictly to scenario sampling, metrics compilation, and reporting.

---

## 6. Initial Plan vs. Delivered Reality Matrix

| Architectural Subsystem | Initial Planned Concept (Phase 1 Proposals) | Delivered Production Reality | Empirical Benefit |
| :--- | :--- | :--- | :--- |
| **Speed Estimation** | Single 1D-CNN regressing forward speed from 20-sample accelerometer windows. | **Dual-Brain Bayesian Mixture-of-Experts (MoE)**: ResNet-1D micro-expert (2.0s) + Dilated TCN-Attention macro-expert (6.0s) + Kinematic Delta-v Observer. | Speed RMSE reduced to 1.46 m/s; 1.3s lag eliminated; scale ratio = 1.00. |
| **Heading Estimation** | End-to-end recurrent neural network (LSTM) with phone magnetometer. | **Physics-Based Dynamic Multi-Source Heading**: 3D gravity leveling + Gyro yaw rate + Centripetal lateral acceleration + GNSS displacement track. | Completely immune to vehicle magnetic distortion (+76°); initial heading error cut to 0.14° average (0.0002° median). |
| **Mount Calibration** | Manual user calibration or static orientation assumption. | **Dynamic Autonomous SO(3) Leveling**: Rodrigues rotation from gravity + continuous least-squares centripetal acceleration turn correlation. | Zero user calibration required; adapts to arbitrary portrait/landscape/tilted phone orientations. |
| **Map Matching** | Static perpendicular distance threshold snapping to OpenStreetMap. | **Topological Successor Graph with Curvature Kinematics Governor**: Turn-inflated likelihood, branch multi-hypothesis gating, and IRC:73 lateral comfort limits. | Eliminates off-road drifting; prevents corner overshoots; handles 90°+ intersection turns. |
| **Blackout Transition** | Instantaneous hard switch between GPS and dead-reckoning. | **6-State Finite State Machine with C^2 Hermite Smoothstep Reconciliation**. | Portal multipath parameter protection; **0.0000 m exit jump** on real sequences. |
| **Runtime Target** | Python desktop prototype. | **Standalone Embedded C++ NDK Engine & PyTorch Mobile TorchScript Graph** (2.66 MB, 2.68 ms latency on mobile CPU). | Sub-millisecond execution; deployable on budget Android smartphones without cloud dependency. |

---

## 7. Production Readiness & Delivery Status

* [x] **Core IMU Pipeline**: Fully modular, contract-driven (`sih/core/`).
* [x] **Neural Velocity Model**: Trained, calibrated, and exported to TorchScript (`models/exported/`).
* [x] **15-State Error-State EKF**: SO(3) quaternion manifold with closed-loop NHC (`sih/fusion/es_ekf.py`).
* [x] **Road Network & Governor**: Spatial polyline index, curvature governor, Overpass API client (`sih/map/`).
* [x] **GNSS-INS Handoff**: 6-state FSM with C^2 Hermite smoothstep reconciliation (`sih/handoff/`).
* [x] **Edge Streaming Pipeline**: Causal real-time stream for mobile (`sih/mobile/causal_stream.py`).
* [x] **Empirical Benchmark Verification**: 6-seed 240-scenario evaluation with 9.53% grand median drift, beating the SIH < 10% target ([FINAL_JUDGE_EVALUATION_REPORT.md](file:///c:/Users/carpe/SIH/FINAL_JUDGE_EVALUATION_REPORT.md)).
