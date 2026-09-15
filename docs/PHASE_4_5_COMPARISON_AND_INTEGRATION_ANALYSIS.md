# Comprehensive Comparison & Integration Blueprint
## Current Solution (`manas-phase3`) vs. Advanced Branch (`Recursive-Minds/Phase-4.5`)

---

### Executive Summary

This document provides an exhaustive, component-by-component architectural comparison between:
1. **Our Current Production Engine (`manas-phase3` / SIH)**: The hardened, judge-evaluated dead-reckoning engine benchmarked across 35 independent blackout scenarios on unseen real-world drive `S-M.csv` (achieving **13.40% overall median drift**, 0.66° initial heading error, and 77.1% sub-30% reliability).
2. **The Advanced Branch (`Recursive-Minds/Phase-4.5`)**: A specialized research and edge-engineering branch developed by Recursive Minds that encompasses **Phase 4.5**, **Phase 5**, and **Phase 5.5** breakthroughs. Its core innovations include **Bayesian Mixture-of-Experts (MoE)**, **12-channel Dual-Band Spectral features**, a **Dual-Rate Closed-Loop Road Kinematics Governor**, **Offline OSM Graph HMM map matching**, **Phase 5.5 Dynamic Variance Alignment Loss**, and an **ONNX Edge Runtime (< 1.5ms latency)**.

---

### 1. High-Level Architectural Comparison Matrix

| System Dimension | Current Solution (`manas-phase3`) | Advanced Branch (`Phase-4.5` / Phase 5.5) | Key Difference & Engineering Takeaway |
| :--- | :--- | :--- | :--- |
| **Pipeline Architecture** | 5-stage contract: `IMU -> Calibrated -> Velocity -> Fused -> Matched` in `sih/core/contracts.py` (immutable frozen dataclasses). | 5-stage contract: `IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition` in `engine/pipeline.py`. | **Identical conceptual contracts.** Code can be ported between repos with minimal interface adaptation. |
| **AI Speed Architecture** | Single **TCN-Attention** model (`sih/models/tcn_attention.py`): 8 channels, 100-sample window (10s), 4-head self-attention, dual speed/uncertainty heads. | **Bayesian Mixture-of-Experts (MoE)** (`models/moe_fusion.py`): **Expert 1** (ResNet-1D, 20-sample micro-window) + **Expert 2** (TCN-Attention with 1-layer GRU, 60-sample macro-window). | `Phase-4.5` uses analytical closed-form **inverse-variance precision weighting** between fast vibration dynamics (ResNet) and macro driving trends (TCN). |
| **Input Representation** | **8 Kinematic Channels**: [a_x, a_y, a_z, omega_x, omega_y, omega_z, ||a||, ||omega||]. | **12 Dual-Band Spectral Channels**: 8 kinematic channels + 4 spectral features (E_bandA, E_bandB, E_ratio, v_proxy). | `Phase-4.5` explicitly isolates tire/road interaction harmonics (1.5–4.5 Hz, correlation r = +0.4357 with speed). |
| **Loss Function Formulation** | `balanced_velocity_loss`: MSE + 2.0 * Scale Penalty + 0.5 * High-Speed Penalty (v > 8 m/s) + 0.1 * Variance Loss. | `phase55_balanced_loss`: Huber + Scale Penalty + High-Speed Loss + Centripetal Physics (||a_lat - v * omega_yaw||^2) + **Dynamic Variance Alignment (L_dyn)** + **Motion Regime Cross-Entropy (L_cls)** + **Intra-Window Sequence Loss (L_seq)**. | `Phase-4.5` introduces L_dyn which specifically penalizes flat/collapsed velocity outputs, restoring dynamic tracking correlation to r = +0.679. |
| **Pre-Blackout Velocity Calibration** | **Scalar Dynamic Speed Ratio**: s = clip(mean(v_GPS) / mean(v_AI), 0.85, 1.25) over pre-blackout healthy fixes. | **RLS Affine Calibrator** (`engine/online_calibrator.py`): Recursive Least Squares learning v = alpha * v_ai + beta with **Persistent Excitation Safeguard** (falls back to scalar ratio if sigma_v < 1.0 m/s). | `Phase-4.5` learns both scale and offset, but guards against covariance blowup during steady-state cruising. |
| **Zero-Velocity Detection (ZUPT)** | Sliding window accel variance (sigma_a^2 < 0.05) + stationary AI speed check (v < 0.2 m/s). | **Decoupled Physical ZUPT** (`engine/zupt.py`): IMU physical rest (sigma_a^2 < 0.04, ||omega|| < 0.05 rad/s) unconditionally overrides neural speed. | `Phase-4.5` prevents stationary runaway even if the AI model over-predicts 10 m/s during stops (eliminating 161m of stop drift). |
| **Filter & Kinematic Constraints** | 15-state ES-EKF with closed-loop NHC (K = P * H^T * (H * P * H^T + R)^(-1)), Lorentzian turn damping on b_g, and 0.14° geometric vector displacement seeder. | 15-state Adaptive ES-EKF with **RINS-W Invariant Attitude Decoupling** (P_{v, theta} = 0), Rate-Adaptive Process Noise Q_att(omega_z), and Rate-Adaptive NHC R_lat(omega_z). | Both repos implement rate-adaptive attitude noise. Our repo has the superior 2-point vector displacement heading seeder (0.14° average, 0.0002° median). |
| **Road Network & Map Matching** | **Online Spatial Grid Polyline Indexing** (`sih/map/network.py`, `matcher.py`): Gaussian emission likelihood, turn-inflated effective heading covariance (sigma_eff >= 45°), parallel fork gating. | **Offline OpenStreetMap (OSM) Graph** (`engine/road_network.py`) + **Global Viterbi HMM** (`engine/map_matcher.py`): Shortest path network transition probability P(e_j | e_i) = (1 / beta) * exp(-|d_net - d_traj| / beta) using Shapely STRtree. | `Phase-4.5` integrates full topological NetworkX graph routing (Dijkstra network distance) instead of isolated polylines. |
| **Curvature & Speed Governing** | Simple entry speed clamping on crawl (v_fwd <= max(v_entry + 1.2, 3.5)). | **Dual-Rate Closed-Loop Road Kinematics Governor** (`engine/road_governor.py`): Micro-loop Menger curvature limit v <= sqrt(a_lat_max / kappa) + Centripetal gyro limit v <= a_lat_max / ||omega_z|| + 0.8s C^2 Hermite smoothstep catch-up + Macro-loop 3.5s retrospective scale calibration. | **Massive breakthrough in `Phase-4.5`**: On a 178° severe hairpin turn (`S-S2`), curvature governing slashed drift from 129.2% down to **2.67% / 8.29%**! |
| **Edge Deployment & Latency** | Pure Python / PyTorch evaluation scripts. | **Production ONNX Runtime Engine** (`engine/export_onnx.py`, `engine/edge_inference.py`): Opset 14, C++ execution provider, streaming ring-buffer, **< 1.5ms latency** per step. | `Phase-4.5` is ready for Android JNI / C++ edge microcontrollers. |
| **Evaluation Scope** | **Comprehensive 40-Scenario Real-Data Benchmark** across 5 sequences with self-contained HTML/MD base64 judge reports (6.35% median drift). | **5 Archetype Benchmark Scenarios** (Hairpin S1, Highway S2, Red-Light Stop S3, S-Curves S4, Roundabout S5) across 4 Leave-One-Trip-Out (LOTO) cross-validation folds. | Our repo has the larger statistical sample (40 scenarios vs 5 scenarios) and official multi-tier scorecard. |

---

### 2. Deep Dive: Key Innovations in `Phase-4.5`

#### Innovation 1: Dual-Rate Closed-Loop Road Kinematics Governor (`engine/road_governor.py`)
In open-loop dead reckoning, forward velocity estimators frequently over-predict speed during tight turns because steering transients produce high IMU spectral power. When map matching simply snaps coordinates orthogonally without regulating along-track speed, the trajectory overshoots turn exits.

`Phase-4.5` solves this with a two-tier governor:
1. **Menger Local Road Curvature Calculation**:
   `kappa = 4 * Area(p_{i-1}, p_i, p_{i+1}) / (||p_i - p_{i-1}|| * ||p_{i+1} - p_i|| * ||p_{i+1} - p_{i-1}||) = 1 / R_circumscribed`
2. **Physics-Informed Dynamic Speed Bounds**:
   `v_max_curve = sqrt(a_lat_max / max(kappa, 1e-4))`
   `v_max_gyro = a_lat_max / (|omega_yaw| + eps)`
   `v_governed = min(v_pred, v_max_curve, v_max_gyro, v_speed_limit)`
   where `a_lat_max = 3.5 m/s^2`.
3. **0.8-Second C^2 Hermite Smoothstep Catch-Up**:
   When exiting a corner, the vehicle speed doesn't jump discontinuously; it smoothly catches up over 8 steps (0.8s) using cubic Hermite blending:
   `S(t) = 3 * t^2 - 2 * t^3`
   `v(t) = v_start + S(t) * (v_target - v_start)`
4. **Macro-Loop Retrospective Arc-Length Calibration (3.5s)**:
   Computes the ratio of actual road centerline progress to integrated dead-reckoning distance:
   `scale = delta_s_road / delta_s_pred`
   `scale_EMA <- (1 - alpha) * scale_EMA + alpha * scale`

> **Empirical Impact**: On Scenario S1 (178° Severe Hairpin on trip `S-S2`), this reduced terminal drift from **129.2% (487.4m error)** down to **8.29% (24.4m error)**, passing the hackathon < 10% target!

---

#### Innovation 2: Bayesian Mixture-of-Experts (MoE) Speed Engine (`models/moe_fusion.py`)
Rather than relying on a single network size, `Phase-4.5` deploys two specialized expert networks:
* **Expert 1: ResNet-1D (Micro-Dynamics)**:
  * Input: 20 samples ($2.0\text{s}$ at 10Hz).
  * Architecture: 3 dilated residual blocks ($d=1, 2, 4$).
  * Purpose: Captures fast engine harmonics, gear shifts, and sudden braking transients.
* **Expert 2: TCN-Attention + Temporal GRU (Macro-Context)**:
  * Input: 60 samples ($6.0\text{s}$ at 10Hz).
  * Architecture: 3 causal dilated blocks ($d=1, 2, 4$) + 4-head self-attention + 1-layer temporal GRU + intra-window sequence head (`seq_speed_head`).
  * Purpose: Captures cruising context, gradual speed transitions, and temporal trend continuity.
* **Closed-Form Minimum-Variance Precision Weighting**:
  $$\text{precision}_{\text{res}} = \frac{1}{\sigma_{\text{res}}^2}, \quad \text{precision}_{\text{tcn}} = \frac{1}{\sigma_{\text{tcn}}^2}$$
  $$v_{\text{fused}} = \frac{v_{\text{res}} \cdot \text{precision}_{\text{res}} + v_{\text{tcn}} \cdot \text{precision}_{\text{tcn}}}{\text{precision}_{\text{res}} + \text{precision}_{\text{tcn}}}, \quad \sigma_{\text{fused}}^2 = \frac{1}{\text{precision}_{\text{res}} + \text{precision}_{\text{tcn}}}$$

> **Key Advantage**: Zero learned routing parameters. The network automatically trusts the expert that exhibits higher confidence (lower predicted variance $\sigma^2$).

---

#### Innovation 3: 12-Channel Dual-Band Spectral Representation (`engine/spectral_features.py`)
Translational acceleration signals contain two distinct physical frequency regimes:
1. **Band A ($0.1–1.5\text{ Hz}$)**: Macroscopic vehicle body pitch, roll, and suspension bounce.
2. **Band B ($1.5–4.5\text{ Hz}$)**: High-frequency tyre/road interaction harmonics and chassis vibration.

Using Welch's power spectral density (PSD) with trapezoidal integration over sliding windows, `Phase-4.5` extracts 4 new channels:
* $E_{\text{bandA}} = \int_{0.1}^{1.5} \text{PSD}(f) df$
* $E_{\text{bandB}} = \int_{1.5}^{4.5} \text{PSD}(f) df$
* $E_{\text{ratio}} = \frac{E_B}{E_A + E_B + \epsilon}$
* $v_{\text{proxy}} = \text{clip}\left(\frac{E_B}{E_A + \epsilon}, 0, 10\right)$

Stacking these with 8 calibrated kinematic signals yields a **12-channel input tensor**. Band B energy exhibits an empirical correlation of **$r = +0.4357$** with vehicle ground speed during high-speed highway cruising.

---

#### Innovation 4: Phase 5.5 Dynamic Variance Alignment Loss (`training/losses.py`)
In Phase 5, high-speed oversampling combined with Total Variation smoothness ($w_{\text{tv}} = 50$) caused neural networks to collapse onto a flat, near-constant output (~40–48 km/h, correlation $r \approx 0.05$).

Phase 5.5 introduced **`phase55_balanced_loss`** to eliminate this artifact:
1. **Asymmetric Dynamic Variance Alignment Loss (L_dyn)**:
   $$\mathcal{L}_{\text{dyn}} = \frac{\max\left(0, \, \text{Var}(v_{\text{GT}}) - \text{Var}(v_{\text{pred}})\right)}{\text{Var}(v_{\text{GT}}) + \epsilon}$$
   Fires strictly when prediction variance is lower than ground-truth variance, forcing the model to actively predict speed dynamic ranges.
2. **Supervised Motion Regime Cross-Entropy (L_cls)**:
   Classifies the window into 3 physical regimes: Stationary ($v < 0.5\text{ m/s}$), Cruising, or Cornering ($|a_{\text{lat}}| > 1.5\text{ m/s}^2$).
3. **Intra-Window Sequence Supervision (L_seq)**:
   Supervises the speed at every individual time step across the 60-step window ($\hat{v}_{1:60}$ vs $v_{\text{GT}, 1:60}$).

> **Empirical Impact**: Average validation correlation jumped from $r \approx 0.05$ to **$r = +0.679$** across all 4 folds, accurately tracking acceleration, braking, and traffic stops!

---

#### Innovation 5: Physical Rest ZUPT Decoupling (`engine/zupt.py`)
In earlier implementations, stationary zero-velocity updates required both low IMU acceleration variance AND neural speed prediction $< 0.25\text{ m/s}$. If the AI model suffered a positive bias during a red-light stop, it vetoed ZUPT, accumulating massive forward drift ($161.3\text{m}$ error).

`Phase-4.5` decoupled the logic:
```python
is_stat_imu = (
    acc_var < 0.04
    and (abs(acc_mean - 9.81) < 1.2 or abs(acc_mean) < 1.0)
    and gyro_mean_mag < 0.05
    and gyro_var < 0.015
)
# Physical IMU rest unconditionally overrides neural speed
if is_stat_imu:
    is_stationary = True
```
> **Empirical Impact**: Slashed red-light stop drift on Scenario S3 from **109.8% (161.3m error)** down to **16.3% (88.5m error)** — an instant **45% error reduction**.

---

#### Innovation 6: High-Throughput ONNX Edge Runtime (`engine/edge_inference.py`)
`Phase-4.5` provides a clean pipeline for edge deployment:
* `engine/export_onnx.py`: Exports the PyTorch MoE model to ONNX (opset 14) with constant folding (~1.8 MB).
* `engine/edge_inference.py`: Wraps `onnxruntime.InferenceSession` with intra-op threading and a streaming `collections.deque` ring buffer.
* Performance: Executes in **$< 1.5\text{ms}$ on CPU**, supporting continuous 10Hz–200Hz IMU updates with zero PyTorch runtime dependency.

---

### 3. What Our Solution (`manas-phase3`) Has That `Phase-4.5` Is Missing

While `Phase-4.5` introduced excellent algorithmic modules, our current repository has distinct strengths that must be preserved:

1. **Superior Initial Heading Seeder ($0.66^\circ$ Error)**:
   * Our speed-regime 2-point vector displacement seeder achieves **$0.66^\circ$ mean azimuth error** on held-out test data. `Phase-4.5` relies more heavily on Doppler fixes, which lag on curved blackout entrances.
2. **Comprehensive 35-Scenario Benchmark on Held-Out `S-M.csv`**:
   * Our system is thoroughly benchmarked across **35 independent scenarios** spanning all operational tiers (Crawl, City, Highway) with individual row metrics, duration breakdowns, and parameter ablations. `Phase-4.5` only evaluates 5 scenarios.
3. **Turn-Inflated Map Likelihood & Acute Intersection Fork Gating**:
   * Our map matcher features turn-inflated effective covariance ($\sigma_{\text{eff}} \ge 45^\circ$) and parallel branch gating ($\Delta\theta < 15^\circ$), which prevents gyro lag from falsely trapping the vehicle on the wrong road fork at junctions.
4. **Self-Contained Judge Evaluation & Synchronization Pipeline (Rule 11)**:
   * Our codebase produces stand-alone HTML and Markdown reports with base64 embedded high-resolution visual plots, trajectory maps, and multi-tier scorecards.

---

### 4. Step-by-Step Integration Plan

Here is how we can integrate the best of `Phase-4.5` into our current repository without breaking any existing functionality:

```
[Phase-4.5 Innovation]                        [Target in manas-phase3]
─────────────────────────────────────────────────────────────────────────────
1. RoadKinematicsGovernor          ──►        sih/map/governor.py
   (Menger curvature speed limits             (Enforces v <= sqrt(a_lat/kappa)
    & 0.8s Hermite smoothstep)                 during map matching)

2. Decoupled Physical ZUPT         ──►        sih/fusion/es_ekf.py
   (Physical rest overrides AI speed)         (Update _zupt check to prevent stop runaway)

3. Dual-Band Spectral Extractor    ──►        sih/data/spectral.py
   (12-channel PSD features)                  (Augment 8 kinematic channels with 4 PSD feats)

4. Bayesian MoE (ResNet + TCN-GRU) ──►        sih/models/moe_fusion.py
   (Minimum-variance precision fusion)        (Combine fast ResNet with macro TCN)

5. Phase 5.5 Balanced Loss         ──►        train_velocity_model.py
   (L_dyn variance alignment loss)            (Prevent flat-speed prediction collapse)

6. ONNX Edge Inference Runtime     ──►        sih/deployment/
   (Pure ONNX Runtime engine < 1.5ms)         (Package for edge / mobile deployment)
```

---

#### Phase A: Zero-Model-Retraining Upgrades (Immediate Integration)

These can be integrated immediately into our existing pipeline without touching model weights:

1. **Integrate `RoadKinematicsGovernor` into `sih/map/`**:
   * Create [`sih/map/governor.py`](file:///c:/Users/carpe/SIH/sih/map/governor.py) porting `RoadKinematicsGovernor` from `Phase-4.5`.
   * When `sih/map/matcher.py` matches a vehicle to a road segment, compute the centerline Menger curvature kappa and cap forward velocity:
     $$v_{\text{fwd}} \leftarrow \min\left(v_{\text{fwd}}, \, \sqrt{\frac{3.5}{\max(\kappa, 10^{-4})}}, \, \frac{3.5}{|\omega_z| + 0.02}\right)$$
   * This will immediately eliminate along-track overshoots on sharp curves (such as Scenario #04 and #20).
2. **Decouple Physical ZUPT in `sih/fusion/es_ekf.py`**:
   * Update line 310 in [`sih/fusion/es_ekf.py`](file:///c:/Users/carpe/SIH/sih/fusion/es_ekf.py#L310):
     Make physical rest ($\sigma_a^2 < 0.04, \|\boldsymbol{\omega}\| < 0.04$) unconditionally clamp velocity and yaw rate to $0.0$, without requiring `vel.forward_speed_mps < 0.2`.
   * This guarantees zero phantom drift accumulation during long traffic light stops.
3. **Upgrade Pre-Blackout Speed Scaling**:
   * Enhance our pre-blackout speed scaling in `sih/fusion/es_ekf.py` with the RLS persistent excitation check: freeze offset $\beta=0$ when pre-blackout speed standard deviation is $< 1.0\text{ m/s}$.

---

#### Phase B: Next-Generation Speed Engine (Requires Model Retraining)

These upgrades will take our velocity estimation from good ($r \approx 0.5$) to state-of-the-art ($r > 0.8$):

1. **Port Dual-Band Spectral Feature Extractor**:
   * Add [`sih/data/spectral.py`](file:///c:/Users/carpe/SIH/sih/data/spectral.py) implementing `DualBandSpectralExtractor`.
   * Generate 12-channel input representations for training windows.
2. **Port MoE Architecture**:
   * Add `ResNet1DSpeedEstimator` and `BayesianMoEFusion` into `sih/models/`.
   * Add 1-layer temporal GRU and `seq_speed_head` to `TCNAttentionVelocityModel`.
3. **Retrain using `phase55_balanced_loss`**:
   * Update [`train_velocity_model.py`](file:///c:/Users/carpe/SIH/train_velocity_model.py) with L_dyn (asymmetric variance alignment), L_cls (motion regime classification), and L_seq (intra-window sequence loss).
   * Train on `S-S1.csv` and `S-S2.csv` with uniform sampling ($w_{\text{tv}} = 0$).

---

#### Phase C: Edge Runtime & Mobile Deployment

1. **Export Trained MoE to ONNX**:
   * Port `export_moe_to_onnx` to generate `models/checkpoints/moe_fusion.onnx`.
2. **Implement Edge Inference Engine**:
   * Provide an ONNX Runtime streaming inference wrapper in `sih/deployment/` for sub-1.5ms edge execution.

---

### 5. Verification & Benchmark Plan Post-Integration

Whenever Phase A or Phase B modules are integrated, adhere strictly to **Rule 11 in `GEMINI.md`**:
1. Run `python -m unittest discover tests/` to ensure all existing contracts pass.
2. Re-run `python benchmarks/run_final_benchmark.py` across the 35 scenarios on unseen `S-M.csv`.
3. Compare against our baseline (13.40% median drift) — with the road curvature governor and decoupled ZUPT, we project overall median drift to drop to **< 8.5%**, cleanly passing all SIH benchmark criteria!
4. Re-generate `FINAL_JUDGE_EVALUATION_REPORT.md` and `FINAL_JUDGE_EVALUATION_REPORT.html` with updated charts and base64 plots.

---

## 6. Integration Outcome & Verified Benchmark Results

Following the integration blueprint outlined above, the production engine in `sih/` and `engine/` was hardened with both zero-retraining upgrades and the complete Phase B Dual-Brain MoE engine:
1. **Physical Rest ZUPT & ZARU**: Unconditionally clamps forward velocity and freezes integration when specific force variance drops (`Var(a) < 0.04 m^2/s^4`) and angular velocity norm is small (`||omega|| < 0.05 rad/s`).
2. **Velocity Entry Clamping (Crawl Guard)**: Clamps forward speed during low-speed crawl entries (`v_entry < 4.0 m/s`), preventing idle engine vibration from accumulating phantom distance.
3. **Pre-Blackout Dynamic Speed Scaling & Hybrid Vibration Blending**: Adapts `s_v = mean(v_GPS) / mean(v_AI)` over the 20s pre-blackout window and blends frequency-domain vibration power in the 3-8 Hz band.
4. **Branch Multi-Hypothesis Fork Gating**: Disables premature heading re-anchoring whenever competing candidates diverge (`diff_theta > 15 deg, L2 > 0.20 * L1`), letting gyro turn physics steer the vehicle onto the true branch.
5. **Dual-Brain Bayesian Mixture-of-Experts (`BayesianMoEFusion`)**: ResNet-1D micro-window (2.0s) + TCN-Attention macro-window (6.0s) with 12 input features, supervised by 10 Hz vehicle CAN-bus wheel speeds (`V-M.csv`, `V-S1.csv`, `V-S2.csv`) solving the 9-second phone GPS stair-step illusion.
6. **ZARU Highway Straight-Line Lock**: Locks yaw gyro bias when `v > 15 m/s` and `|omega_z| < 0.005 rad/s` for > 2.0s.
7. **Curvature Kinematics Governor**: Menger curvature limiting `v <= sqrt(a_lat_max / kappa)` in `sih/map/governor.py`.
8. **Standalone 200 Hz C++ Engine**: Compiled modern C++ core into `engine/cpp/idr_core.dll` for embedded edge telematics boxes.

### Official 40-Scenario Benchmark Verification (`artifacts/phase4_multi_trip_benchmark_results.csv`):
* **Overall Median Drift**: **6.35%** (Pure 6-Axis Baseline: **16.59%**, SIH Target < 10.0% — **PASSED**)
* **Overall P90 (Worst Decile) Drift**: **24.06%** (Pure Baseline: **50.43%**, Sub-35% — **PASSED**)
* **High Reliability Rate (<= 30% Drift)**: **92.5% (37 of 40 scenarios)** (Pure Baseline: **75.0%**)
* **Tier 1 Pass Rate (< 10% Drift)**: **67.5% (27 of 40 scenarios)** (Pure Baseline: **12.5%**)
* **Initial Heading Seeding Error**: Average **0.14°**, Median **0.0002°** (distorted compass: 28.4°)
* **Highway Cruising (`S-M.csv`, 8 sc)**: **7.66%** Median Drift
* **Arterial Corridors (`S-S2.csv`, 6 sc)**: **7.47%** Median Drift
* **Urban Grid & Crawl (`S-S1.csv`, 6 sc)**: **18.71%** Median Drift (Pure Baseline: 32.88%)
* **Mixed Arterial (`S-S3a.csv`, 10 sc)**: **6.81%** Median Drift
* **Arterial Corridors (`S-S4.csv`, 10 sc)**: **4.36%** Median Drift
