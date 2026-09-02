# Smartphone Intelligent Dead Reckoning (SIH): Benchmark Tuning & Architecture Assessment Report

## 1. Technical Explanation: Attempt 4 vs. Final Production

Attempt 4 (dynamic speed scale factor added) regressed median drift to **100.67%**, whereas Final Production achieved **72.51%** median drift (**71.15%** when speed scaling is isolated off). Below is the complete line-by-line breakdown of every parameter, formula, and logic difference between the two configurations:

| Technical Component | Attempt 4 Configuration | Final Production Configuration | Direct Performance Impact |
| :--- | :--- | :--- | :--- |
| **Blackout Entry Heading Fix** | Filter heading was initialized 30s prior to blackout and smoothed open-loop through noisy pre-blackout GNSS bearing updates (`0.15 * y_hdg`), leaving residual $4^\circ - 8^\circ$ heading errors at $t_{\text{bo\_start}}$. | At $t = t_{\text{bo\_start}}$, filter heading $\psi$ is explicitly reset to the exact last valid GNSS course-over-ground bearing (`ekf._heading_rad = radians(g.bearing_deg)`). | **Primary Driver**: Eliminates initial heading error at blackout entry, dropping transverse position error. |
| **Straight-Line Gyro Bias Correction ($\delta b_g^z$)** | Gyro bias state updates from GNSS Kalman updates were completely frozen (`_bg` remained `0`). | Added pre-blackout straight-line gyro bias learning: when `abs(y_hdg) < 3.0°` and `speed > 3.0 m/s`, `_bg[2] += 0.02 * y_hdg`. | Calibrates Earth-vertical MEMS gyro bias from pre-blackout GNSS course rate, preventing quadratic yaw integration drift. |
| **Speed Scale Initialization** | `initial_speed_scale` was hardcoded to `3.40` globally. | `initial_speed_scale` defaults to `1.00` and adapts dynamically only when valid GNSS speeds are observed. | Prevents over-scaling speed on urban/suburban segments where AI velocity predictions were already accurate. |
| **Outage Displacement Error Computation** | Measured total cumulative position error relative to global ENU origin from $t=0$. | Evaluates exact blackout displacement vector difference: $\|(\mathbf{p}_{\text{est\_end}} - \mathbf{p}_{\text{est\_start}}) - (\mathbf{p}_{\text{gt\_end}} - \mathbf{p}_{\text{gt\_start}})\|$. | Isolates position drift accumulated strictly *during* the blackout window. |

---

## 2. Mandatory Distance Statement

> **At 72.51% median drift, the filter's typical positioning error remains 7.25 times higher than (62.51 percentage points away from) the true competition benchmark target of <10% drift.**

---

## 3. Isolated Single-Parameter Tuning Matrix (16 Experiments)

Every experiment below changed **exactly one parameter at a time** from the baseline reference (`turn_th=1.5°/s, max_bg=0.5°/s, cool=0.5s, spd_mode=dynamic, r_nhc=0.20`):

| Exp ID | Parameter Changed | Value Tested | Combined Median Drift % | Combined Worst-Case Drift % | Benchmark Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ref** | Baseline Reference | Default | **72.51%** | **189.20%** | Baseline |
| **A.1** | `max_gyro_bias` | $\pm 0.30^\circ/\text{s}$ | 72.64% | 189.20% | Regressed (+0.13%) |
| **A.2** | `max_gyro_bias` | $\pm 0.40^\circ/\text{s}$ | 73.45% | 189.20% | Regressed (+0.94%) |
| **A.3** | `max_gyro_bias` | $\pm 0.50^\circ/\text{s}$ | **72.51%** | **189.20%** | **Optimal Clip** |
| **A.4** | `max_gyro_bias` | $\pm 0.60^\circ/\text{s}$ | 72.79% | 189.20% | Slight Regression |
| **A.5** | `max_gyro_bias` | $\pm 0.75^\circ/\text{s}$ | 72.79% | 189.20% | Slight Regression |
| **A.6** | `max_gyro_bias` | $\pm 1.00^\circ/\text{s}$ | 72.79% | 189.20% | Slight Regression |
| **B.1** | `speed_scale_mode` | `off` | **71.15%** | 196.39% | **Best Median (-1.36%)** |
| **B.2** | `speed_scale_mode` | `static_1.0` | 71.15% | 196.39% | Identical to `off` |
| **B.3** | `speed_scale_mode` | `static_3.4` | 329.44% | 612.24% | Severe Failure |
| **B.4** | `speed_scale_mode` | `dynamic` | 72.51% | **189.20%** | **Best Worst-Case** |
| **C.1** | `r_nhc` (lateral noise) | 0.05 | 72.51% | 189.20% | Invariant |
| **C.2** | `r_nhc` (lateral noise) | 0.10 | 72.51% | 189.20% | Invariant |
| **C.3** | `r_nhc` (lateral noise) | 0.20 | 72.51% | 189.20% | Invariant |
| **C.4** | `r_nhc` (lateral noise) | 0.50 | 72.51% | 189.20% | Invariant |
| **C.5** | `r_nhc` (lateral noise) | 1.00 | 72.51% | 189.20% | Invariant |

---

## 4. Benchmark Performance & Trajectory Map Visualizations

### Sensitivity & 50-Scenario Distributions
![Isolated Single-Parameter Sensitivity Analysis](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/isolated_experiments_benchmark.png)

![50 Blackout Scenarios Performance Comparison](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/50_scenarios_benchmark_comparison.png)

### 2D Spatial Trajectory Maps (Blue GT Fixes, Black Outage Dots, Red Dead Reckoning Fixes)

#### A. Trip S-S1: 60s GNSS Blackout Outage (at t=300s)
![Trip S-S1 60s Blackout Trajectory](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/map_s_s1_60s_blackout.png)

#### B. Trip S-S1: 30s GNSS Blackout Outage (at t=120s)
![Trip S-S1 30s Blackout Trajectory](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/map_s_s1_30s_blackout.png)

#### C. Trip S-S2 (Unseen Validation): 30s GNSS Blackout Outage
![Trip S-S2 30s Blackout Trajectory](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/map_s_s2_30s_blackout.png)

#### D. Worst-Case Outage (Scenario 2: 34s Outage at t=4810s, Reduced from 806% to 177% Drift)
![Worst-Case Outage Scenario 2 Trajectory](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/map_worst_case_rand_bo_02.png)

## 5. Honest Architecture Assessment & Tuning Headroom

### Can Hyperparameter Tuning Alone Reach <10% Median Drift?
**No.** Further tuning of ES-EKF noise covariances ($Q, R$), clip bounds, or thresholds within the current sensor-fusion architecture will **not** bring median drift from ~71% down to <10%. 

### Why the Current Architecture Has Hit a Structural Ceiling:
1. **Unconstrained Yaw Integration Noise**: Over a 60-second GNSS outage at highway speeds ($25\text{ m/s}$, distance $1500\text{m}$), an uncorrected gyro heading error of just **$1.0^\circ$** causes $\sin(1.0^\circ) \times 1500\text{m} = \mathbf{26.2\text{ meters}}$ of transverse drift. On short outages ($50-100\text{m}$), even a $0.5^\circ$ heading noise floor produces >25% relative drift error.
2. **Lack of Absolute Heading Reference During Outages**: MEMS smartphone gyroscopes suffer from dynamic bias instability and temperature drift during turns. Without an external orientation reference (magnetometer or road geometry), open-loop gyro integration during turns inevitably drifts.

### What Structurally Needs to be Built Next:
To reach the **<10% competition target**, the system structurally requires the higher-level pipeline layers:

1. **Map-Matching & Road Network Snap Engine (Phase 4)**:
   - Projecting EKF position estimates onto vectorized road network polylines (OpenStreetMap / GIS geometry) provides an absolute heading constraint during GNSS outages, constraining drift to the road lane.
2. **Zero Angular Rate Updates (ZARU) & Stop Detection**:
   - Explicitly locking heading during stationary traffic pauses or queueing prevents MEMS noise accumulation when stopped.
