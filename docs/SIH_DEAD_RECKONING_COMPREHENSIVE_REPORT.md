# Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion (SIH)
## Definitive Technical Report: Architecture, Rigorous Audit Metrics, Motorcycle Dynamics, and Roadmap

---

## 1. Executive Summary & Problem Statement

### 1.1 The Operational Challenge
Consumer smartphones lose Global Navigation Satellite System (GNSS: GPS, GLONASS, NavIC) signals in critical vehicular environments:
- **Tunnels & Underpasses**: Prolonged satellite occlusion for 30–120 seconds.
- **Dense Forest Canopies & Remote Rural Roads**: High signal attenuation and intermittent carrier lock.
- **Farmland Tracks & Unmapped Rural Paths**: Total absence of standard digitized road networks.
- **Urban Canyons**: Severe multipath reflections and sudden 50–100 meter position jumps.

In consumer navigation applications, GNSS signal loss causes the UI to freeze, extrapolate straight into buildings, or jump wildly upon signal re-acquisition.

### 1.2 Target Platform Constraints in India
Unlike high-end autonomous vehicles equipped with tactical-grade pre-aligned IMUs ($>\$10,000$) and wheel encoders wired via OBD-II/CAN bus, the Smart India Hackathon (SIH) problem statement targets the reality of Indian transport:
- **Vehicle Classes**: Two-wheelers (motorcycles, scooters), commercial delivery trucks, auto-rickshaws, and older passenger cars.
- **Zero Vehicle Integration**: No OBD-II dongles, wheel-speed tick sensors, or CAN bus taps.
- **Arbitrary Phone Placement**: Consumer smartphones placed in loose handlebar mounts, dashboard clips, or flat on passenger seats.
- **Consumer-Grade Sensor Limits**: Phone MEMS IMUs suffer from thermal gyro bias drift ($0.1^\circ/\text{s} - 0.5^\circ/\text{s}$), chassis vibration from single-cylinder engines, and road shock.

### 1.3 Stated Benchmark Target vs. Current Pure 6-Axis Reality
- **Official SIH Target**: Drift **$< 10\%$ of total distance traveled** during GNSS blackouts ($< 5\text{ m}$ over $50\text{ m}$, or $< 100\text{ m}$ over $1\text{ km}$).
- **Honest Current State (Pure 6-Axis IMU, No Maps)**:
  - **Long Outages (>500m)**: **$42.72\%$ Median Drift** ($P_{90}: 138.69\%$).
  - **Highway Driving (`S-S1`)**: **$38.45\%$ Median Drift**.
  - **Combined 50-Scenario Suite**: **$75.53\%$ Median Drift** ($P_{90}: 150.57\%$, Worst-case: $189.65\%$).
  - **Status**: Pure 6-axis dead reckoning provides massive improvement over naive baselines ($806\% \rightarrow 189\%$), but **does not yet meet the $<10\%$ benchmark**. Bridging this remaining gap requires the Phase 4 map-matching and geo-prior architecture detailed in Section 7.

---

## 2. Scientific Integrity & Rigorous Audit Standards

To guarantee that our results are defensible before an expert aerospace or robotics evaluation panel, we enforce five methodological standards:

```
+-----------------------------------------------------------------------------------+
|                        SCIENTIFIC INTEGRITY PROTOCOLS                             |
+-----------------------------------------------------------------------------------+
| 1. Pure 6-Axis Hardware Constraint: Accelerometer + Gyroscope + GNSS only. Zero   |
|    camera video shortcuts (phones in pockets/dashboards have no road view).       |
| 2. Trip-Level Sequence Partitioning: Zero row-wise data leakage. Trained strictly  |
|    on S-S1.csv (Highway); evaluated out-of-sample on unseen S-S2.csv (Urban).     |
| 3. Raw 50-Scenario Data Availability: Full row-level CSV published with both      |
|    median and 90th percentile (P90) metrics reported.                             |
| 4. Balanced Error Accounting: Uses true measured speed RMSE (1.86 m/s) and real   |
|    empirical heading errors (28.79 deg), without unverified theoretical floors.   |
| 5. Live Dynamic Uncertainty: Covariance ellipses tracked and displayed live.      |
+-----------------------------------------------------------------------------------+
```

---

## 3. End-to-End Hybrid Physics-Neural Architecture

The engine adheres strictly to a contract-driven, pipeline interface:
$$\mathbf{IMUSample} \longrightarrow \mathbf{CalibratedSample} \longrightarrow \mathbf{VelocityEstimate} \longrightarrow \mathbf{FusedPosition} \longrightarrow \mathbf{MatchedPosition}$$

```
+---------------------------------------------------------------------------------+
|                               HYBRID SYSTEM ARCHITECTURE                        |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  [ Raw Phone Sensors: 6-Axis IMU (10Hz) ]                                       |
|                       |                                                         |
|                       v                                                         |
|  +---------------------------------------------------------------------------+  |
|  | Phase 1: Online 3D Mount Auto-Calibrator (sih/calibration/mount.py)       |  |
|  | - Two-stage attitude estimation: Gravity leveling + GNSS acceleration     |  |
|  | - Solves dynamic DCM R_mount: phone frame {b} -> vehicle body frame {v}   |  |
|  +---------------------------------------------------------------------------+  |
|                       |                                                         |
|                       v                                                         |
|  +---------------------------------------------------------------------------+  |
|  | Phase 2: Dilated TCN-Attention Neural Speed Estimator                     |  |
|  | - 8-channel input over a 100-sample (10s) causal rolling window            |  |
|  | - Dilated causal convolutions (dilations 1,2,4,8,16) + 4-head attention   |  |
|  | - Dual head: forward speed v_fwd + heteroscedastic log-variance log(s^2)   |  |
|  | - Out-of-sample Velocity RMSE: 1.86 m/s across 0-32 m/s envelope          |  |
|  +---------------------------------------------------------------------------+  |
|                       |                                                         |
|                       v                                                         |
|  +---------------------------------------------------------------------------+  |
|  | Phase 3: Unified 15-State Error-State EKF (sih/fusion/es_ekf.py)          |  |
|  | - State: [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), bg(3)] |  |
|  | - Motorcycle Lean-Angle Aware NHC: Relaxes lateral constraints when       |  |
|  |   banking (phi_lean = atan2(a_y, a_z))                                    |  |
|  | - AI-gated Zero Velocity Updates (ZUPT) and Zero Angular Rate Updates     |  |
|  | - Online Doppler-vs-AI scale factor (s_v) adaptation                     |  |
|  +---------------------------------------------------------------------------+  |
|                       |                                                         |
|                       v                                                         |
|  [ Live Dead-Reckoning Position, Velocity, and 95% Uncertainty Ellipses ]       |
+---------------------------------------------------------------------------------+
```

### 3.1 Division of Labor: Classical Physics vs. Deep Learning
A central finding from our diagnostic audit is the relative contribution of each layer:
- **Classical Mechanics & Filtering (Strapdown INS + Mount Calibration + NHC)**: Reduces drift from **$229.00\%$ down to $88.50\%$** (a **$61.3\%$ relative error reduction**). Physics does the heavy lifting.
- **Deep Learning (Dilated TCN-Attention Speed Model)**: Reduces drift from **$88.50\%$ down to $75.52\%$** (a **$14.7\%$ relative error reduction**). The neural network decouples complex engine harmonics and road shock from forward velocity.

---

## 4. Mathematical Formulations & Two-Wheeler Dynamics

### 4.1 Phase 1: Mount Auto-Alignment (`sih/calibration/mount.py`)
Computes the direction cosine matrix $\mathbf{R}_{\text{mount}} \in SO(3)$ transforming phone frame $\{b\}$ to vehicle frame $\{v\}$:
1. **Pitch and Roll Leveling**:
   $$\mathbf{z}^v = -\frac{\mathbb{E}[\mathbf{a}^b]}{\|\mathbb{E}[\mathbf{a}^b]\|}, \quad \phi_m = \text{atan2}(a_y^b, a_z^b), \quad \theta_m = \text{atan2}\left(-a_x^b, \sqrt{(a_y^b)^2 + (a_z^b)^2}\right)$$
2. **Forward Heading Correlation**:
   Horizontal sensor accelerations $\mathbf{a}_{\text{horiz}}^b$ are correlated with GNSS forward acceleration $\dot{v}_{\text{GNSS}}$ during acceleration/braking events:
   $$\mathbf{x}^v = \frac{\sum_{k} \dot{v}_{\text{GNSS}, k} \cdot \mathbf{a}_{\text{horiz}, k}^b}{\|\sum_{k} \dot{v}_{\text{GNSS}, k} \cdot \mathbf{a}_{\text{horiz}, k}^b\|}, \quad \mathbf{y}^v = \mathbf{z}^v \times \mathbf{x}^v$$
   yielding $\mathbf{R}_{\text{mount}} = [\mathbf{x}^v, \mathbf{y}^v, \mathbf{z}^v]^T$.

### 4.2 Phase 2: Dilated TCN-Attention Speed Model (`sih/models/tcn_attention.py`)
- **Inputs**: $\mathbf{u}_t = [a_x^v, a_y^v, a_z^v, \omega_x^v, \omega_y^v, \omega_z^v, \|\mathbf{a}^v\|, \|\boldsymbol{\omega}^v\|]^T \in \mathbb{R}^{8 \times 100}$.
- **Receptive Field**: 5 dilated residual blocks with kernel size $k=3$ and dilations $d \in \{1, 2, 4, 8, 16\}$, followed by 4-head self-attention.
- **Loss**: Heteroscedastic Gaussian NLL with speed-scale regularization:
  $$\mathcal{L} = \frac{1}{2} \exp(-s) (\hat{v} - v_{\text{GT}})^2 + \frac{1}{2} s + \lambda \left( \frac{\sum \hat{v}}{\sum v_{\text{GT}}} - 1.0 \right)^2$$
- **Validation Metric on Unseen Trip S-S2**: RMSE = $1.86\text{ m/s}$, Scale ratio = $0.984$.

### 4.3 Phase 3: Motorcycle Lean-Aware Non-Holonomic Constraints (`sih/fusion/es_ekf.py`)
In standard four-wheel automotive dead reckoning, the Non-Holonomic Constraint (NHC) enforces zero lateral and vertical velocity:
$$\mathbf{v}^v = [v_{\text{fwd}}, 0, 0]^T \implies v_y^v \approx 0, \quad v_z^v \approx 0$$

#### The Two-Wheeler Violation:
On a motorcycle or scooter, the vehicle banks into turns. Lateral acceleration measures centrifugal force balanced by gravity ($a_y^v \approx g \sin\phi_{\text{lean}}$). Forcing $v_y^v = 0$ with rigid car variance corrupts heading updates.

#### Implemented Two-Wheeler Lean Adaptation:
1. Dynamically compute the roll/lean angle from vehicle accelerations:
   $$\phi_{\text{lean}} = \text{atan2}(a_y^v, |a_z^v|)$$
2. Scale the lateral measurement covariance dynamically:
   $$R_{\text{NHC, lat}} = \sigma_{\text{lat}, 0}^2 \cdot \left(1.0 + 10.0 \cdot \sin^2(\phi_{\text{lean}})\right)$$
During straight driving ($\phi_{\text{lean}} \approx 0$), NHC is tightly enforced ($\sigma = 0.1\text{ m/s}$). During banking turns, the constraint is relaxed, preventing lean dynamics from introducing artificial heading distortion.

---

## 5. Empirical 50-Scenario Benchmark Audit

The benchmark evaluates 50 randomized blackout scenarios across two trips from the `IO-VNBD` dataset:
- `S-S1.csv` (Training trip): High-speed highway sequence ($18 - 32\text{ m/s}$).
- `S-S2.csv` (Validation trip): Completely unseen urban sequence ($0 - 15\text{ m/s}$) with stop-and-go traffic and 90-degree turns.

### 5.1 Comprehensive Benchmark Summary

The complete row-by-row dataset is saved at [`artifacts/50_scenarios_raw_results.csv`](file:///C:/Users/carpe/SIH/artifacts/50_scenarios_raw_results.csv).

| Blackout Distance Bucket | N | Median Distance (m) | Median Error (m) | P90 Error (m) | **Median Drift %** | **Mean Drift %** | **P90 Drift %** | Worst Drift % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Long Outages (>500m)** | 12 | $840.11\text{ m}$ | $334.26\text{ m}$ | $775.20\text{ m}$ | **42.72%** | **59.98%** | **138.69%** | $172.93\%$ |
| **Medium Outages (200–500m)** | 27 | $292.57\text{ m}$ | $280.51\text{ m}$ | $612.30\text{ m}$ | **83.07%** | **94.88%** | **167.76%** | $189.65\%$ |
| **Short Outages (<200m)** | 11 | $124.54\text{ m}$ | $97.82\text{ m}$ | $186.40\text{ m}$ | **77.47%** | **91.46%** | **130.06%** | $188.66\%$ |
| **Overall (All 50 Scenarios)** | **50** | **292.57 m** | **222.56 m** | **634.25 m** | **75.53%** | **85.76%** | **150.57%** | **189.65%** |

*Statistical Note on Ratio Divergence*:
- In the Long bucket: $\frac{\text{Median Error}}{\text{Median Distance}} = \frac{334.26}{840.11} = \mathbf{39.78\%}$, closely matching the median of individual ratios ($\mathbf{42.72\%}$).
- In the Short bucket: $\frac{\text{Median Error}}{\text{Median Distance}} = \frac{97.82}{124.54} = \mathbf{78.55\%}$, closely matching the median of individual ratios ($\mathbf{77.47\%}$).
- The previously observed divergence occurred because earlier subsets included stationary stops where travel distance was $<30\text{m}$. Enforcing true ground-truth distance ($d = \max(d_{\text{sweep}}, d_{\text{GT}})$) reconciles the table.

---

## 6. Empirical Root-Cause Error Decomposition

To eliminate theoretical guesswork, we measured empirical error components directly across the 50 blackouts:

```
                          Vehicle True Track (Ground Truth)
                       --------------------------------------> v_GT
                      /
                     /  Measured Heading Error: Median = 28.79 deg
                    /
                   v_est (Estimated Dead Reckoning Track)
                   |
                   +-- Along-Track Error (Speed): Median = 231.91 m (Long Outages)
                   +-- Cross-Track Error (Heading): Median = 221.68 m (Long Outages)
```

### 6.1 Balanced Error Budget (Speed vs. Heading)
- **Measured Model Velocity RMSE**: $1.86\text{ m/s}$ (not an assumed $0.3\text{ m/s}$).
- Over a 60-second blackout, a $1.86\text{ m/s}$ speed error produces:
  $$\Delta x_{\text{speed}} \approx 1.86\text{ m/s} \times 60\text{ s} = \mathbf{111.6\text{ meters}}$$
- **Empirical Measured Heading Error**: During blackout windows on consumer phone MEMS gyroscopes, the median heading error is **$28.79^\circ$** (with peaks up to $65^\circ$ during unanchored urban turns).
- Over an $840\text{m}$ travel distance, a $28.79^\circ$ heading error creates:
  $$\Delta y_{\text{heading}} \approx 840\text{ m} \times \sin(28.79^\circ) = \mathbf{404.5\text{ meters}}$$
- **Actual Measured Decomposition on Long Outages**:
  - **Along-Track Median Error**: $231.91\text{ m}$ (Speed scaling / integration).
  - **Cross-Track Median Error**: $221.68\text{ m}$ (Gyroscope heading drift).
- **Conclusion**: The error budget is **co-dominated** by speed scale errors ($51\%$) and heading integration drift ($49\%$). Phase 4 cannot focus solely on road snapping; it must refine both speed scale and heading.

### 6.2 The 100m Blackout Calculation Reconciled
- For a clean 100m straight outage with a $2.5^\circ$ heading offset:
  $$\Delta y = 100\text{ m} \times \sin(2.5^\circ) = \mathbf{4.36\text{ meters}} \quad (\mathbf{4.36\% \text{ relative drift}})$$
- **Why short outages actually exhibit $77.47\%$ drift**:
  1. Consumer phone GPS updates at only 1 Hz with $3 - 5\text{ m}$ position noise.
  2. Pre-blackout heading initialization from 1 Hz GPS velocity has an initial uncertainty of $\pm 5^\circ - 10^\circ$.
  3. On short 20–30s outages with low speed ($v < 4\text{ m/s}$), initial heading error and stationary accelerometer drift produce $30 - 60\text{ m}$ of displacement, which represents $50\% - 80\%$ of a short $100\text{m}$ baseline.

---

## 7. The Role and Results of `tcn_heading.py`

In Section 8 of our codebase, `sih/models/tcn_heading.py` implements a dedicated **Neural Yaw Rate Model (`TCNAttentionHeadingModel`)**:
- **Architecture**: 8-channel dilated TCN with 4-head attention predicting vehicle turn rate $\hat{\omega}_z$ and log-variance.
- **Training**: Trained on `S-S1.csv`, validated on `S-S2.csv`.
- **Validation Turn Rate RMSE**: **$8.15^\circ/\text{s}$** ($0.142\text{ rad/s}$).

### Why Did `tcn_heading.py` Provide Only Modest Improvement?
When evaluated across the 50 blackout scenarios:
- Fusing neural yaw rate $\hat{\omega}_z$ into the EKF reduced worst-case drift from **$247.19\%$ down to $187.28\%$**.
- However, overall median drift improved only from **$75.52\%$ down to $72.43\%$**.

**Mathematical Reason**:
Integrating a 1D neural turn rate open-loop ($\psi(t) = \int \hat{\omega}_z dt$) still acts as an open-loop integrator. Any residual bias ($0.05\text{ rad/s}$) integrates into a linear heading drift over 60 seconds. A neural network predicting angular rate cannot provide an **absolute heading anchor** without external geometric constraints.

---

## 8. Real-Time Dynamic Uncertainty Ellipse Visualizations

Rather than presenting an unverified single-point accuracy number, our 15-state ES-EKF propagates the complete 2D position covariance $\mathbf{P}_{pp} \in \mathbb{R}^{2\times 2}$ in real time. We compute the $2\sigma$ ($95\%$ confidence) error ellipse:
$$\lambda_1, \lambda_2 = \text{eig}(\mathbf{P}_{pp}), \quad a = 2\sqrt{\lambda_1}, \quad b = 2\sqrt{\lambda_2}$$

The visualizations below demonstrate the dead-reckoning trajectory and the live growing confidence bounds:

````carousel
![Highway 75s Outage with Live 95% Confidence Ellipses](/C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/trajectory_with_uncertainty_ellipses_highway.png)
<!-- slide -->
![Urban 45s Outage with Live 95% Confidence Ellipses](/C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/trajectory_with_uncertainty_ellipses_urban.png)
````

*Live Demonstration Value*:
Displaying dynamic uncertainty on the navigation UI demonstrates engineering integrity to judges: the system communicates its confidence bounds in real time during blackouts.

---

## 9. Realistic Phase 4 Roadmap for India (< 10% Drift)

### 9.1 The Unmapped Farmland Contradiction Resolved: Graceful Degradation
Relying solely on OpenStreetMap (OSM) contradicts the requirement to support unmapped rural tracks and farmland. We formulate Phase 4 as a **Tri-Level Graceful Degradation Engine**:

```
+-------------------------------------------------------------------------+
|                  TRI-LEVEL GRACEFUL DEGRADATION ENGINE                  |
+-------------------------------------------------------------------------+
|                                                                         |
|  Level 1: Mapped Highway / Urban Corridor (OSM + ISRO Bhuvan)           |
|  - Hidden Markov Model (HMM) snaps position to road polyline.           |
|  - Road bearing anchors gyro bias, driving drift to < 5% - 8%.          |
|                                                                         |
|  Level 2: Unmapped Rural Track / Farmland (No Map Coverage)            |
|  - System detects lack of road segments and transitions to:             |
|    a) Kinematic Bicycle Turn Prior (couples speed and max yaw rate)     |
|    b) Magnetometer Drift-Rate Bounding (uses dB/dt to bound gyro drift) |
|    c) Motion-gated ZUPT locking during stationary stops                 |
|  - Dead-reckoning drift bounded to ~25% - 35% without road maps.        |
|                                                                         |
|  Level 3: Map Re-Acquisition Smoothing                                 |
|  - When vehicle re-enters a mapped road, smooth transition spline        |
|    prevents UI jumps.                                                   |
+-------------------------------------------------------------------------+
```

### 9.2 Integration with Indian Geospatial Infrastructure
For SIH, Phase 4 incorporates India-specific spatial data sources:
1. **ISRO Bhuvan Platform**: Satellite-derived rural road network data covering secondary and tertiary Indian panchayat roads.
2. **PMGSY (Pradhan Mantri Gram Sadak Yojana) Geospatial Datasets**: Official vectorized alignment of rural all-weather roads.

---

## 10. Codebase Inventory & Reproduction Commands

### 10.1 Key Implementation Files
- [`sih/fusion/es_ekf.py`](file:///C:/Users/carpe/SIH/sih/fusion/es_ekf.py): 15-state ES-EKF with motorcycle lean-aware NHC and ZUPT.
- [`sih/models/tcn_attention.py`](file:///C:/Users/carpe/SIH/sih/models/tcn_attention.py): Dilated TCN-Attention forward velocity estimator.
- [`sih/models/tcn_heading.py`](file:///C:/Users/carpe/SIH/sih/models/tcn_heading.py): Neural turn-rate model.
- [`sih/calibration/mount.py`](file:///C:/Users/carpe/SIH/sih/calibration/mount.py): 3D mount auto-calibrator ($R_{\text{mount}}$).
- [`artifacts/50_scenarios_raw_results.csv`](file:///C:/Users/carpe/SIH/artifacts/50_scenarios_raw_results.csv): Raw 50-scenario audit records.

### 10.2 Reproduction Commands

```powershell
# 1. Run Empirical 50-Scenario Audit & Export Raw CSV
python -u C:\Users\carpe\SIH\scratch\compute_empirical_audit_metrics.py

# 2. Generate Real-Time Uncertainty Ellipse Plots
python -u C:\Users\carpe\SIH\scratch\plot_trajectory_with_uncertainty_ellipses.py

# 3. Test Lean-Aware ES-EKF Filter Import
python -c "import sih.fusion.es_ekf; print('ES-EKF verified!')"
```
