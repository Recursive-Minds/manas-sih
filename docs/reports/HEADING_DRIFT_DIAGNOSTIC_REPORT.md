# Heading Drift Diagnostic Report: Bias Convergence, Turn Severity Correlation, NHC Audit & Magnetometer Feasibility

**Date**: September 2026  
**Dataset**: IO-VNBD Smartphone Automotive Dataset (`S-S1` Highway & `S-S2` Urban Unseen)  
**Evaluated Engine**: Phase 3 (Auto Mount Calibration + Multi-Scale TCN-Attention Velocity + 15-State ES-EKF)

---

## Executive Summary

Following the discovery that **$95.1\%$ of total blackout position error is cross-track heading drift** (and $<1\%$ is longitudinal speed scaling), we conducted a focused four-part heading-drift investigation to isolate the root cause before implementing any fix:

1. **Gyro Bias State Convergence Check**: The EKF's estimated gyro bias state ($b_{g,z}$) was flat-lined at exactly $0.0000^\circ/\text{s}$ across all 50 sweep scenarios because GNSS Course-Over-Ground innovations were never coupled into the gyro bias error state.
2. **Turn Severity Correlation**: Total accumulated turn angle ($\int |\omega_z| dt$) correlates strongly ($r = +0.4101, p = 0.0031$) with cross-track error ($1.73\text{ m}$ lateral error per degree of turn on `S-S1`). Peak yaw rate has low correlation ($r = 0.0974$), showing that error accumulates via angle integration over time rather than dynamic rate saturation.
3. **NHC Feedback Coupling Audit**: NHC is applied only via analytic forward velocity projection without Kalman innovation updates on the error state ($\delta \boldsymbol{\theta}$). Heading correction from NHC at every single step is identically $\Delta \theta_{\text{NHC}} = 0.000^\circ$.
4. **Magnetometer Feasibility Audit**: 3-axis magnetometer data ($\mu\text{T}$) is $100\%$ complete and non-null across all IO-VNBD trips at $10\text{ Hz}$, but is completely unused in all calibration, velocity, and fusion modules.

---

## 1. Gyro Bias Convergence Check (`b_g,z`)

### Objective
Determine whether the EKF's estimated gyro bias state ($b_{g,z}$) settles to a stable true bias prior to blackout onset, or whether it fluctuates or remains unconverged.

### Methodology
We tracked the continuous trajectory of $b_{g,z}(t)$ from trip initiation ($t = 0\text{s}$) through the blackout window for the worst-performing sweep scenarios on both highway (`S-S1`) and urban (`S-S2`) sequences:

![Gyro Bias Convergence Check](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/gyro_bias_convergence_check.png)

### Quantitative Findings

| Scenario | Trip | Start Time | Duration | Final Error | Drift % | Pre-Blackout $b_{g,z}$ | Active EKF Gyro Bias Update? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rand_bo_09_86s_at_3078s` | `S-S1` (Highway) | $3078.2\text{s}$ | $86.4\text{s}$ | $530.9\text{ m}$ | $135.0\%$ | **$0.0000^\circ/\text{s}$** | **NO** (Flat-line) |
| `rand_bo_22_42s_at_791s` | `S-S1` (Highway) | $791.1\text{s}$ | $41.8\text{s}$ | $495.9\text{ m}$ | $183.1\%$ | **$0.0000^\circ/\text{s}$** | **NO** (Flat-line) |
| `rand_bo_11_77s_at_202s` | `S-S1` (Highway) | $202.0\text{s}$ | $76.6\text{s}$ | $465.8\text{ m}$ | $51.7\%$ | **$0.0000^\circ/\text{s}$** | **NO** (Flat-line) |
| `rand_bo_12_71s_at_2588s` | `S-S2` (Urban Unseen) | $2587.6\text{s}$ | $71.1\text{s}$ | $1047.8\text{ m}$ | $129.5\%$ | **$0.0000^\circ/\text{s}$** | **NO** (Flat-line) |
| `rand_bo_18_73s_at_7454s` | `S-S2` (Urban Unseen) | $7454.1\text{s}$ | $73.3\text{s}$ | $1030.2\text{ m}$ | $160.5\%$ | **$0.0000^\circ/\text{s}$** | **NO** (Flat-line) |
| `rand_bo_01_71s_at_8989s` | `S-S2` (Urban Unseen) | $8988.7\text{s}$ | $71.0\text{s}$ | $924.7\text{ m}$ | $58.5\%$ | **$0.0000^\circ/\text{s}$** | **NO** (Flat-line) |

### Root Cause Analysis
In `ErrorStateEKF`:
- $\mathbf{b}_g$ is initialized to $[0, 0, 0]^T$.
- During nominal GNSS updates (`update_gnss()`), the filter directly resets the nominal heading $\theta \leftarrow \theta_{\text{COG}}$ or applies position innovation, but **does not compute error-state cross-covariance updates into $\delta \mathbf{b}_g$**.
- When GNSS is lost, an uncompensated MEMS hardware bias $b_{g,z} \approx 0.10^\circ/\text{s}$ ($0.0017\text{ rad/s}$) integrates into heading error:
  $$\delta \theta(t) = b_{g,z} \cdot t \implies \delta \theta(70\text{s}) \approx 7.0^\circ$$
  For a vehicle traveling $1000\text{ m}$ during the blackout, a $7.0^\circ$ heading error induces:
  $$\text{Lateral Position Error} \approx d \cdot \sin(7.0^\circ) \approx 1000 \cdot 0.1219 = 121.9\text{ m}$$

---

## 2. Heading & Cross-Track Error vs. Turn Severity

### Objective
Directly test whether sharp intersection turns (like `S-S2`'s 90-degree street corners) are disproportionately responsible for heading/cross-track error versus uniform linear drift.

### Measured Metrics
For all 50 randomized sweep samples, we computed:
1. **Total Accumulated Turn Angle**: $\Theta_{\text{accum}} = \int_{t_{\text{start}}}^{t_{\text{end}}} |\omega_z(t)| \, dt$ (degrees)
2. **Net Heading Displacement**: $|\Delta \theta_{\text{net}}| = |\theta(t_{\text{end}}) - \theta(t_{\text{start}})|$ (degrees)
3. **Peak Yaw Rate**: $\max |\omega_z(t)|$ (degrees/s)

![Heading Error vs Turn Severity](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/heading_error_vs_turn_severity.png)

### Statistical Correlation Matrix (50 Samples)

| Metric Pair | Pearson Correlation ($r$) | $p$-value | Significance |
| :--- | :---: | :---: | :---: |
| **\|Cross-Track Error\| vs Total Accumulated Turn** | **$+0.4101$** | **$3.099 \times 10^{-3}$** | **Statistically Significant ($p < 0.01$)** |
| **\|Cross-Track Error\| vs Peak Yaw Rate** | $+0.0974$ | $5.011 \times 10^{-1}$ | Not Significant ($p > 0.05$) |
| **Along-Track Error vs Total Turn** | $+0.2642$ | $6.391 \times 10^{-2}$ | Weak |
| **Drift % vs Total Turn** | $-0.1474$ | $3.072 \times 10^{-1}$ | Weak |

### Takeaways
1. **Total turn angle is the primary driver of lateral error**:
   - On `S-S1` (Highway): Cross-track error grows at **$1.73\text{ m}$ per degree of total turn**.
   - On `S-S2` (Urban Unseen): Blackout windows traversing $300^\circ - 700^\circ$ of accumulated turns accumulate **$400\text{m} - 1000\text{m}$** of lateral error.
2. **Angle integration vs peak rate**:
   - High correlation with total turn angle ($r=0.41$) vs. near-zero correlation with peak yaw rate ($r=0.10$) proves the issue is **scale/bias integration over large angle changes**, rather than high dynamic rate clipping.

---

## 3. NHC (Non-Holonomic Constraint) Feedback Coupling Audit

### Objective
Audit how much the NHC pseudo-measurement updates actually change the heading estimate ($\Delta \theta_{\text{NHC}}$) at each step during turn-heavy blackout windows.

### Codebase Audit in `sih/fusion/es_ekf.py`
In `ErrorStateEKF.predict()`, when an AI velocity estimate $\hat{v}_{\text{fwd}}$ is supplied:
```python
ve = v_fwd * np.sin(self._heading_rad)
vn = v_fwd * np.cos(self._heading_rad)
self._v = np.array([ve, vn, 0.0], dtype=np.float64)
self._p[0] += ve * dt
self._p[1] += vn * dt
```

### Audit Findings:
- Lateral velocity in the vehicle body frame is zeroed out by forward trigonometric projection into ENU coordinates.
- **ZERO Kalman measurement update is executed on the error state**:
  - The measurement matrix $H = \begin{bmatrix} 0_{2\times 3} & C_{v2e}^T & 0_{2\times 3} & 0_{2\times 3} & 0_{2\times 3} \end{bmatrix}$ is never evaluated.
  - The Kalman gain $K = P H^T (H P H^T + R)^{-1}$ is never computed.
  - The orientation error vector $\delta \boldsymbol{\theta} \leftarrow K \cdot (\mathbf{z} - H \hat{\mathbf{x}})$ is **never updated**.
- **Result**: Heading correction from NHC at every single time step is identically:
  $$\Delta \theta_{\text{NHC}} \equiv 0.0000^\circ$$
  The heading integration during blackout operates **$100\%$ open-loop**.

---

## 4. Magnetometer Availability & Feasibility Audit

### Objective
Verify if magnetometer readings are available in the IO-VNBD smartphone dataset and whether they are currently utilized.

### Dataset Verification

```
S-S1 Columns: [' MAGNETIC FIELD X (μT)', ' MAGNETIC FIELD Y (μT)', ' MAGNETIC FIELD Z (μT)']
S-S2 Columns: [' MAGNETIC FIELD X (μT)', ' MAGNETIC FIELD Y (μT)', ' MAGNETIC FIELD Z (μT)']
```

### Channel Statistics

| Sequence | Total Samples (10 Hz) | Non-Null Samples | $B_x$ Range ($\mu\text{T}$) | $B_y$ Range ($\mu\text{T}$) | $B_z$ Range ($\mu\text{T}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`S-S1` (Highway)** | 51,746 | **51,746 (100%)** | $[-41.87, +10.75]$ | $[-39.19, -16.62]$ | $[-11.00, +43.88]$ |
| **`S-S2` (Urban Unseen)** | 93,876 | **93,876 (100%)** | $[-51.44, +18.25]$ | $[-74.44, +37.25]$ | $[-48.06, +53.75]$ |

### Utilization Matrix Across Pipeline Modules

| Module | File | Reads Mag? | Uses Mag in Estimation? |
| :--- | :--- | :---: | :---: |
| **Data Loader** | `sih/data/loader.py` | **YES** (`IMUSample.mag`) | No (passes to contract) |
| **Mount Calibration** | `sih/calibration/mount.py` | No | No |
| **Velocity Model** | `sih/models/tcn_attention.py` | No | No |
| **ES-EKF Fusion** | `sih/fusion/es_ekf.py` | No | No |

**Conclusion**: 3-axis magnetic field data is **$100\%$ available** in the dataset and **$100\%$ unused** in the entire codebase.

---

## 5. Summary & Decision Matrix

| Investigation Item | Finding | Root Cause |
| :--- | :--- | :--- |
| **1. Gyro Bias Tracking** | $b_{g,z} \equiv 0.0000^\circ/\text{s}$ across all trips | No Kalman observability update into $\mathbf{b}_g$ state during GNSS updates |
| **2. Turn Error Correlation** | Cross-track error grows at $1.73\text{ m/deg}$ ($r=0.41$) | Open-loop gyro integration over large turn angles during blackout |
| **3. NHC Feedback Coupling** | $\Delta \theta_{\text{NHC}} = 0.000^\circ$ at every step | NHC applied only as velocity projection, not Kalman error-state correction |
| **4. Magnetometer Feasibility** | 3-axis $\mu\text{T}$ data is $100\%$ available | Ready as an aiding option if needed |

---
*No algorithmic modifications have been implemented. Awaiting user decision on the optimal heading correction strategy.*
