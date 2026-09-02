# Closed-Loop EKF & 50-Sample Sweep Verification Report

This report documents the implementation of the closed-loop Non-Holonomic Constraint (NHC) measurement update, observable gyro bias estimation ($b_{g,z}$), the 50-sample randomized blackout sweep evaluation, and the turn-severity correlation analysis.

---

## 1. Proper NHC Measurement Update Implementation

The open-loop velocity projection was replaced in [`es_ekf.py`](file:///c:/Users/carpe/SIH/sih/fusion/es_ekf.py) with a real closed-loop Extended Kalman Filter correction mapping lateral centripetal acceleration innovation to orientation error $\delta \theta$.

When the vehicle executes a turn with yaw rate $\omega_z$ and forward speed $v_{\text{fwd}}$, the physical centripetal acceleration experienced in the vehicle body right-axis (index 1) is:

$$\text{a}_{\text{lat, expected}} = -v_{\text{fwd}} \cdot \omega_z$$

The innovation between measured lateral acceleration $a_{\text{lat, meas}}$ and expected centripetal acceleration is used to compute a direct Kalman update on heading:

$$\Delta \theta_{\text{NHC}} = K_{\text{NHC}} \cdot (a_{\text{lat, meas}} - a_{\text{lat, expected}})$$

### Verification of $\Delta \theta_{\text{NHC}}$ Per Step
Across turn-heavy segments in the dataset, $\Delta \theta_{\text{NHC}}$ is now **actively non-zero** (previously identically $0.0000^\circ/\text{step}$):
- **Mean Correction during turns**: $0.0013^\circ - 0.0015^\circ / \text{step}$
- **Peak Correction during sharp turns**: $0.0084^\circ / \text{step}$ ($\approx 0.84^\circ/\text{s}$ at 100 Hz)

---

## 2. Gyro Bias Observability ($b_{g,z}$) During GNSS-Aided Periods

GNSS Course-Over-Ground (COG) bearing ($\theta_{\text{GNSS}} = \text{atan2}(v_E, v_N)$) was wired into the 2-state heading/bias Kalman sub-filter $\mathbf{x}_{\text{hdg}} = [\theta, b_{g,z}]^T$. 

The measurement matrix $\mathbf{H} = [1, 0]$ updates heading directly, while cross-covariance $P_{\theta, b_g}$ drives live estimation of $b_{g,z}$. In addition, a Zero Angular Rate Update (ZARU) is applied when stationary ($v < 0.2\text{ m/s}$).

### Observed $b_{g,z}$ Behavior
- **Before Fix**: $b_{g,z} \equiv 0.0000^\circ/\text{s}$ (flat-lined permanently).
- **After Fix**: $b_{g,z}$ actively converges during GNSS-aided driving:
  - **S-S1 (Highway)**: $b_{g,z}$ settles to $+0.0261^\circ/\text{s}$ to $+0.2452^\circ/\text{s}$ pre-blackout.
  - **S-S2 (Urban)**: $b_{g,z}$ settles between $-0.0377^\circ/\text{s}$ and $+0.0820^\circ/\text{s}$ pre-blackout.

![Gyro Bias Convergence](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/fixed_gyro_bias_convergence.png)

---

## 3. Re-Run of 50-Sample Randomized Sweep (Baseline vs. Fixed EKF)

The exact 50 randomized blackout scenarios from [`randomized_blackout_sweep_results.csv`](file:///c:/Users/carpe/SIH/artifacts/randomized_blackout_sweep_results.csv) were re-evaluated with the closed-loop EKF. Results are saved in [`fixed_sweep_results.csv`](file:///c:/Users/carpe/SIH/artifacts/fixed_sweep_results.csv).

### Summary Comparison Table

| Metric | Baseline (Old) | Fixed EKF (New) | Delta |
| :--- | :---: | :---: | :---: |
| **S-S1 Highway Sub-Dataset (N=25)** | | | |
| Drift % — Mean | 102.96% | 277.27% | +174.31% |
| **Drift % — Median** | **58.02%** | **95.16%** | **+37.14%** |
| Drift % — 90th Percentile | 166.55% | 177.47% | +10.92% |
| Drift % — Worst Case | 806.03% | 4671.89% | +3865.86% |
| Position Error m — Mean | 223.58 m | 262.82 m | +39.24 m |
| **Position Error m — Median** | **188.55 m** | **218.41 m** | **+29.86 m** |
| Position Error m — 90th Percentile | 456.66 m | 511.93 m | +55.27 m |
| Position Error m — Worst Case | 530.88 m | 657.28 m | +126.40 m |
| | | | |
| **S-S2 Urban Sub-Dataset (N=25)** | | | |
| Drift % — Mean | 105.86% | 114.35% | +8.49% |
| **Drift % — Median** | **108.91%** | **103.35%** | **-5.56%** |
| Drift % — 90th Percentile | 208.38% | 130.53% | **-77.85%** |
| Drift % — Worst Case | 256.91% | 436.91% | +180.00% |
| Position Error m — Mean | 371.74 m | 453.67 m | +81.93 m |
| **Position Error m — Median** | **269.31 m** | **299.35 m** | **+30.04 m** |
| Position Error m — 90th Percentile | 863.49 m | 973.92 m | +110.43 m |
| Position Error m — Worst Case | 1047.83 m | 1746.30 m | +698.47 m |
| | | | |
| **COMBINED ALL TRIPS (N=50)** | | | |
| Drift % — Mean | 104.41% | 195.81% | +91.40% |
| **Drift % — Median** | **73.54%** | **101.05%** | **+27.51%** |
| Drift % — 90th Percentile | 183.90% | 162.12% | **-21.78%** |

---

## 4. Turn-Severity Correlation Analysis

The scatter plot comparing total accumulated turn angle $\int |\omega_z| dt$ vs. final position error before and after the fix is shown below.

![Turn Severity Scatter Plot](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/fixed_heading_error_vs_turn_severity.png)

- **Before Fix (Baseline)**: $r = +0.5732, \quad p < 0.0001 \quad (\text{slope } = 1.73\text{ m/deg})$
- **After Fix (Closed-Loop EKF)**: $r = +0.6063, \quad p < 0.0001 \quad (\text{slope } = 2.14\text{ m/deg})$

### Root-Cause Diagnosis of Correlation Persistence
1. **GNSS Course-Over-Ground (COG) Velocity Lag**: GNSS velocity estimates derived from Doppler/carrier phase have an intrinsic $\sim 0.5\text{s} - 1.0\text{s}$ latency during dynamic vehicle turns.
2. **False Bias Contamination**: When GNSS COG is fed into the Kalman filter right before or during a curve, the filter interprets the GNSS velocity vector lag as physical gyro bias $b_{g,z}$ (e.g. driving $b_{g,z}$ up to $+0.2452^\circ/\text{s}$).
3. **Blackout Integration**: Once blackout commences, this artificially inflated $b_{g,z}$ is integrated open-loop, accumulating $\sim 7.35^\circ$ of heading error over 30s, which amplifies cross-track position drift during turns.

---

## Key Files & Artifacts
- **Report Document**: [`EKF_CLOSED_LOOP_SWEEP_REPORT.md`](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/EKF_CLOSED_LOOP_SWEEP_REPORT.md)
- **Sweep Results CSV**: [`fixed_sweep_results.csv`](file:///c:/Users/carpe/SIH/artifacts/fixed_sweep_results.csv)
- **Production ES-EKF Code**: [`es_ekf.py`](file:///c:/Users/carpe/SIH/sih/fusion/es_ekf.py)
