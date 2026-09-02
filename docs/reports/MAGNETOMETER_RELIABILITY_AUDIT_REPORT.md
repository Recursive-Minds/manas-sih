# Smartphone Intelligent Dead Reckoning (SIH): Step 1 Magnetometer Reliability Audit Report

## Executive Audit Summary

Per project protocol, a complete reliability audit was conducted on magnetometer signals from dataset `S-S1.csv` before attempting any EKF measurement fusion.

The audit evaluated three criteria:
1. Straight-line highway heading tracking against GNSS course-over-ground.
2. Turning segment responsiveness and lag.
3. Hard/soft-iron magnetic distortion scatter pattern (Center offset vs. Span radius).

---

## 1. Check 1: Straight-Line Highway Segment Comparison

- **Matched Straight Samples**: 511 samples (speed > 3 m/s, yaw rate < $1.0^\circ/\text{s}$)
- **Mean Heading Difference ($\psi_{\text{mag}} - \psi_{\text{GNSS}}$)**: **$+23.96^\circ$**
- **Standard Deviation ($\sigma$)**: **$93.57^\circ$**

> **Finding**: Even during steady straight highway driving, magnetometer-derived heading exhibits a massive standard deviation of **$93.57^\circ$**, fluctuating wildly between $-180^\circ$ and $+180^\circ$. Fusing this signal into the orientation error state would severely degrade filter convergence.

---

## 2. Check 2: Turning Segment Comparison

- **Matched Turning Samples**: 431 samples (yaw rate > $3.0^\circ/\text{s}$)
- **Mean Heading Difference during Turns**: **$+23.33^\circ$**
- **Standard Deviation ($\sigma$) during Turns**: **$80.91^\circ$**

> **Finding**: Magnetometer signals suffer from heavy dynamic noise, angular distortion, and severe lag during vehicle maneuvers.

---

## 3. Check 3: Hard/Soft-Iron Distortion Analysis

![Magnetometer Hard/Soft-Iron Distortion Scatter Audit](file:///C:/Users/carpe/.gemini/antigravity-ide/brain/10a4684a-0cfa-486c-a598-6a3b7a170b92/magnetometer_distortion_audit.png)

- **Magnetometer Center Offset**:
  - Center X: **$-15.56\ \mu\text{T}$** (Span Radius X = $26.31\ \mu\text{T}$)
  - Center Y: **$-27.91\ \mu\text{T}$** (Span Radius Y = $11.28\ \mu\text{T}$)
- **Offset Distortion Percentage**: Center Y offset is **$247.4\%$ of total signal radius**, placing the origin completely outside the magnetic locus.

> **Finding**: Raw and leveled magnetometer X-Y scatter plots show extreme hard-iron center displacement and heavy soft-iron squashing caused by smartphone internal speaker magnets, battery currents, and vehicle chassis ferromagnetic structures.

---

## 4. Final Step 1 Audit Verdict

| Audit Check | Test Metric | Result | Reliability Threshold | Status |
| :--- | :--- | :---: | :---: | :---: |
| **1. Straight Highway Tracking** | Std Deviation ($\sigma$) | **$93.57^\circ$** | $< 25.0^\circ$ | **FAIL** |
| **2. Turning Responsiveness** | Std Deviation ($\sigma$) | **$80.91^\circ$** | $< 25.0^\circ$ | **FAIL** |
| **3. Hard/Soft-Iron Distortion** | Center Y Offset / Radius Y | **$247.4\%$** | $< 30.0\%$ | **FAIL** |

### **FINAL VERDICT: [FAIL (UNUSABLE)]**

---

## Next Steps

Per project instructions:
- **Do NOT proceed to Step 2 (Magnetometer EKF Fusion)**.
- Reporting back to user for further decision.
