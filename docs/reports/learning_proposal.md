# Learning Proposal: Kinematic Calibration & AI Velocity Scaling in Smartphone Dead Reckoning

## 1. Context & Motivation
During real-trip benchmarking of Smartphone Intelligent Dead Reckoning (IDR) without map-matching:
1. **Regression Shrinkage on High Speeds**: Using bounded Huber regularization on Gaussian Negative Log-Likelihood (NLL) caused the neural network to under-predict high vehicle speeds ($20-30\text{ m/s}$) by $\approx 20\%$, producing an along-track distance deficit.
2. **Mount Tilt Cosine Attenuation**: Single-axis gyro integration ($Y$-axis pitch) in an upright cradle tilted at angle $\alpha$ measures $\omega \cos(\alpha)$, which underestimates vehicular turns by the projection angle unless full 3D leveled coordinates are reconstructed.
3. **Visual vs Numerical Alignment**: True physical dead reckoning requires simultaneously balancing along-track distance scaling ($s(t) = \int v dt$) and cross-track heading turn geometry ($\theta(t) = \theta_0 - \int \omega_z dt$).

---

## 2. Proposed Rule Update: [GEMINI.md](file:///c:/Users/carpe/SIH/GEMINI.md)

### Proposed Addition to `GEMINI.md`
```markdown
## Deep Learning & Kinematics Principles for Smartphone IDR
8. **High-Speed Velocity Scaling**: When training neural velocity estimators from IMU, avoid loss functions that heavily compress gradients on high speeds (e.g. tight Huber thresholds). Verify that the predicted speed scale ratio $\frac{\sum \hat{v}}{\sum v_{\text{GT}}} \approx 1.00$.
9. **3D Mount Invariance**: Always apply 3D SO(3) rotational data augmentation during IMU model training to prevent memorization of static cradle gravity vectors.
10. **Trajectory Error Decomposition**: When evaluating dead-reckoning performance on real data, decompose position errors into along-track (speed scale) and cross-track (turn rate / heading) components to diagnose drift root causes.
```

---

## 3. User Action
Please review and confirm if you would like me to persist these principles to [GEMINI.md](file:///c:/Users/carpe/SIH/GEMINI.md).
