# Project Memory & Architecture Guide: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion (SIH)

Refer to [CLAUDE.md](file:///c:/Users/carpe/SIH/CLAUDE.md) for full context, rules, workflow principles, and phase roadmaps.

## Key Rules for this Workspace
1. **One narrow task at a time**: Never expand scope silently.
2. **Real Data Verification**: Never report a step complete without concrete, inspectable output (metrics, plots, error rates) on real data.
3. **Data Splits**: ALWAYS split datasets strictly by trip/sequence, NEVER by row.
4. **Modularity**: All stages (`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`) must communicate strictly through fixed contracts behind abstract interfaces, configured at a single assembly point.
5. **Sensor-Agnostic Core**: Core engine is decoupled from Android-specific code and must support phone IMUs (10Hz) up to high-rate external IMUs (~200Hz).
6. **Benchmark Target**: Drift < 10% of total distance travelled during GNSS blackout (< 5m over 50m, or < 100m over 1km).
7. **No Assumptions on Datasets**: Never assume dataset availability, file schemas, or remote storage structure — always search, probe, and verify real repository files/URLs before hardcoding.
8. **High-Speed Velocity Scaling**: When training neural velocity estimators from IMU, avoid loss functions that compress gradients on high speeds (e.g. tight Huber thresholds). Verify that the predicted speed scale ratio sum(v_hat) / sum(v_GT) is approximately 1.00.
9. **3D Mount Invariance**: Always apply 3D SO(3) rotational data augmentation during IMU model training to prevent memorization of static cradle gravity vectors.
10. **Trajectory Error Decomposition**: When evaluating dead-reckoning performance on real data, decompose position errors into along-track (speed scale) and cross-track (turn rate / heading) components to diagnose drift root causes.
11. **End-to-End Benchmark & Report Synchronization Pipeline**:
Whenever any algorithmic change, model retraining, or parameter update is made, follow this mandatory 4-step execution chain:
- **Step 1**: Implement the changes directly in production code (`benchmarks/run_final_benchmark.py`, `sih/`).
- **Step 2**: Re-run the full 40-scenario benchmark on real data to produce new coordinates and error metrics.
- **Step 3**: Re-render all trajectory maps, gallery plots, and distribution charts from the newly generated benchmark run.
- **Step 4**: Re-generate BOTH `FINAL_JUDGE_EVALUATION_REPORT.md` AND `FINAL_JUDGE_EVALUATION_REPORT.html` with updated tables, scorecards, and newly rendered base64 images embedded directly. Never report an update complete without running this entire chain end-to-end.
12. **Clean Plain-Text Math Formatting**: NEVER use raw LaTeX syntax (e.g. `$`, `$$`, `\approx`, `\frac`, `\text{}`, `\omega`, `\Delta`, `\sigma`) in user responses, implementation plans, or generated markdown files. Always format mathematical expressions and physical variables in readable plain text or standard Unicode symbols (e.g. `p_head = exp(-0.5 * (diff_h / sigma)^2)`, `omega_z >= 2.5 deg/s`, `90 degrees`, `kappa = 1/R`, `a_lat <= 1.2 m/s^2`).

