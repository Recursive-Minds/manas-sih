# MASTER PROMPT (UPDATED) — Intelligent Dead Reckoning (IDR) with GNSS Fusion

Paste this as your next message to Claude Code / Antigravity in the existing project (`c:/Users/carpe/SIH/`). It supersedes the earlier master prompt's "where we are" section — the phase and file structure below are what actually exists now, not a plan.

---

## Instruction to the agent — read this first

Update the persistent project memory file (`CLAUDE.md` / equivalent) to replace its "current state" section with everything below. Do not start any new implementation yet. First, summarize back to me in your own words: (1) what's actually built and passing right now, (2) the specific anomaly and gaps listed in Section 3 below, (3) confirm you understand we are NOT starting Phase 4 (map matching) yet. Then stop and wait for me to assign the first specific investigation task.

---

## 1. Actual current state (verified, not aspirational)

The pipeline is built through Phase 3 and follows the modular contract strictly:
`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`

Implemented and working:
- **Calibration** (`sih/calibration/mount.py`): 3D gravity leveling + centripetal-correlation yaw-axis lock with polarity detection.
- **AI velocity estimator** (`sih/models/tcn_attention.py`, `sih/velocity/ai_estimator.py`): multi-scale dilated TCN (dilations 1/2/4/8/16) + 4-head self-attention over a 100-step (10s) rolling window of 8 input channels, with a dual head predicting forward speed and log-variance, trained with speed-stratified importance weighting.
- **Fusion** (`sih/fusion/es_ekf.py`): 15-state error-state EKF (position, velocity, attitude, accel bias, gyro bias) with non-holonomic constraints, ZUPT/ZARU, and an online Doppler-vs-AI speed scale factor (`s_v`) that adapts during GNSS availability and freezes during blackout.
- **Eval harness** (`sih/eval/metrics.py`, `sih/eval/benchmark.py`): position RMSE, max error, drift %, along/cross-track decomposition, simulated blackout injection.

## 2. Current benchmark results (real, measured — treat as ground truth for now)

| Trip | Scenario | Pipeline | Drift % | Result |
|---|---|---|---|---|
| S-S1 | 60s blackout @ 300s | Phase 3 (ES-EKF + AI) | 8.69% | PASSED |
| S-S1 | 30s blackout @ 120s | Phase 3 (ES-EKF + AI) | 13.72% | FAILED |
| S-S2 (unseen) | 30s blackout @ 120s | Phase 3 (ES-EKF + AI) | 25.59% | FAILED |

Phase 2 (EKF+NHC, no AI) and naive-baseline numbers are also recorded in the full report and confirm the AI velocity layer is doing real, large work (e.g. 98.87% -> 8.69% on the first scenario). This is genuine, verified progress — the issue is consistency and generalization, not whether the core idea works.

## 3. Specific issues to resolve before touching Phase 4 — do these first, in this order

These are the actual next tasks. Do not reorder into map-matching work until these are closed out or deliberately deferred with my sign-off.

1. **Diagnose the S-S1 anomaly**: the 30s@120s blackout (13.72% drift) performed worse than the 60s@300s blackout (8.69%) despite being shorter. Plot (a) the calibration's estimated leveling/yaw-lock convergence over the first 120s, and (b) the value of the online scale factor `s_v` over the same window, for both scenarios side by side. Hypothesis to test first: insufficient convergence time before the earlier blackout. Do not move to a fix until this specific hypothesis is confirmed or ruled out with a plot.
2. **Run a randomized blackout sweep**: instead of the fixed 2-3 injection points, run many blackout injections (varied start time and duration, e.g. 20-30 per trip) across both S-S1 and S-S2, and report the drift % distribution (mean, median, 90th percentile, worst-case) rather than single numbers. Treat the existing 3 numbers as anecdotes until this exists.
3. **Wire the predicted uncertainty into the fusion filter**: currently `log(sigma^2)` is output by the velocity model but, as far as this document shows, not consumed downstream. Use it to scale the EKF's measurement noise covariance for the velocity update, so the filter trusts the AI estimate less exactly when the model itself is uncertain. Test before/after on the same blackout sweep from step 2.
4. **Break down the S-S2 generalization gap by regime**: segment the unseen trip's error by speed band and by turn density (or however motion_state already classifies segments) to identify whether the 25.59% is concentrated in specific conditions (e.g. stop-and-go, sharp turns) or roughly uniform. This determines whether the fix is "collect more diverse training data" or "the filter/model has a structural limitation."
5. **Sanity-check the S-S2 baseline number**: 2145% naive-baseline drift is far larger than S-S1's 229%/316% baselines. Confirm this isn't a coordinate-frame, unit, or file-parsing discrepancy specific to S-S2 before treating it as a real reflection of urban driving difficulty.
6. **Re-verify the trip-level train/validation split**: explicitly show, don't just assert, that no row or window from S-S2 appears in training data for the velocity model, and vice versa.
7. **Log scale-factor (`s_v`) clipping frequency**: report how often `s_v` hits its 0.7/1.6 clip bounds across both trips. Frequent clipping indicates the AI velocity model's raw output is systematically miscalibrated, which the scale factor is currently masking rather than fixing — if this is common, the fix belongs in the velocity model, not the scale factor.

## 4. Working rules — unchanged from before, restate for this session

1. One narrow task at a time — work through Section 3's items one at a time, in order, not in parallel.
2. Always show a real plot or real number from real data before calling a step done.
3. Proactively flag any suspiciously good or suspiciously bad result before I have to ask — including if a "fix" makes a number look implausibly perfect.
4. Do not fabricate or estimate benchmark numbers — only report what's actually computed from the data.
5. Train/validation splits remain strictly by trip, never by row, for any retraining.
6. Present options with trade-offs for any non-trivial design decision (e.g. how to weight the uncertainty in the EKF's R matrix) — don't decide unilaterally.
7. Do not begin Phase 4 (map matching) or any new architecture/model work until Section 3 is resolved or I explicitly say to defer a specific item.

---

**End of updated context. Confirm you've saved this, summarize the anomaly and the four other issues back to me in your own words, then wait for me to assign the first task from Section 3.**
