# Phone Task Log

This log records every command and process executed during the On-Device Phone Phase, per Ground Rule 1.

---

### Step 0: Baseline and Runtime Audit
- **Timestamp**: 2026-09-26 01:08:00 +05:30
- **Base Tag**: `docs-verified`
- **Branch**: `phone/s0-baseline` (branched from `docs-verified` at commit `1076f8e`)
- **Commands Executed**:
  1. `git tag -l "docs-verified"`: Verified tag exists.
  2. `python scripts/check_number_registry.py`: PASS (all numbers registered and consistent).
  3. `python scripts/check_doc_numbers.py`: PASS (all document numbers match truth).
  4. `git checkout -b phone/s0-baseline docs-verified`: Created branch.
  5. `python -m pytest tests -q 2>&1 | Tee-Object -FilePath "logs\phone\step0_pytest_baseline.log"`:
     - Exit Code: 0
     - Log Path: `logs/phone/step0_pytest_baseline.log`
     - Result: 127 passed, 1 skipped in 466.59s.
  6. `python scripts/quick_parity.py 2>&1 | Tee-Object -FilePath "logs\phone\step0_quick_parity_normal.log"`:
     - Exit Code: 0
     - Log Path: `logs/phone/step0_quick_parity_normal.log`
     - Result: 5/5 passed (<0.01m), exact 0.0000 m endpoint and trajectory diff.
  7. `python scripts/quick_parity.py --raw 2>&1 | Tee-Object -FilePath "logs\phone\step0_quick_parity_raw.log"`:
     - Exit Code: 1 (endpoint diffs vs batch > 5m on scenarios #23, #25, #26, #30 due to live CausalSpeedSmoother)
     - Log Path: `logs/phone/step0_quick_parity_raw.log`
     - Result: #22: 4.4905m (PASS), #23: 69.2377m (diff), #25: 33.6536m (diff), #26: 7.7096m (diff), #30: 11.3600m (diff).
  8. `python scratch/measure_stage_b_latency.py 2>&1 | Tee-Object -FilePath "logs\phone\step0_stage_b_latency.log"`:
     - Exit Code: 0
     - Log Path: `logs/phone/step0_stage_b_latency.log`
     - Result:
       - CPU: Mean 5.059 ms (P50: 4.922 ms, P90: 6.302 ms, P99: 8.060 ms, Max: 8.940 ms)
       - CUDA: Mean 9.773 ms (P50: 9.233 ms, P90: 13.719 ms, P99: 17.918 ms, Max: 23.377 ms)
  9. Import Audit (`scratch/audit_engine_only.py`, `scratch/trace_pandas.py`):
     - Identified laptop-only server layer (`aiohttp`, `web`, tile proxy, `LiveEvaluator`).
     - Identified engine runtime dependencies (`numpy`, `scipy.signal`, `torch`, `sih.*`).
     - Identified transitive bloat (`pandas`, `pyarrow`, `tqdm` via `sih/data/__init__.py` and `sih/__init__.py`).
- **Artifacts Saved**:
  - `BASELINE_ONDEVICE.json`: Complete record of baseline metrics.
  - `logs/phone/step0_pytest_baseline.log`
  - `logs/phone/step0_quick_parity_normal.log`
  - `logs/phone/step0_quick_parity_raw.log`
  - `logs/phone/step0_stage_b_latency.log`

---

### Step 0b: Raw-Mode Gap Isolation and Causality Audit
- **Timestamp**: 2026-09-26 01:25:00 +05:30
- **Branch**: `phone/s0-baseline`
- **Commands Executed**:
  1. Added `use_speed_smoother: bool = True` constructor flag to `EngineAdapterStageB` in `server/engine_adapter.py`.
  2. Added `--no-smoother` flag and speed diff reporting to `scripts/quick_parity.py`.
  3. `python scripts/quick_parity.py --raw --no-smoother 2>&1 | Tee-Object -FilePath "logs\phone\step0b_raw_no_smoother.log"`:
     - Exit Code: 1
     - Log Path: `logs/phone/step0b_raw_no_smoother.log`
     - Result: Speed diffs during blackout are 0.012 to 0.016 m/s (> 1e-4); endpoint diffs are 0.69m to 29.12m (> 0.01m).
  4. `python scripts/quick_parity.py`:
     - Exit Code: 0
     - Result: 5/5 scenarios exact 0.0000 m parity (<0.01m).
  5. `python -m pytest tests/test_no_future_leak.py tests/test_causal_streaming.py tests/test_app_no_leak.py tests/test_round1.py -q`:
     - Exit Code: 0
     - Result: 30 passed in 350.49s.
  6. Causality audit: Verified batch `predict_velocities` causality (2nd-order Butterworth via `sosfilt` with carried state in `sih/features/streaming.py:65, 103-104`, trailing causal windows in `sih/models/inference.py:151-158`, NO `filtfilt`, NO centred windows).
  7. Root cause of raw divergence isolated: Identified feature buffer empty state and spectral window warm-up transient at `sih/features/streaming.py:124-126`, leading to speed distortion at `warmup_start_ns` (5.91 m/s diff on step 0) which biases the learned pre-blackout scale factor in `dead_reckoning_engine.py:228`.
- **Decision**: Option 3.b.

---

### Step 0c: Pre-Roll Feature Priming and Raw Mode Parity
- **Timestamp**: 2026-09-26 02:45:00 +05:30
- **Branch**: `phone/s0-baseline`
- **Commands Executed**:
  1. Added `prime_features(imu_samples, calib_samples=None)` to `server/engine_adapter.py`.
     - Runs feature extraction + model (+ smoother if enabled) causally from trip start to warmup start.
     - Appends to `recent_ai_speeds` and `recent_imu_calib` without touching EKF session, GNSS buffers, or state machines.
  2. Updated `scripts/quick_parity.py`:
     - In `--raw` mode, invokes `adapter.prime_features(trip.imu_samples[:j_warm], calib_samples=calibs[:j_warm])` from trip start.
     - Removed artificial batch `v_preds` pre-fill in raw mode (live model produces its own speeds).
     - Added `--cpu` flag for exact CPU bit-parity.
  3. `python scripts/quick_parity.py --raw --no-smoother --cpu`:
     - Exit Code: 0
     - Result: 5/5 scenarios pass with exact 0.0000 m endpoint diff (< 0.05m) and max speed diff <= 1.0e-5 m/s (< 1e-4 m/s).
  4. Updated `EngineAdapterStageB` constructor default to `use_speed_smoother: bool = False`.
  5. Added unit test `tests/test_app_raw_parity.py`:
     - Verifies raw mode pre-roll on all 5 canonical S-S3a scenarios.
     - Enforces max endpoint diff < 0.05 m and max speed diff < 1e-4 m/s.
     - Result: 5 passed in 47.25s.
  6. Verification:
     - `python scripts/round1_eval.py --tag parity_check --configs config/round1/baseline_off.json --assert-parity results/round1/pre_patch/baseline_off_scenarios.csv`: PASS (max diff 5.68e-14 m).
     - `python scripts/quick_parity.py`: PASS (5/5 scenarios 0.0000 m endpoint diff).
     - Repo-wide `pytest`: 132 passed, 1 skipped in 490.74s (0:08:10).

---

### Step 0d: Production Causal Smoother Audit and Parity Confirmation
- **Timestamp**: 2026-09-26 11:05:00 +05:30
- **Branch**: `phone/s0-baseline`
- **Causality & Code Audit**:
  - `sih/models/inference.py:101`: `predict_velocities` has parameter `apply_smoothing: bool = True`.
  - `sih/models/inference.py:195-197`: calls `CausalSpeedSmoother(a_max_mps2=3.5, a_min_mps2=-5.0, tau_s=0.25).filter_sequence(raw_preds, dt_s=0.1)`.
  - `sih/fusion/speed_smoother.py:39-81`: `CausalSpeedSmoother` is strictly causal; uses 1st-order forward recursion `v_smooth = (1 - alpha) * v_prev + alpha * v_clamped` on samples `<= t` with acceleration slew rate limiting `[-5.0, +3.5] m/s^2`. No `filtfilt`, no centered windows, no backward pass.
  - Reverted decision: The benchmarked system operates with `apply_smoothing=True`. The Step 0c smoother comparison compared against an un-benchmarked no-smoother baseline; the batch numbers (16.25m, 67.77m, 77.30m, 122.77m, 7.04m) match smoother-ON.
- **Commands Executed**:
  1. Reverted `EngineAdapterStageB` constructor default to `use_speed_smoother: bool = True` in `server/engine_adapter.py`.
  2. `python scripts/quick_parity.py --raw --cpu`:
     - Exit Code: 0
     - Result: 5/5 scenarios pass with exact 0.0000 m endpoint diff (< 0.05m), 0.0000 m max trajectory diff, and max speed diff <= 0.000005 m/s (5.0e-6 m/s, target < 1e-4 m/s).
  3. Updated `tests/test_app_raw_parity.py`:
     - Tests production configuration (`use_speed_smoother=True`, `apply_smoothing=True`) across all 5 canonical scenarios.
     - Retains `test_raw_pre_rolled_parity_no_smoother_extra` as an extra test.
     - Result: 6 passed in 59.26s.
  4. `python scripts/round1_eval.py --tag parity_check --configs config/round1/baseline_off.json --assert-parity results/round1/pre_patch/baseline_off_scenarios.csv`:
     - Exit Code: 0
     - Result: PARITY PASS (max diff 5.68e-14 m <= 1e-9 m).
  5. `python scripts/quick_parity.py`:
     - Exit Code: 0
     - Result: 5/5 scenarios pass with exact 0.0000 m endpoint diff.
  6. Repo-wide `pytest`:
     - Result: 133 passed, 1 skipped.


