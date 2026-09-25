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
