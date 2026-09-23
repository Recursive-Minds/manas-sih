# Round 1 Task Log

This log records every command and process executed during Round 1 tuning and evaluation, per Hard Rule 5.

---

### Step 0: Reference Baseline Pytest Run
- **Start Time**: 2026-09-23 19:40:51 +05:30
- **End Time**: 2026-09-23 19:47:07 +05:30
- **Command**: `python -m pytest tests -q --continue-on-collection-errors 2>&1 | Tee-Object -FilePath "logs\round1\step0_pytest_baseline.log"`
- **Purpose**: Record pass/fail test counts before applying Round 1 pack.
- **Exit Code**: 1
- **Log Path**: `logs/round1/step0_pytest_baseline.log`
- **Key Numbers**: 102 passed, 1 failed (`test_benchmark_script_has_no_stale_literals` on `/ 35` in docstring), 1 skipped. Total 104 tests in 372.16s.
- **Commit Hash**: `fc2854d64c61b2cbfcdd5bf81cf0c95abe7e4c9d`

---

### Step 1: Unzip Inactive Round 1 Pack
- **Start Time**: 2026-09-23 19:47:48 +05:30
- **End Time**: 2026-09-23 19:47:49 +05:30
- **Command**: `python -c "import os, zipfile; z = zipfile.ZipFile('sih_round1_pack.zip'); ... z.extract(info, '.')"`
- **Purpose**: Extract Round 1 pack files into repository root with non-overwriting semantics (`unzip -n`).
- **Exit Code**: 0
- **Log Path**: `logs/round1/step1_unzip.log`
- **Key Numbers**: 32 files extracted, 0 skipped (all 32 files were new).
- **Commit Hash**: `10d5341`

---

### Step 2: Reference Baseline Run (pre_patch)
- **Start Time**: 2026-09-23 19:48:33 +05:30
- **End Time**: 2026-09-23 19:52:17 +05:30
- **Command**: `python scripts/round1_eval.py --tag pre_patch --configs config/round1/baseline_off.json 2>&1 | Tee-Object -FilePath "logs\round1\step2_pre_patch.log"`
- **Purpose**: Run reference baseline before any engine edits across 6 canonical seeds (236 scenarios).
- **Exit Code**: 0
- **Log Path**: `logs/round1/step2_pre_patch.log`
- **Key Numbers**:
  - Median of seed medians: 12.968%
  - Mean +- std of seed medians: 13.194 +- 0.879%
  - P90 drift: 40.823%
  - T1 (<10%) share: 41.53%
  - Beats pure share: 81.78%
  - Worst drift: 172.789%
  - Median |AT|: 33.27 m, Median |CT|: 5.89 m
  - Per-seed medians: 12345: 13.25%, 45736: 12.29%, 75496: 12.24%, 314159: 12.69%, 541098: 14.32%, 987654: 14.38%
  - By domain: Arterial: 14.53%, Highway: 8.88%, Mixed: 10.83%, Urban: 15.26%
- **Commit Hash**: `1e18708`

---

### Step 3: Apply Marked Edits and Verify Parity
- **Start Time**: 2026-09-23 19:53:11 +05:30
- **End Time**: 2026-09-23 20:02:51 +05:30
- **Commands**:
  1. `python scripts/round1_apply_edits.py --check 2>&1 | Tee-Object -FilePath "logs\round1\step3_apply_edits_check.log"`
     - Purpose: Dry run anchor verification for engine and Kotlin edits.
     - Exit Code: 0
     - Key Numbers: 11 anchors verified (8 engine, 3 Kotlin), all valid.
  2. `python scripts/round1_apply_edits.py 2>&1 | Tee-Object -FilePath "logs\round1\step3_apply_edits.log"`
     - Purpose: Apply marked Round 1 edits.
     - Exit Code: 0
     - Key Numbers: 8 edits written to `sih/engine/dead_reckoning_engine.py`, 3 edits to `MainActivity.kt`.
  3. `python -m pytest tests -q --continue-on-collection-errors 2>&1 | Tee-Object -FilePath "logs\round1\step3_pytest_post_patch.log"`
     - Purpose: Regression testing post-patch.
     - Exit Code: 1
     - Key Numbers: 115 passed (102 baseline + 13 new passes), 1 failed (baseline stale literal check), 1 skipped. No new failures.
  4. `python scripts/round1_eval.py --tag parity --configs config/round1/baseline_off.json config/round1/diagnostics.json --assert-parity results/round1/pre_patch/baseline_off_scenarios.csv 2>&1 | Tee-Object -FilePath "logs\round1\step3_parity.log"`
     - Purpose: Verify bit-exact parity against pre-patch baseline.
     - Exit Code: 0
     - Key Numbers: `PARITY PASS` (max_abs_diff_map_err_m = 5.68e-14, exactly 0 within float64 precision).
- **Log Paths**:
  - `logs/round1/step3_apply_edits_check.log`
  - `logs/round1/step3_apply_edits.log`
  - `logs/round1/step3_pytest_post_patch.log`
  - `logs/round1/step3_parity.log`
- **Commit Hash**: `82332d7`

---

### Step 4: Worst-Scenario Autopsy (T2)
- **Start Time**: 2026-09-23 20:03:31 +05:30
- **End Time**: 2026-09-23 20:05:05 +05:30
- **Command**: `python scripts/round1_autopsy.py --seed 541098 --ids 9 40 35 --worst 5 2>&1 | Tee-Object -FilePath "logs\round1\step4_autopsy.log"`
- **Purpose**: Decompose error sources (speed scale, stop creep, heading cross-track) and inspect T9 double scale factor on Seed 541098.
- **Exit Code**: 0
- **Log Path**: `logs/round1/step4_autopsy.log`
- **Key Numbers**:
  - Seed median drift: 14.32% over 40 scenarios.
  - Worst 5 scenarios: #40 (101.0%), #9 (91.0%), #35 (47.6%), #38 (34.6%), #37 (32.7%).
  - Main causes: All top 5 driven by SPEED/SCALE error.
  - Double speed-scale check (T9): Median engine scale 1.201, median effective scale 1.201 (0.0% difference, does not differ by >2%).
- **Commit Hash**: Pending commit (`round1: autopsy results`).



