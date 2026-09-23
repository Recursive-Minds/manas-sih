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
- **Commit Hash**: Pending commit (`round1: pre_patch baseline results`).

