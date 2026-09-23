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
- **Commit Hash**: `ec8b740`

---

### Step 5: Ablation (13 configs x 6 canonical seeds)
- **Start Time**: 2026-09-23 20:05:44 +05:30
- **End Time**: 2026-09-23 20:32:03 +05:30
- **Command**: `python scripts/round1_eval.py --tag ablation --configs config/round1/baseline_off.json config/round1/t9_engine.json config/round1/t9_ekf.json config/round1/t3_stop.json config/round1/t4_gyro.json config/round1/t5_hold.json config/round1/t5_decay.json config/round1/t7_band.json config/round1/t8_junction.json config/round1/t9e_t3.json config/round1/t9e_t8.json config/round1/t9e_t3_t8.json config/round1/all_on.json 2>&1 | Tee-Object -FilePath "logs\round1\step5_ablation.log"`
- **Purpose**: Paired ablation of each Round 1 flag (and key combos) vs baseline, all 6 canonical seeds.
- **Exit Code**: 0
- **Log Path**: `logs/round1/step5_ablation.log`
- **Key Numbers (ablation table)**:

| Config | Median | Mean +- Std | P90 | T1 | Worst | vs_first |
|---|---|---|---|---|---|---|
| baseline_off | 12.97% | 13.19 +- 0.88 | 40.8% | 0.42 | 172.8% | (baseline) |
| t9_engine | 12.77% | 13.17 +- 1.05 | 40.8% | 0.42 | 172.8% | B7/W8 delta=+0.00pp |
| t9_ekf | 13.63% | 13.76 +- 1.14 | 45.3% | 0.37 | 153.3% | B107/W113 delta=+0.01pp |
| t3_stop | 12.97% | 13.19 +- 0.88 | 40.8% | 0.42 | 172.8% | B5/W5 delta=+0.00pp |
| t4_gyro | 13.81% | 13.65 +- 0.76 | 40.8% | 0.40 | 172.8% | B13/W29 delta=+0.00pp |
| t5_hold | 19.19% | 19.11 +- 4.99 | 57.9% | 0.34 | 127.2% | B91/W131 delta=+1.82pp |
| t5_decay | 13.17% | 13.20 +- 3.17 | 44.4% | 0.42 | 186.6% | B103/W85 delta=-0.09pp |
| t7_band | 12.80% | 12.62 +- 0.59 | 40.3% | 0.41 | 172.6% | B76/W57 delta=-0.03pp |
| t8_junction | 12.78% | 12.82 +- 0.95 | 40.8% | 0.43 | 172.8% | B19/W8 delta=+0.00pp |
| t9e_t3 | 12.77% | 13.17 +- 1.05 | 40.8% | 0.42 | 172.8% | B12/W13 delta=+0.00pp |
| t9e_t8 | 12.77% | 12.80 +- 1.10 | 40.8% | 0.43 | 172.8% | B26/W15 delta=+0.00pp |
| t9e_t3_t8 | 12.77% | 12.80 +- 1.10 | 40.8% | 0.43 | 172.8% | B31/W20 delta=+0.00pp |
| all_on | 12.84% | 12.68 +- 0.74 | 40.3% | 0.41 | 172.6% | B76/W80 delta=+0.00pp |

- **Commit Hash**: `f0a8a25`

---

### Step 6: T6 Interval-Loss Velocity Fine-Tuning & Evaluation
- **Start Time**: 2026-09-23 20:33:00 +05:30
- **End Time**: 2026-09-23 20:55:00 +05:30
- **Commands**:
  1. `python sih/models/train_interval.py --smoke 2>&1 | Tee-Object -FilePath "logs\round1\step6_smoke.log"`
     - Purpose: 2-epoch smoke test on interval-loss fine-tuning.
     - Exit Code: 0
     - Key Numbers: Best epoch 2, validation int_err_median 0.1022.
  2. `python sih/models/train_interval.py --lam 0.5 2>&1 | Tee-Object -FilePath "logs\round1\step6_lam0.5.log"`
     - Purpose: Train candidate lambda = 0.5.
     - Exit Code: 0
     - Key Numbers: Validation int_err_median: 0.1305 (base) -> 0.1214 (best epoch 4, RMSE 2.98, scale 1.098).
  3. `python sih/models/train_interval.py --lam 1.0 2>&1 | Tee-Object -FilePath "logs\round1\step6_lam1.0.log"`
     - Purpose: Train candidate lambda = 1.0.
     - Exit Code: 0
     - Key Numbers: Validation int_err_median: 0.1305 (base) -> 0.1218 (best epoch 5, RMSE 2.98, scale 1.084).
  4. `python sih/models/train_interval.py --lam 2.0 2>&1 | Tee-Object -FilePath "logs\round1\step6_lam2.0.log"`
     - Purpose: Train candidate lambda = 2.0.
     - Exit Code: 0
     - Key Numbers: Validation int_err_median: 0.1305 (base) -> 0.1192 (best epoch 5, RMSE 3.01, scale 1.087).
  5. `python scripts/round1_eval.py --tag t6_lam0.5 --model-path models/checkpoints/round1_interval_lam0.5_s42.pt --configs config/round1/baseline_off.json config/round1/t9_engine.json 2>&1 | Tee-Object -FilePath "logs\round1\step6_eval_lam0.5.log"`
     - Purpose: Benchmark evaluation for lambda = 0.5 across 6 canonical seeds.
     - Exit Code: 0
     - Key Numbers:
       - `baseline_off`: median 11.05%, mean 11.31 +- 2.11%, p90 43.8%, T1 0.45, worst 177.7%
       - `t9_engine`: median 11.60%, mean 11.52 +- 2.22%, p90 43.8%, T1 0.45, worst 177.7%
  6. `python scripts/round1_eval.py --tag t6_lam1.0 --model-path models/checkpoints/round1_interval_lam1.0_s42.pt --configs config/round1/baseline_off.json config/round1/t9_engine.json 2>&1 | Tee-Object -FilePath "logs\round1\step6_eval_lam1.0.log"`
     - Purpose: Benchmark evaluation for lambda = 1.0 across 6 canonical seeds.
     - Exit Code: 0
     - Key Numbers:
       - `baseline_off`: median 12.23%, mean 11.51 +- 1.93%, p90 44.9%, T1 0.44, worst 175.1%
       - `t9_engine`: median 12.23%, mean 11.51 +- 1.93%, p90 44.4%, T1 0.44, worst 175.1%
  7. `python scripts/round1_eval.py --tag t6_lam2.0 --model-path models/checkpoints/round1_interval_lam2.0_s42.pt --configs config/round1/baseline_off.json config/round1/t9_engine.json 2>&1 | Tee-Object -FilePath "logs\round1\step6_eval_lam2.0.log"`
     - Purpose: Benchmark evaluation for lambda = 2.0 across 6 canonical seeds.
     - Exit Code: 0
     - Key Numbers:
       - `baseline_off`: median 11.58%, mean 11.47 +- 2.15%, p90 44.5%, T1 0.43, worst 175.4%
       - `t9_engine`: median 11.58%, mean 11.47 +- 2.15%, p90 43.5%, T1 0.43, worst 175.4%
- **Log Paths**:
  - `logs/round1/step6_smoke.log`
  - `logs/round1/step6_lam0.5.log`
  - `logs/round1/step6_lam1.0.log`
  - `logs/round1/step6_lam2.0.log`
  - `logs/round1/step6_eval_lam0.5.log`
  - `logs/round1/step6_eval_lam1.0.log`
  - `logs/round1/step6_eval_lam2.0.log`
  - `logs/round1/train_interval_lam0.5_s42.json`
  - `logs/round1/train_interval_lam1.0_s42.json`
  - `logs/round1/train_interval_lam2.0_s42.json`
- **Commit Hash**: `27f725c`




