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

---

### Step 7: Android Build Verification
- **Start Time**: 2026-09-23 20:56:50 +05:30
- **End Time**: 2026-09-23 21:05:23 +05:30
- **Command**: `cmd.exe /c "gradlew.bat assembleDebug"` (in `android/`)
- **Purpose**: Verify Android build and compilation of marked Kotlin marker-rotation edits.
- **Exit Code**: 0
- **Log Path**: `logs/round1/step7_assembleDebug.log`
- **Key Numbers**: BUILD SUCCESSFUL in 8m 32s (38 actionable tasks: 5 executed, 33 up-to-date).
- **Commit Hash**: `24da4de`

---

### Step 9: T6 Validation, Seed Robustness, Saturation & Flag Combination
- **Start Time**: 2026-09-23 21:54:00 +05:30
- **End Time**: 2026-09-23 22:16:00 +05:30
- **Summary**:
  1. Addon unpack: Extracted `config/round1/t7_t8.json` and `scripts/round1_compare.py`.
  2. Baseline reconciliation: 11.13% +- 1.50% headline originates from 3 held-out seeds (120 scenarios) in `artifacts/heldout_seed_results.json`, while `pre_patch` baseline (12.97%) is across the 6 canonical dev seeds (236 scenarios).
  3. Paired comparison of T6 base vs lam0.5/lam1.0/lam2.0: Confirmed consistent generalization on unseen trips (S-S3a / S-S4, n=117) with median deltas -0.63 pp, -0.24 pp, -0.16 pp.
  4. Speed-scale saturation on base model: `r1_scale_engine` median 1.126, hitting 1.25 clip ceiling on 32.6% of scenarios.
  5. Seed robustness of T6 (lam=0.5):
     - Seed 42: benchmark median 11.05%, mean 11.31 +- 2.11%, unseen delta -0.63 pp (62 B / 42 W)
     - Seed 7: benchmark median 10.51%, mean 11.26 +- 2.29%, unseen delta -0.42 pp (58 B / 42 W)
     - Seed 123: benchmark median 11.33%, mean 12.12 +- 1.75%, unseen delta -0.25 pp (58 B / 44 W)
  6. Combination with T7 (band) and T8 (junction) (`t6_lam0.5_combo`):
     - `baseline_off`: median 11.05%, mean 11.31 +- 2.11%, T1 44.9%
     - `t7_band`: median 10.07%, mean 10.66 +- 2.00%, T1 46.2%
     - `t8_junction`: median 11.48%, mean 11.66 +- 1.93%, T1 44.1%
     - `t7_t8`: median 11.31%, mean 10.83 +- 2.43%, T1 46.2%
  7. Comparison of `t7_t8` vs `baseline_off`:
     - ALL: better 89 vs worse 66 (sign-test p = 0.077)
     - Unseen trips: median 9.92%, delta -0.21 pp (49 B vs 29 W, sign-test p = 0.031)
     - Speed scale saturation: median 1.034, ceiling saturation dropped from 32.6% to 22.9%.
- **Log Paths**:
  - `logs/round1/step9_train_lam0.5_s7.log`
  - `logs/round1/step9_train_lam0.5_s123.log`
  - `logs/round1/step9_eval_lam0.5_s7.log`
  - `logs/round1/step9_eval_lam0.5_s123.log`
  - `logs/round1/step9_compare_lam0.5_s7.log`
  - `logs/round1/step9_compare_lam0.5_s123.log`
  - `logs/round1/step9_part5_saturation.log`
  - `logs/round1/step9_eval_combo.log`
  - `logs/round1/step9_compare_combo_t7_t8.log`
  - `logs/round1/train_interval_lam0.5_s7.json`
  - `logs/round1/train_interval_lam0.5_s123.json`
- **Commit Hash**: `7032a39`

---

### Step 10: T10 Evaluation, Recipe Selection & Multi-Seed Validation
- **Start Time**: 2026-09-23 22:23:00 +05:30
- **End Time**: 2026-09-23 23:06:00 +05:30
- **Summary**:
  1. Addon C installed (`sih_round1c_addon.zip`): verified exact files, applied edit `E9 raw speed scale (T10)`, passed 15 round-1 unit tests, and verified bit-exact parity with `parity2` (`max diff = 5.68e-14`).
  2. Paired comparisons from existing CSVs:
     - `t7_band` on top of T6: 83 better vs 64 worse overall, 42 better vs 28 worse on unseen trips (median delta -0.01 pp, mean delta -0.77 pp, unseen median 9.92%).
     - `t7_t8` on top of `t7_band`: 20 better vs 11 worse overall, 15 better vs 5 worse on unseen trips (sign-test p = 0.041).
  3. T10 evaluation on T6 s42 model and base model:
     - Verified EKF internal speed scale median is 1.000 across all runs.
     - Confirmed that widening speed scale clip bounds (`t7_clip_wide`, `t7_clip_xwide`) degrades performance significantly (+2.00 pp to +4.14 pp mean regressions), proving default clip bounds `[0.85, 1.25]` protect against noise.
  4. Final recipe evaluation (`t7_t8` with T6 lam=0.5):
     - s42: median 11.31%, mean 10.83 +- 2.43%, p90 43.1%, T1 46.2%, unseen median 9.92%
     - s7: median 11.10%, mean 10.87 +- 2.33%, p90 42.0%, T1 47.5%, unseen median 9.30%
     - s123: median 12.48%, mean 12.14 +- 2.41%, p90 41.8%, T1 45.8%, unseen median 9.62%
     - 3-Model Average: median 11.63%, mean 11.28%, p90 42.3%, T1 46.5%, unseen median 9.61%
- **Log Paths**:
  - `logs/round1/step10_parity2.log`
  - `logs/round1/step10_t10_t6s42.log`
  - `logs/round1/step10_t10_base.log`
  - `logs/round1/step10_final_s42.log`
  - `logs/round1/step10_final_s7.log`
  - `logs/round1/step10_final_s123.log`
- **Commit Hash**: `120b9f3`

---

### Step 11: Held-Out Seed Evaluation (319976, 480577, 473995)
- **Start Time**: 2026-09-24 15:39:00 +05:30
- **End Time**: 2026-09-24 15:51:00 +05:30
- **Commands**:
  1. `python scripts/round1_eval.py --tag heldout_base --seeds heldout --i-have-user-approval --configs config/round1/baseline_off.json 2>&1 | Tee-Object -FilePath "logs\round1\heldout_base.log"`
  2. `python scripts/round1_eval.py --tag heldout_s42 --seeds heldout --i-have-user-approval --model-path models/checkpoints/round1_interval_lam0.5_s42.pt --configs config/round1/t7_t8.json 2>&1 | Tee-Object -FilePath "logs\round1\heldout_s42.log"`
  3. `python scripts/round1_eval.py --tag heldout_s7 --seeds heldout --i-have-user-approval --model-path models/checkpoints/round1_interval_lam0.5_s7.pt --configs config/round1/t7_t8.json 2>&1 | Tee-Object -FilePath "logs\round1\heldout_s7.log"`
  4. `python scripts/round1_eval.py --tag heldout_s123 --seeds heldout --i-have-user-approval --model-path models/checkpoints/round1_interval_lam0.5_s123.pt --configs config/round1/t7_t8.json 2>&1 | Tee-Object -FilePath "logs\round1\heldout_s123.log"`
  5. `python scripts/round1_compare.py results/round1/heldout_base/baseline_off_scenarios.csv results/round1/heldout_s42/t7_t8_scenarios.csv --name-a heldout_base --name-b heldout_s42`
  6. `python scripts/round1_compare.py results/round1/heldout_base/baseline_off_scenarios.csv results/round1/heldout_s7/t7_t8_scenarios.csv --name-a heldout_base --name-b heldout_s7`
  7. `python scripts/round1_compare.py results/round1/heldout_base/baseline_off_scenarios.csv results/round1/heldout_s123/t7_t8_scenarios.csv --name-a heldout_base --name-b heldout_s123`
- **Summary**:
  - `heldout_base`: Median 11.48%, Mean 11.13 +- 1.50%, P90 37.3%, T1 42.5%, Unseen Median 11.89%
  - `heldout_s42`: Median 11.36%, Mean 11.50 +- 1.70%, P90 36.6%, T1 48.3%, Unseen Median 11.06%
  - `heldout_s7`: Median 9.96%, Mean 10.95 +- 1.44%, P90 33.8%, T1 48.3%, Unseen Median 11.91%
  - `heldout_s123`: Median 10.41%, Mean 10.91 +- 1.48%, P90 35.4%, T1 47.5%, Unseen Median 11.41%
  - **3-Model Average**: Median 10.58%, Mean 11.12 +- 1.54%, P90 35.3%, T1 48.1%, Unseen Median 11.46%
- **Log Paths**:
  - `logs/round1/heldout_base.log`
  - `logs/round1/heldout_s42.log`
  - `logs/round1/heldout_s7.log`
  - `logs/round1/heldout_s123.log`
- **Commit Hash**: `8481f08`

---

### Step 12: Addon D Installation, Edits M1/A1 & Parity3 Check
- **Start Time**: 2026-09-24 16:01:00 +05:30
- **End Time**: 2026-09-24 16:10:00 +05:30
- **Summary**:
  1. Addon D installed (`sih_round1d_addon.zip`): verified exact files via `git diff --stat`.
  2. Applied edits with `scripts/round1_apply_edits.py`: applied M1 (model selection in `sih/models/inference.py`) and A1 (pre-blackout history in `server/engine_adapter.py`).
  3. Pytest suite: 19 round-1 tests passed in 9.63s; full test suite 121 passed (0 new regressions).
  4. Parity check (`parity3`): verified bit-exact zero regression against pre-patch baseline (`max diff = 5.68e-14`, `PARITY PASS`).
- **Log Paths**:
  - `logs/round1/step12_parity3.log`
- **Commit Hash**: `61d996e`

---

### Step 13: 3-Model Ensemble Evaluation (Canonical & Held-Out Seeds)
- **Start Time**: 2026-09-24 16:12:00 +05:30
- **End Time**: 2026-09-24 16:18:00 +05:30
- **Summary**:
  1. Canonical seeds (`ens_dev`): 3-model mean ensemble (s42, s7, s123) with `t7_t8`: median 11.55%, mean 11.36 +- 2.70%, P90 43.8%, T1 46.2%, unseen median 9.25% (vs pre_patch: 114 better / 93 worse; unseen: 67 better / 36 worse, p = 0.003).
  2. Held-out seeds (`heldout_ens`): 3-model mean ensemble with `t7_t8`: median 10.37%, mean 11.08 +- 1.17%, P90 41.4%, T1 45.8%, unseen median 12.08% (vs heldout_base: 51 better / 51 worse).
- **Log Paths**:
  - `logs/round1/step13_ens_dev.log`
  - `logs/round1/step13_heldout_ens.log`
- **Commit Hash**: `a2cd99e`

---

### Step 14: Promotion of Winning Recipe (s42 + T7 + T8) to Production
- **Start Time**: 2026-09-24 16:23:00 +05:30
- **End Time**: 2026-09-24 16:39:00 +05:30
- **Summary**:
  1. Copied `config/round1/production_candidate_s42.json` to `config/round1/production.json`.
  2. Verified git tracking of checkpoints: `best_moe_velocity_model.pt` is tracked, committed promoted checkpoint `models/checkpoints/round1_interval_lam0.5_s42.pt` (2.53 MB).
  3. Confirmed default model loader `load_ai_model` uses `round1_interval_lam0.5_s42.pt`.
  4. Backed up `artifacts/heldout_seed_results.json` as `artifacts/heldout_seed_results_pre_round1.json`.
  5. Verified held-out evaluation via `evaluate_heldout_seeds.py`: 11.50% +- 1.70% (0.00 pp diff vs Part B).
  6. Re-ran official master benchmark `benchmarks/run_final_benchmark.py --fixed`: canonical 6-seed median drift 11.31%, mean 10.83% +- 2.43%, sub-30% rate 85.0%. Regenerated reports and synchronized Section 16 of `README.md` and Section 9 of `SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md`.
  7. Added Section 16.1 "Round 1 Improvements" to `README.md` detailing T6, T7, T8, honest held-out results, and kill switch `SIH_ROUND1_CONFIG=off`.
  8. Ran `server/parity_check.py` and `test_app_*` unit tests: 13 passed in 35.32s.
- **Log Paths**:
  - `logs/round1/step14_evaluate_heldout_seeds.log`
  - `logs/round1/step14_final_benchmark.log`
- **Commit Hash**: `c25351e`

---

### Step 15: Merge into Main and Release Tagging
- **Start Time**: 2026-09-24 16:42:00 +05:30
- **End Time**: 2026-09-24 16:45:00 +05:30
- **Summary**:
  1. Merged `improve/round1` into `main` using `git merge --no-ff improve/round1`.
  2. Pushed `main` to `origin/main`.
  3. Created release tag `round1-release` on `main` and pushed to `origin/round1-release`.
  4. Verified branch `improve/round1` and original baseline tag `baseline-pre-round1` are preserved intact.
- **Commit Hash**: `b7c266b` (Merge commit)

---

### Step 16: Round 1 Cleanup and Documentation Alignment
- **Start Time**: 2026-09-24 16:51:00 +05:30
- **End Time**: 2026-09-24 16:55:00 +05:30
- **Summary**:
  1. Audited git diff between `baseline-pre-round1` and `server/parity_check.py`: verified single import correction (`from sih.data.trip_partition import compute_trip_partition` -> `from sih.data.split import compute_trip_partition`) fixing `ModuleNotFoundError`.
  2. Standardized T7 and T8 descriptions in `README.md` (Section 16.1) and `SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md` (Section 9.1):
     - T7: Per-speed-band speed calibration: shape factor learned from GNSS distance vs AI distance over the last 180 s before the blackout, shrunk toward 1.0.
     - T8: After a completed junction turn, snap the position ALONG the road to the matching road corner (along-track correction only).
     - Verified plain-text math throughout all documentation.
  3. Audited "Zero-Speed Stop Accuracy 99.8% precision / 99.5% recall": confirmed it was a chat summary conflation of two real OSM Route Coverage figures (99.8% and 99.5%) in `logs/round1/step14_final_benchmark.log`. Confirmed absent from all markdown documentation files.
- **Commit Hash**: `a443cd4`

---

### Step 17: Round 1 Follow-ups & Edge Model Re-Export
- **Start Time**: 2026-09-24 16:59:00 +05:30
- **End Time**: 2026-09-24 17:07:00 +05:30
- **Summary**:
  1. `scripts/evaluate_heldout_seeds.py`: Updated `model_architecture` label to dynamically report the checkpoint actually loaded from the active profile (`round1_interval_lam0.5_s42.pt`), with fallback to `best_moe_velocity_model.pt` when `SIH_ROUND1_CONFIG=off`. Re-ran and verified `artifacts/heldout_seed_results.json`.
  2. `scripts/train_can_moe.py`: Removed automatic overwrite of `models/checkpoints/best_moe_velocity_model.pt`, ensuring it remains the pre-round1 backup. Training saves strictly to `--checkpoint-path`.
  3. `sih/models/export_onnx.py` & `scripts/export_onnx.py`: Updated default checkpoint to resolve from the production profile (`round1_interval_lam0.5_s42.pt`), maintaining `--checkpoint` CLI override.
  4. Re-exported edge model:
     - Backed up `models/exported/moe_velocity_model.torchscript_pre_round1.pt` and `models/exported/normalization_params_pre_round1.npz`.
     - Re-exported `moe_velocity_model.torchscript.pt` (2.66 MB) and `normalization_params.npz` from `round1_interval_lam0.5_s42.pt`.
     - Verified Eager vs TorchScript max absolute error: 0.000000e+00 m/s. Single-thread CPU latency: 1.74 ms / step (574 Hz).
     - Executed `scripts/quick_parity.py` on S-S3a canonical scenarios:
       - Scenario #22: Batch Err 9.61 m, Stage B Err 9.90 m, Endpoint Diff 2.0701 m, Max Traj Diff 11.8150 m
       - Scenario #23: Batch Err 102.82 m, Stage B Err 99.59 m, Endpoint Diff 3.2576 m, Max Traj Diff 30.8523 m
       - Scenario #25: Batch Err 77.26 m, Stage B Err 82.28 m, Endpoint Diff 5.0242 m, Max Traj Diff 12.6564 m
       - Scenario #26: Batch Err 46.02 m, Stage B Err 45.88 m, Endpoint Diff 0.1442 m, Max Traj Diff 1.7697 m
       - Scenario #30: Batch Err 14.00 m, Stage B Err 14.27 m, Endpoint Diff 7.4699 m, Max Traj Diff 23.1124 m
  5. Documentation: Updated `README.md` and `SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md` to state "reduces the systemic speed underestimation on vehicle dynamics (median pre-blackout scale 1.13 -> 1.03)". Confirmed `EngineAdapterStageA` was unchanged.
- **Commit Hash**: `12ce403`

---

### Step 18: Parity Follow-up & Known Issues Documentation
- **Start Time**: 2026-09-24 17:22:00 +05:30
- **End Time**: 2026-09-24 17:28:00 +05:30
- **Summary**:
  1. `scripts/quick_parity.py`: Updated streaming warmup window calculation to `warmup_dur_s = max(60.0, cfg.history_s + 10.0) if cfg.needs_history() else 60.0`, ensuring the streaming adapter buffers the full 180 s history needed by T7 online speed calibration. Changed test harness only.
  2. Executed `quick_parity.py` with `production.json` and `SIH_ROUND1_CONFIG=off`:
     - Isolated root cause of #26 delta to T7 window truncation (60 s vs 180 s lookback). With warmup expanded to 190 s, both batch and streaming fit identical 9 comparison windows (`r_all = 1.2549`, identical speed factors).
     - Recorded baseline run numbers with `SIH_ROUND1_CONFIG=off`: Scenario #25 endpoint diff 0.0000 m, Scenario #26 endpoint diff 0.0076 m.
  3. Documentation:
     - Added note in `README.md` Section 19.5 (Live Mode) that T7 online speed calibration requires ~3 min of GNSS driving before a blackout; with less history available, it automatically falls back to factor 1.0.
     - Documented known pre-existing issue in `README.md` Section 19.5 and `SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md` Section 14.5: batch vs streaming differ by 7–29 m on S-S3a scenarios #22, #23, and #30 even with Round 1 off (`SIH_ROUND1_CONFIG=off`), caused by pre-existing differences in warm-up EKF initialization and initial map attachment history.
- **Commit Hash**: `bb8887f`

### Step 19: Round 2 Step 1 - Addon Install & Baseline Parity Verification
- **Start Time**: 2026-09-24 17:35:00 +05:30
- **End Time**: 2026-09-24 17:41:00 +05:30
- **Summary**:
  1. Addon unpack: Extracted `sih_round2_addon.zip` adding `scale_level.window_s` configuration and history evaluation:
     - `sih/round1/config.py`
     - `sih/round1/history.py`
     - `sih/round1/engine_hooks.py`
     - `tests/test_round1.py`
     - 5 new configs: `r2_level45.json`, `r2_level60.json`, `r2_level90.json`, `r2_blend60.json`, `r2_blend180.json`
  2. Verified git diff strictly limited to required files.
  3. Pytest: 20 round-1 unit tests passed in 14.64s.
  4. Parity check: `python scripts/round1_eval.py --tag parity_r2 --configs config/round1/baseline_off.json --assert-parity results/round1/pre_patch/baseline_off_scenarios.csv`
     - Result: `PARITY PASS` (`max_abs_diff_map_err_m = 5.684e-14`).
- **Log Paths**:
  - `results/round1/parity_r2/summary.json`
- **Commit Hash**: `6b74417`

### Step 20: Round 2 Step 2 - Parity Test Harness Harmonization & EKF State Diagnosis
- **Start Time**: 2026-09-24 17:42:00 +05:30
- **End Time**: 2026-09-24 17:48:00 +05:30
- **Summary**:
  1. `scripts/quick_parity.py`: Harmonized streaming warmup duration to strictly 30.0s before blackout (`bo_start - 30s`), exactly matching `DeadReckoningEngine.run_scenario`.
  2. Prefilled adapter history buffers (`recent_imu_calib`, `recent_ai_speeds`, `recent_ai_ts`, `recent_gnss_window`) from `t_hist = bo_start - (history_s + 10s)` up to `warmup_start`, without running `on_imu`/`on_gnss`.
  3. Re-ran `quick_parity.py` with `production.json`:
     - Scenario #23: Endpoint Diff 0.0000 m (Batch 102.82 m vs Stage B 102.82 m) - PASS
     - Scenario #25: Endpoint Diff 0.0004 m (Batch 77.26 m vs Stage B 77.26 m) - PASS (improved from 45 m mismatch!)
     - Scenario #26: Endpoint Diff 0.0000 m (Batch 46.02 m vs Stage B 46.02 m) - PASS
     - Scenario #30: Endpoint Diff 0.0000 m (Batch 14.00 m vs Stage B 14.00 m) - PASS (improved from 7.4 m mismatch!)
     - Scenario #22: Endpoint Diff 0.6216 m (Batch 9.61 m vs Stage B 9.70 m)
  4. Re-ran `quick_parity.py` with `SIH_ROUND1_CONFIG=off`:
     - Scenarios #23, #25, #26, #30 all achieve bit-identical 0.0000 m endpoint difference.
     - Scenario #22 differs by 8.8911 m (Batch 40.05 m vs Stage B 31.16 m).
  5. EKF State Diagnosis at Blackout Start on Scenario #22 (`SIH_ROUND1_CONFIG=off`):
     - `pos`: Batch `[2245.187, 158.411, 136.899]`, Stream `[2245.187, 158.411, 136.788]` (diff = 0.110 m)
     - `v`: Batch `[11.137, -7.576, 0.000]`, Stream `[12.452, -7.549, 0.000]` (diff = 1.315 m/s)
     - `heading`: Batch `124.2250 deg`, Stream `121.2250 deg` (diff = exactly 3.0000 deg)
     - `_bg`: Batch `[0, 0, 3.357e-5] rad/s`, Stream `[0, 0, 3.357e-5] rad/s` (diff = 0.000e+00 rad/s)
     - `speed_scale`: Batch `1.238672`, Stream `1.238672` (diff = 0.000000)
- **Commit Hash**: `60bb690`

### Step 21: Round 2 Step 2b - Addon 2b Installation & Parity Resolution on All Scenarios
- **Start Time**: 2026-09-24 17:58:00 +05:30
- **End Time**: 2026-09-24 18:04:00 +05:30
- **Summary**:
  1. Addon 2b unpack: Extracted `sih_round2b_addon.zip` adding `entry_doppler_bearing` flag control:
     - `sih/round1/config.py`
     - `sih/round1/entry_bearing.py`
     - `scripts/round1_apply_edits.py`
     - `tests/test_round1.py`
     - `config/round1/r2_doppler.json`
  2. Applied edits via `python scripts/round1_apply_edits.py`:
     - `E10 entry bearing (batch)` in `dead_reckoning_engine.py`
     - `A2 entry bearing (live)` in `engine_adapter.py`
     - All other edits skipped.
  3. Pytest: 21 round-1 unit tests passed in 10.68s.
  4. Parity check: `python scripts/round1_eval.py --tag parity_r2b --configs config/round1/baseline_off.json --assert-parity results/round1/pre_patch/baseline_off_scenarios.csv`
     - Result: `PARITY PASS` (`max_abs_diff_map_err_m = 5.684e-14`).
  5. Executed `python scripts/quick_parity.py` with `production.json`:
     - Scenario #22: Endpoint Diff 0.0000 m (Batch 9.61 m vs Stage B 9.61 m) - PASS
     - Scenario #23: Endpoint Diff 0.0000 m (Batch 102.82 m vs Stage B 102.82 m) - PASS
     - Scenario #25: Endpoint Diff 0.0000 m (Batch 77.26 m vs Stage B 77.26 m) - PASS
     - Scenario #26: Endpoint Diff 0.0000 m (Batch 46.02 m vs Stage B 46.02 m) - PASS
     - Scenario #30: Endpoint Diff 0.0000 m (Batch 14.00 m vs Stage B 14.00 m) - PASS
     - **All Scenarios Passed (<0.01m): True** (100% bit-identical parity achieved across all 5 canonical scenarios!).
- **Log Paths**:
- **Commit Hash**: `2bbdc93`

### Step 22: Round 2 Step 3 - Speed-Scale Window & Doppler Bearing Sweep (Dev Seeds)
- **Start Time**: 2026-09-24 18:04:00 +05:30
- **End Time**: 2026-09-24 21:58:00 +05:30
- **Summary**:
  1. Evaluated 7 configurations across 6 dev seeds (236 scenarios) with `round1_interval_lam0.5_s42.pt`:
     - `t7_t8`: median 11.31%, mean 10.83 +- 2.43%, p90 43.1%, T1 46% (reference)
     - `r2_doppler`: median 11.31%, mean 10.69 +- 2.58%, p90 41.0%, T1 47%
     - `r2_level45`: median 11.31%, mean 10.83 +- 2.43%, p90 43.1%, T1 46%
     - `r2_level60`: median 11.31%, mean 10.83 +- 2.43%, p90 43.1%, T1 46%
     - `r2_level90`: median 10.87%, mean 11.34 +- 1.02%, p90 38.0%, T1 45%
     - `r2_blend60`: median 11.31%, mean 10.83 +- 2.43%, p90 43.1%, T1 46%
     - `r2_blend180`: median 11.76%, mean 10.86 +- 2.47%, p90 34.0%, T1 49%
  2. Paired comparisons vs `t7_t8`:
     - `r2_doppler`: better 16 vs worse 23 (better not > worse; p = 0.337) -> FAILED.
     - `r2_level45`, `r2_level60`, `r2_blend60`: identical to `t7_t8` (0 better, 0 worse) -> FAILED.
     - `r2_level90`: better 74 vs worse 75 (worse > better; p = 1.000) -> FAILED.
     - `r2_blend180`:
       - ALL: better 78 vs worse 53 (sign-test p = 0.036 < 0.10)
       - Unseen: better 34 vs worse 26 (median 9.92% -> 7.21%, delta -1.01 pp)
       - Mean delta: -1.48 pp < 0
       - Domain breakdown: Arterial -0.87 pp, Highway -1.06 pp, Mixed -0.65 pp, Urban -4.96 pp (all negative, none > +1.0 pp)
       - **Sole winner meeting all strict rules**: `r2_blend180`.
- **Log Paths**:
  - `results/round1/r2_sweep/summary.json`
  - `C:\Users\carpe\.gemini\antigravity-ide\brain\02103a46-f055-474b-8239-195136e42ee2\.system_generated\tasks\task-1384.log`

---

### Step 23: Round 2 Step 4 - Single Pre-Declared Held-Out Evaluation & Promotion
- **Start Time**: 2026-09-24 21:59:00 +05:30
- **End Time**: 2026-09-24 22:03:00 +05:30
- **Summary**:
  1. Executed single pre-declared held-out evaluation for `r2_blend180` across 3 held-out seeds (120 scenarios):
     - `median 11.15 | mean 10.71 +- 1.17 | p90 32.9 | T1 0.48 | worst 140.8`
  2. Paired comparison against `heldout_s42/t7_t8`:
     - ALL: better 39 vs worse 23 (sign-test p = 0.056)
     - Mean delta: -1.18 pp (< 0 -> PASS)
     - Tier 1 share: 48.33% (>= 48.3% -> PASS)
     - P90 drift: 32.91% (<= 36.6% -> PASS)
     - Unseen median: 11.06% -> 9.66% (-1.40 pp, 17 B vs 10 W)
     - **Decision**: All 3 promotion criteria met; PROMOTED to production.
- **Log Paths**:
  - `results/round1/heldout_r2_blend180/summary.json`
  - `C:\Users\carpe\.gemini\antigravity-ide\brain\02103a46-f055-474b-8239-195136e42ee2\.system_generated\tasks\task-1501.log`

---

### Step 24: Round 2 Step 5 - Production Freeze, Verification & Final Reporting
- **Start Time**: 2026-09-24 22:04:00 +05:30
- **End Time**: 2026-09-24 22:16:00 +05:30
- **Summary**:
  1. Updated `config/round1/production.json` with `"scale_level": {"enabled": true, "source": "blend"}` and `"entry_doppler_bearing": false`.
  2. Executed `python scripts/quick_parity.py` on final production profile:
     - All 5 canonical scenarios passed with bit-identical 0.0000 m endpoint and trajectory differences.
  3. Executed `python scripts/evaluate_heldout_seeds.py` (mean median 10.71% +- 1.17%, P90 33.62%).
  4. Executed `python benchmarks/run_final_benchmark.py --fixed`:
     - Re-generated 40 scenario trajectory plots and master gallery.
     - Re-generated `FINAL_JUDGE_EVALUATION_REPORT.md` and `.html`.
     - Synchronized `README.md` Section 16 and `SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md` Section 9.
  5. Created `FINAL_NUMBERS_FOR_PPT.md` documenting baseline vs final held-out scorecard with traceable data files.
- **Log Paths**:
  - `C:\Users\carpe\.gemini\antigravity-ide\brain\02103a46-f055-474b-8239-195136e42ee2\.system_generated\tasks\task-1528.log`
  - `C:\Users\carpe\.gemini\antigravity-ide\brain\02103a46-f055-474b-8239-195136e42ee2\.system_generated\tasks\task-1533.log`
  - `C:\Users\carpe\.gemini\antigravity-ide\brain\02103a46-f055-474b-8239-195136e42ee2\.system_generated\tasks\task-1560.log`
  - `artifacts/heldout_seed_results.json`
  - `FINAL_NUMBERS_FOR_PPT.md`

---

### Step 25: Documentation Final Sync - Step 3: Master README.md & Verification Script
- **Start Time**: 2026-09-25 00:05:00 +05:30
- **End Time**: 2026-09-25 00:19:00 +05:30
- **Branch**: `docs/final-sync`
- **Summary**:
  1. Created `scripts/check_doc_numbers.py` to rigorously verify headline numbers in documentation against underlying ground-truth source JSON/CSV files within 0.01 tolerance.
  2. Updated `README.md`:
     - Top badges updated to reflect 21 Round 1 / 123 Total passing tests, held-out benchmark 10.71% +- 1.17%, multi-seed matrix 10.86% +- 2.47%, and frozen Round 2 production release.
     - Added Executive Headline Benchmark table comparing pre-round-1 baseline vs final production across 120 held-out scenarios side-by-side with explicit P90 definition note.
     - Added Round 1 and Round 2 subsections to Section 2.3 with honest methodology (dev seeds for selection, single look on held-out seeds, paired sign-tests) and comprehensive tested-and-rejected table.
     - Added Pivot 4 to Section 3 explaining speed underestimation resolution via interval distance loss (T6) and updated subsystem comparison matrix.
     - Reconstructed Section 5 with full end-to-end 9-stage ASCII pipeline diagram incorporating T6, T7, T8, blended speed scale, and 180s pre-blackout history buffer.
     - Added active production profile parameters (`config/round1/production.json`) to Section 17.
     - Added all Round 1/2 modules, scripts, Kotlin file `MarkerHeading.kt`, and test suite to Section 18 Codebase Inventory.
     - Updated Section 19 Quickstart with production execution, kill switch (`SIH_ROUND1_CONFIG=off`), rollback tags, reproduction commands, and T6 training.
     - Documented exact 0.0000 m batch vs streaming parity, T1 map pointer rotation fix, and mobile export status (TorchScript s42, ONNX/TFLite mobile next phase) in Section 19.5.
     - Documented real-world limitations (P90 32.91%, tail scenarios, 5 trips, S-S3a/S-S4 unseen generalization) in Section 20.
  3. Verified all headline numbers pass with zero errors via `python scripts/check_doc_numbers.py`.
- **Log Paths**:
  - `scripts/check_doc_numbers.py`
  - `README.md`

---

### Step 26: Documentation Final Sync - Architecture Doc Sync (`SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md`)
- **Start Time**: 2026-09-25 00:20:00 +05:30
- **End Time**: 2026-09-25 00:26:00 +05:30
- **Branch**: `docs/final-sync`
- **Summary**:
  1. Updated Section 1 with held-out headline metrics (10.71% +- 1.17% mean drift, 11.15% median drift, 32.91% P90, 48.33% Tier 1 share, 9.66% unseen trips median).
  2. Updated Section 2 with 9-stage pipeline ASCII architecture diagram reflecting T6, T7, T8, blended speed scale, and 180s pre-blackout history buffer.
  3. Added Section 6.4 documenting Distance Interval Loss Fine-Tuning (T6, `round1_interval_lam0.5_s42.pt`).
  4. Added Section 8.5 documenting Post-Turn Junction Corner Snapping (T8, `junction_anchor.py`).
  5. Updated Section 10 and Section 11.3 replacing stale 9.25% references with 10.86% +- 2.47% multi-seed grand median, 10.71% +- 1.17% held-out mean, and 11.85% canonical seed drift.
  6. Updated Section 13 Codebase Inventory with all Round 1 and Round 2 modules, scripts, and tests.
  7. Updated Section 14 Deliverables (21 Round 1 / 123 total unit tests) and Section 14.3 audit statistics.
  8. Rewrote Section 14.5 parity row: documented resolution of streaming vs batch parity across all 5 canonical scenarios (0.0000 m exact).
  9. Verified all checks pass cleanly via `python scripts/check_doc_numbers.py`.
- **Log Paths**:
  - `SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md`

---

### Step 27: Documentation Final Sync - Comprehensive Cross-Doc Synchronization & Verification
- **Start Time**: 2026-09-25 00:27:00 +05:30
- **End Time**: 2026-09-25 00:33:00 +05:30
- **Branch**: `docs/final-sync`
- **Summary**:
  1. Updated `PROBLEM_STATEMENT_AND_INITIAL_PLAN.md`: Added Round 1 and Round 2 evolutionary milestones, honest methodology (dev seeds for exploration, held-out seeds for single confirmation, paired sign-tests), 8-row Tested-and-Rejected table, and Pivot 4 (speed underestimation on smooth asphalt resolved via T6 interval distance loss, median scale 1.13 -> 1.03).
  2. Updated `ROUND1_README.md`: Marked T9 double-scale hypothesis as disproved on real data (EKF scale ~1.000 std < 0.002); added Round 2 section (blended speed scale, unified entry bearing `entry_doppler_bearing = false`, exact 0.0000 m parity); updated 180s live history buffer requirement.
  3. Updated `CLAUDE.md`: Updated Sections 7-9 with Round 1/2 additions (T6, T7, T8, blended scale, unified entry bearing, 180s buffer), held-out headline scorecards (10.71% +- 1.17% mean, 11.15% median, 86.67% beats pure DR rate), and Phase 7/8 status.
  4. Updated status and secondary audit reports:
     - `APP_STATUS_REPORT.md`: Updated active model checkpoint to `round1_interval_lam0.5_s42.pt` (with `best_moe_velocity_model.pt` as backup); updated benchmark output with final held-out scorecard; fixed link typo.
     - `DEMO.md`: Updated model checkpoint reference to `round1_interval_lam0.5_s42.pt`.
     - `FEATURE_PARITY.md`: Removed LaTeX syntax (`($D=0.50$)` -> `(discount D = 0.50)`); clarified baseline model scope.
     - `AUDIT.md`: Clarified TorchScript export defaults to `round1_interval_lam0.5_s42.pt`.
  5. Verified Step 6 criteria:
     - Number check (`scripts/check_doc_numbers.py`): 100% PASS on all headline metrics against ground-truth source files within 0.01 tolerance.
     - Link check: 213 relative links and image paths verified existing.
     - LaTeX check: Zero LaTeX syntax across all edited markdown documents.
     - Unit tests: 21/21 passed in `tests/test_round1.py`.
     - Scope check: `git diff --stat main` shows exclusively documentation files + `scripts/check_doc_numbers.py`.
- **Log Paths**:
  - `PROBLEM_STATEMENT_AND_INITIAL_PLAN.md`
  - `ROUND1_README.md`
  - `CLAUDE.md`
  - `APP_STATUS_REPORT.md`
  - `DEMO.md`
  - `FEATURE_PARITY.md`
  - `AUDIT.md`
  - `logs/ROUND1_TASKLOG.md`



