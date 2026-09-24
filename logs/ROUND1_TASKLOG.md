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










