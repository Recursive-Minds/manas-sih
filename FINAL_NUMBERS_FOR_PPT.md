# SIH PS 26168: Final Held-Out Evaluation Benchmark Numbers

This document provides the authoritative, empirical comparison between the pre-round 1 baseline model and the finalized production system on strictly held-out evaluation seeds (`[319976, 480577, 473995]`).

---

## 1. Executive Summary Table: Baseline vs Final Production

| Metric | Pre-Round 1 Base | Final Production (Round 2 Promoted) | Improvement / Delta | Data Source File |
| :--- | :--- | :--- | :--- | :--- |
| **Total Scenarios Evaluated** | 120 | 120 | 3 seeds x 40 scenarios | `results/round1/heldout_base/summary.json` & `results/round1/heldout_r2_blend180/summary.json` |
| **Median Drift (Seed Medians)** | 11.48% | 11.15% | -0.33 pp | `results/round1/heldout_base/summary.json` & `results/round1/heldout_r2_blend180/summary.json` |
| **Cross-Seed Mean +- Std** | 11.13% +- 1.50% | 10.71% +- 1.17% | -0.42 pp (std down by 22%) | `results/round1/heldout_base/summary.json` & `artifacts/heldout_seed_results.json` |
| **P90 Drift (90th Percentile)** | 37.25% | 32.91% | -4.34 pp (error tail tightened) | `results/round1/heldout_base/summary.json` & `results/round1/heldout_r2_blend180/summary.json` |
| **Tier 1 Share (< 10% Drift)** | 42.50% (51/120) | 48.33% (58/120) | +5.83 pp (+7 scenarios into Tier 1) | `results/round1/heldout_base/summary.json` & `results/round1/heldout_r2_blend180/summary.json` |
| **Unseen Trips Median (S-S3a, S-S4)** | 11.89% (n=60) | 9.66% (n=60) | -2.23 pp (breaks the 10% barrier) | `results/round1/heldout_base/baseline_off_scenarios.csv` & `results/round1/heldout_r2_blend180/r2_blend180_scenarios.csv` |
| **Beats Pure Dead-Reckoning Rate**| 83.33% (100/120) | 86.67% (104/120) | +3.34 pp | `results/round1/heldout_base/summary.json` & `artifacts/heldout_seed_results.json` |

---

## 2. Per-Seed Breakdown (Final Production System)

Source file: `artifacts/heldout_seed_results.json`

| Held-Out Seed | Map Drift Median | Pure DR Drift | Tier 1 Count (< 10%) | Sub-30% Count | Beats Pure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 319976** | 11.86% | 22.34% | 18 / 40 (45.0%) | 34 / 40 (85.0%) | 33 / 40 (82.5%) |
| **Seed 480577** | 11.15% | 23.90% | 17 / 40 (42.5%) | 34 / 40 (85.0%) | 37 / 40 (92.5%) |
| **Seed 473995** | 9.11% | 22.57% | 23 / 40 (57.5%) | 35 / 40 (87.5%) | 34 / 40 (85.0%) |
| **Average / Summary** | **10.71% +- 1.17%** | **22.93% +- 0.69%** | **19.3 / 40 (48.33%)** | **34.3 / 40 (85.83%)** | **34.7 / 40 (86.67%)** |

---

## 3. Domain Drift Comparison (Final Production System)

Source files: `results/round1/heldout_base/summary.json` vs `results/round1/heldout_r2_blend180/summary.json`

| Operational Domain | Base Model Median Drift | Final System Median Drift | Improvement |
| :--- | :--- | :--- | :--- |
| **Highway** | 9.98% | 9.47% | -0.51 pp |
| **Arterial** | 13.79% | 11.49% | -2.30 pp |
| **Urban** | 9.43% | 8.09% | -1.34 pp |
| **Mixed (Challenging)** | 12.88% | 9.47% | -3.41 pp |

---

## 4. Final Architecture Recipe

Active profile: `config/round1/production.json`

1. **Velocity Estimator Checkpoint**: `models/checkpoints/round1_interval_lam0.5_s42.pt` (dual-expert MoE trained with symmetric interval loss lambda = 0.5, seed 42).
2. **Speed Scale Adaptation**: T7 online per-band speed calibration (`online_calib: enabled`) blended with 180s causal pre-blackout GNSS-to-AI ratio (`scale_level: enabled, source: blend`).
3. **Map Matching Anchor**: T8 post-turn junction anchor snapping along the matched road corner (`junction: enabled`).
4. **Heading Seeding**: Causal 1 Hz geometric vector displacement bearing (`entry_doppler_bearing: false`).
5. **Streaming/Batch Parity**: 100% bit-identical (0.0000 m endpoint difference and 0.0000 m max trajectory difference across all canonical scenarios).
