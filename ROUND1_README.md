# Round 1: accuracy and reliability pack

Everything in this pack sits behind a flag, and every flag is OFF by default.
With all flags OFF the engine runs exactly the old code path, because
`self.r1 is None`. `scripts/round1_eval.py --assert-parity` checks this on the real benchmark.

## What is inside

| ID | What it does | Flag (config/round1/*.json) | Files |
|---|---|---|---|
| T1 | Fixes the pointer turning the wrong way. osmdroid's `Marker.rotation` turns counter-clockwise, but our headings are compass bearings (clockwise). | Kotlin constant `MarkerHeading.OSMDROID_ROTATION_IS_CCW` | `android/.../ui/MarkerHeading.kt` and 3 marked lines in `MainActivity.kt` |
| T2 | Worst-scenario autopsy that splits the error into speed, stop-creep and heading parts, and runs the T9 check. | none (read-only) | `scripts/round1_autopsy.py` |
| T3 | Sticky stop detector with hysteresis. It kills the AI "creep" speed while the vehicle idles at a stop. | `stop.enabled` | `sih/round1/stop_detector.py` |
| T4 | Estimates the gyro yaw scale factor from GNSS turns before the blackout. | `gyro_scale.enabled` | `sih/round1/gyro_scale.py` |
| T5 | Alternative speed modes: `hold_entry`, and `entry_offset_decay` (the entry-speed offset fades with tau = 20 s). | `speed_mode.mode` | `sih/round1/engine_hooks.py` |
| T6 | Interval (distance) loss with alpha emulation, plus a fine-tune script. It never writes the canonical checkpoint. | training CLI `--lam` | `sih/models/interval_loss.py`, `interval_dataset.py`, `scripts/train_interval_moe.py` |
| T7 | Online per-speed-band speed calibration. It learns the shape of the AI speed error from ~180 s of GNSS positions before the blackout. | `online_calib.enabled` | `sih/round1/online_speed_calib.py` |
| T8 | Along-track anchoring at junction turns. When the gyro sees a completed turn, the position is snapped along the road to the matching road corner. | `junction.enabled` | `sih/round1/junction_anchor.py` |
| T9 | Fix for the double speed scale (see below). | `scale_fix.source` = `both` (baseline), `engine` or `ekf` | `sih/round1/engine_hooks.py` |

Support files:

- `sih/round1/config.py`: the flags.
- `sih/round1/history.py`: causal history before the blackout (t < blackout start, GNSS and IMU only, no CAN).
- `scripts/round1_apply_edits.py`: the marked edits to existing files. It is idempotent, preserves CRLF, and aborts if an anchor does not match.
- `scripts/round1_eval.py`: the experiment runner. It writes only to `results/round1/`.
- `tests/test_round1.py` and `tests/round1_synth.py`: 13 synthetic tests, no dataset needed.

## Edits to existing files (only these)

`sih/engine/dead_reckoning_engine.py` gets 8 marked `# [ROUND1]` insertions:

- the imports;
- `self.r1 = Round1EngineHooks.create(domain)`;
- one call at the end of `start_blackout`;
- 3 guarded calls in `step`;
- history injection in `run_scenario`;
- extra `r1_*` result keys, added only when a flag is on.

`MainActivity.kt` gets 3 marker-rotation lines.

Nothing else changes. That covers the benchmark script, the EKF, the matcher, README and reports.

## T9: Double-Scale Hypothesis (Disproved on Real Data)

The hypothesis was that blackout speed that reaches the EKF (`v_ai * engine.speed_scale * ekf._speed_scale`) applied the pre-blackout GNSS/AI speed ratio twice:
- `engine.speed_scale` is learned in `start_blackout`.
- `ekf._speed_scale` was suspected of learning a redundant scale in `update_gnss`.

**Empirical Result**: Disproved on real data.
The real-data autopsy (`scripts/round1_autopsy.py`) revealed median EKF internal speed scale 1.000 (autopsy) across all trips and scenarios because the EKF's pre-blackout velocity updates do not inflate scale. Disabling either scale factor produced zero statistical improvement, and T9 was rejected.

## Round 2: Speed-Scaling Optimization, Parity & Final Release

Round 2 optimized the pre-blackout observation windows and unified batch/live behavior:
- **Observation Window Sweep**: Tested 45s, 60s, 90s, 180s, blend60, blend90, and blend180 windows. The winner was `blend180`:
  `scale = 0.5 * (15 s entry GNSS/AI ratio) + 0.5 * (180 s GNSS-distance ratio)`
  clipped to [0.85, 1.25] (1.35 on Highway).
- **One Entry-Bearing Rule for Batch and Live**: `entry_doppler_bearing = false` unifies geometric bearing across both paths. The Doppler option was swept and lost (16 better vs 23 worse, p = 0.337).
- **Exact Streaming/Batch Parity**: 0.0000 m endpoint and trajectory diff across all 5 canonical scenarios in `scripts/quick_parity.py`.
- **Final Release**: Tag `round2-release` marks the frozen production state.

## Heading Seeding Diagnostic Status

`run_scenario` previously recorded `hdg_seed_err` from `ekf_pure._heading_rad` after the blackout loop, which measured drifted exit heading. In commit `51a5e89` (A1 diagnostic fix), `seeded_hdg` was relocated to `session.start_blackout`, recording true seeded heading error at blackout onset (**18.18° mean / 7.05° median** over 236 dev scenarios, `results/round1/hdg_seed/production_scenarios.csv`; 17.15° mean / 8.30° median on Seed 541098).

## Known limits & Operations

- **Pre-Blackout History**: T7 requires ~180 s of GNSS driving history. The live server path now incorporates this buffer in `server/engine_adapter.py`. If fewer than 180 s of driving are available before a blackout, T7 gracefully falls back to factor 1.0.
- **Rollback & Safety**: Production profile is configured in `config/round1/production.json`. The kill switch is `SIH_ROUND1_CONFIG=off`. Permanent rollback tags are `baseline-pre-round1`, `round1-release`, and `round2-release`.
- All synthetic gains are sanity checks only. Keep or drop each flag based on the real paired ablation (`vs_first`: better / worse / worst regression).

## Maths (plain text)

- T4: `s = 1 + (s_raw - 1) * n / (n + n0)`, with `s_raw = sum(d_gnss * d_gyro) / sum(d_gyro^2)`.
- T6: `alpha = clip(mean(v_gt[cal]) / max(0.5, mean(v_hat[cal])), 0.85, 1.25)`.
  `L_int = mean_h |sum(alpha * v_hat * dt) - sum(v * dt)| / max(sum(v * dt), 20 m)`.
  `L = L_phase55 + lambda * L_int`.
- T7: `f_band = (T_band * (r_band / r_all) + T0) / (T_band + T0)`, then `v = v_ai * speed_scale * f(v)`.
- T8: `p <- p + gain * ((C + (d_since_apex + R * (tan(theta/2) - theta/2)) * u_out - p) . u_out) * u_out`.
