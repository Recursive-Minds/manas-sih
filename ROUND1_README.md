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

## T9: likely bug found while writing this (confirm with the autopsy)

The blackout speed that reaches the EKF is:

    v_ai * engine.speed_scale * ekf._speed_scale

Both factors come from the same thing: the pre-blackout ratio of GNSS speed to AI speed.

- `engine.speed_scale` is learned in `start_blackout`.
- `ekf._speed_scale` is learned in `update_gnss` during the 30 s warmup, from GNSS speed divided by the raw AI speed.

The EKF then applies it again in `predict()`: `v_fwd = vel.forward_speed_mps * self._speed_scale`.

So the correction is applied roughly squared. If the AI reads 10 % low, the effective correction becomes +21 % and the vehicle overshoots.

On the synthetic drive, `scale_fix.source = engine` cut the error from 22.1 m to 8.3 m. That is synthetic only; real-data numbers must come from `round1_eval.py`. The autopsy prints `scale eng x ekf = eff` for every scenario, so this can be confirmed directly.

## Also noticed

`run_scenario` reports `hdg_seed_err` from `ekf_pure._heading_rad` after the blackout loop. That makes it the final heading minus the entry heading, not the seeding error. It is a diagnostic only, and this pack does not change it.

## Known limits

- T4 and T7 need the pre-blackout history. The benchmark path provides it. The live server path does not yet, so there those two learners stay inactive (fall back to 1.0). T3, T5, T8 and T9 work in both paths.
- All synthetic gains are sanity checks only. Keep or drop each flag based on the real paired ablation (`vs_first`: better / worse / worst regression).

## Maths (plain text)

- T4: `s = 1 + (s_raw - 1) * n / (n + n0)`, with `s_raw = sum(d_gnss * d_gyro) / sum(d_gyro^2)`.
- T6: `alpha = clip(mean(v_gt[cal]) / max(0.5, mean(v_hat[cal])), 0.85, 1.25)`.
  `L_int = mean_h |sum(alpha * v_hat * dt) - sum(v * dt)| / max(sum(v * dt), 20 m)`.
  `L = L_phase55 + lambda * L_int`.
- T7: `f_band = (T_band * (r_band / r_all) + T0) / (T_band + T0)`, then `v = v_ai * speed_scale * f(v)`.
- T8: `p <- p + gain * ((C + (d_since_apex + R * (tan(theta/2) - theta/2)) * u_out - p) . u_out) * u_out`.
