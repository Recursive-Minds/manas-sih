"""
Applies the round-1 marked edits to existing files.

- Idempotent: an edit whose NEW text is already present is skipped.
- Safe: every OLD anchor must occur exactly once, otherwise NOTHING is written
  and the script exits 1 (report to the user, do not hand-edit around it).
- Preserves CRLF / LF line endings of each file.

Usage (repo root):
    python scripts/round1_apply_edits.py --check     # dry run
    python scripts/round1_apply_edits.py             # apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENGINE = "sih/engine/dead_reckoning_engine.py"
KOTLIN = "android/app/src/main/java/com/recursiveminds/idr/ui/MainActivity.kt"
INFERENCE = "sih/models/inference.py"
ADAPTER = "server/engine_adapter.py"

EDITS = [
    # ---------------- engine -----------------------------------------------------------
    (ENGINE, "E1 imports",
     "from sih.engine.speed_observer import KinematicSpeedObserver\n",
     "from sih.engine.speed_observer import KinematicSpeedObserver\n"
     "# [ROUND1] feature-flagged hooks (all OFF by default -> r1 is None -> baseline path)\n"
     "from sih.round1.engine_hooks import Round1EngineHooks\n"
     "from sih.round1.history import build_pre_blackout_history\n"),
    (ENGINE, "E2 hooks object",
     "        self.blackout_active: bool = False\n\n    def init_from_gnss",
     "        self.blackout_active: bool = False\n"
     "        # [ROUND1] None when every round-1 flag is OFF\n"
     "        self.r1: Optional[Round1EngineHooks] = Round1EngineHooks.create(domain)\n\n"
     "    def init_from_gnss"),
    (ENGINE, "E3 blackout start",
     "            current_yaw_rate_rad_s=turn_rate_entry,\n        )\n\n    def step(",
     "            current_yaw_rate_rad_s=turn_rate_entry,\n        )\n\n"
     "        # [ROUND1] entry-speed bookkeeping for T3/T5\n"
     "        if self.r1 is not None:\n"
     "            self.r1.on_blackout_start(self, recent_ai_speeds, t_entry_ns)\n\n"
     "    def step("),
    (ENGINE, "E4 speed path",
     "        v_ai_cal = float(v_pred) * self.speed_scale\n"
     "        v_pure_fwd, is_stat_pure = self.speed_obs_pure.update(cal, v_ai_cal)\n",
     "        # [ROUND1] T4 gyro scale (returns cal unchanged when OFF)\n"
     "        if self.r1 is not None:\n"
     "            cal = self.r1.pre_step_cal(cal)\n"
     "        v_ai_cal = float(v_pred) * self.speed_scale\n"
     "        # [ROUND1] T7 band factor + T5 speed mode\n"
     "        if self.r1 is not None:\n"
     "            v_ai_cal = self.r1.adjust_ai_speed(float(v_pred), v_ai_cal, t_curr)\n"
     "        v_pure_fwd, is_stat_pure = self.speed_obs_pure.update(cal, v_ai_cal)\n"
     "        # [ROUND1] T3 sticky stop detector\n"
     "        if self.r1 is not None:\n"
     "            v_pure_fwd, is_stat_pure = self.r1.post_observer(self.speed_obs_pure, v_pure_fwd, is_stat_pure, cal, v_ai_cal, t_curr)\n"),
    (ENGINE, "E5 junction anchor",
     "            matched_pos = self.matcher.match(fused_map, ekf=self.ekf_map, domain=self.domain, v_fwd=v_map_fwd)\n\n"
     "        return SteppableStepResult(",
     "            matched_pos = self.matcher.match(fused_map, ekf=self.ekf_map, domain=self.domain, v_fwd=v_map_fwd)\n\n"
     "        # [ROUND1] T8 junction along-track anchoring (map stream only)\n"
     "        if self.r1 is not None:\n"
     "            self.r1.post_step(self, cal, v_map_fwd, t_curr)\n\n"
     "        return SteppableStepResult("),
    (ENGINE, "E6 history",
     "        session.init_from_gnss(warmup_gnss)\n",
     "        session.init_from_gnss(warmup_gnss)\n\n"
     "        # [ROUND1] causal pre-blackout history (t < bo_start) for T3/T4/T7 learners\n"
     "        if session.r1 is not None and session.r1.needs_history:\n"
     "            session.r1.set_history(build_pre_blackout_history(\n"
     "                trip, calib_samples, v_preds, bo_start_ns, session.r1.cfg.history_s))\n"),
    (ENGINE, "E7 result var",
     "        return {\n            \"t_start_s\": (bo_start_ns - t0_ns) * 1e-9,",
     "        result = {\n            \"t_start_s\": (bo_start_ns - t0_ns) * 1e-9,"),
    (ENGINE, "E8 result diagnostics",
     "            \"entry_acq_info\": entry_acq_info,\n        }\n",
     "            \"entry_acq_info\": entry_acq_info,\n        }\n"
     "        # [ROUND1] diagnostics (keys only added when a round-1 flag is ON)\n"
     "        if session.r1 is not None:\n"
     "            result.update(session.r1.summary())\n"
     "        return result\n"),
    (ENGINE, "E9 raw speed scale (T10)",
     "                self.speed_scale = float(np.clip(scale, 0.85, 1.35 if self.domain == \"Highway\" else 1.25))\n",
     "                self.speed_scale_raw = float(scale)  # [ROUND1] T10: unclipped ratio, read by hooks only\n"
     "                self.speed_scale = float(np.clip(scale, 0.85, 1.35 if self.domain == \"Highway\" else 1.25))\n"),
    # ---------------- promotion support (behaviour-neutral until config/round1/production.json exists)
    (INFERENCE, "M1 model selection",
     "    default_tcn_path = os.path.join(root_dir, \"models\", \"checkpoints\", \"best_velocity_model.pt\")\n",
     "    default_tcn_path = os.path.join(root_dir, \"models\", \"checkpoints\", \"best_velocity_model.pt\")\n\n"
     "    # [ROUND1] production profile may choose the speed checkpoint; \"a.pt,b.pt\" = mean ensemble\n"
     "    if model_path is None:\n"
     "        from sih.round1.model_select import resolve_velocity_checkpoint\n"
     "        model_path = resolve_velocity_checkpoint(root_dir)\n"
     "    if model_path and \",\" in model_path:\n"
     "        from sih.round1.model_select import load_mean_ensemble\n"
     "        return load_mean_ensemble(model_path, device, root_dir)\n"),
    (ADAPTER, "A1 live pre-blackout history",
     "        self.session.start_blackout(\n            entry_pos_enu=entry_pos_enu,\n",
     "        # [ROUND1] live pre-blackout history (same data the benchmark uses) for T7 / T10 learners\n"
     "        if self.session.r1 is not None and self.session.r1.needs_history:\n"
     "            from sih.round1.history import build_history_from_buffers\n"
     "            self.session.r1.set_history(build_history_from_buffers(\n"
     "                self.recent_imu_calib, self.recent_ai_speeds, valid_hist_gnss,\n"
     "                self.ref_lat, self.ref_lon, t_entry_ns, self.session.r1.cfg.history_s))\n\n"
     "        self.session.start_blackout(\n            entry_pos_enu=entry_pos_enu,\n"),
    # ---------------- R2: one entry-bearing rule for batch and live
    (ENGINE, "E10 entry bearing (batch)",
     "        pre_gnss_window = SteppableDeadReckoningEngine.synthesize_1hz_gnss_window(\n"
     "            valid_hist_gnss, bo_start_ns, trip.reference_lat_deg, trip.reference_lon_deg\n"
     "        )\n",
     "        pre_gnss_window = SteppableDeadReckoningEngine.synthesize_1hz_gnss_window(\n"
     "            valid_hist_gnss, bo_start_ns, trip.reference_lat_deg, trip.reference_lon_deg\n"
     "        )\n"
     "        # [ROUND1] R2: same entry-bearing rule as the live adapter (flag entry_doppler_bearing)\n"
     "        if session.r1 is not None and session.r1.cfg.entry_doppler_bearing:\n"
     "            from sih.round1.entry_bearing import apply_entry_doppler\n"
     "            pre_gnss_window = apply_entry_doppler(pre_gnss_window, g_entry)\n"),
    (ADAPTER, "A2 entry bearing (live)",
     "        if g_ref is not None and g_ref.bearing_deg is not None and pre_gnss_window:\n",
     "        from sih.round1.entry_bearing import live_override_enabled  # [ROUND1] R2\n"
     "        if g_ref is not None and g_ref.bearing_deg is not None and pre_gnss_window and live_override_enabled():\n"),
    # ---------------- T1 pointer (Kotlin) ------------------------------------------------
    (KOTLIN, "K1 live GNSS marker",
     "                                vehicleMarker?.rotation = gnss.bearingDeg\n",
     "                                vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(gnss.bearingDeg)  // [ROUND1] T1\n"),
    (KOTLIN, "K2 replay GNSS marker",
     "                        vehicleMarker?.rotation = g.bearingDeg?.toFloat() ?: 0f\n",
     "                        vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(g.bearingDeg)  // [ROUND1] T1\n"),
    (KOTLIN, "K3 replay DR marker",
     "                    vehicleMarker?.rotation = d.headingDeg?.toFloat() ?: 0f\n",
     "                    vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(d.headingDeg)  // [ROUND1] T1\n"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry run")
    ap.add_argument("--only", choices=["engine", "kotlin"], default=None)  # kept for compatibility
    args = ap.parse_args()

    root = Path(".").resolve()
    files = {}
    errors = []
    for path, name, old, new in EDITS:
        if args.only == "engine" and path != ENGINE or args.only == "kotlin" and path != KOTLIN:
            continue
        if path not in files:
            raw = (root / path).read_bytes().decode("utf-8")
            files[path] = {"crlf": "\r\n" in raw, "text": raw.replace("\r\n", "\n"), "applied": []}
        f = files[path]
        if new in f["text"]:
            print(f"[skip] {name}: already applied")
            continue
        n = f["text"].count(old)
        if n != 1:
            errors.append(f"{name}: anchor found {n} times in {path}")
            continue
        f["text"] = f["text"].replace(old, new)
        f["applied"].append(name)
        print(f"[ok]   {name}")

    if errors:
        print("\nABORTED - nothing written:")
        for e in errors:
            print("  -", e)
        return 1
    if args.check:
        print("\n--check: all anchors valid, nothing written")
        return 0
    for path, f in files.items():
        if not f["applied"]:
            continue
        out = f["text"].replace("\n", "\r\n") if f["crlf"] else f["text"]
        (root / path).write_bytes(out.encode("utf-8"))
        print(f"wrote {path} ({len(f['applied'])} edits, {'CRLF' if f['crlf'] else 'LF'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
