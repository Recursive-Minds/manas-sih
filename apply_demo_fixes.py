"""
apply_demo_fixes.py  -  demo-path fixes for the on-device benchmark drawer.

Edits (anchored, all-or-nothing, idempotent) in BOTH copies of server/session_core.py:
  server/session_core.py
  android/app/src/ondevice/python/server/session_core.py
and in server/replay.py (+ its ondevice copy) and server/benchmark_setup.py (+ copy).

Usage (repo root):  python apply_demo_fixes.py --check   (dry run)
                    python apply_demo_fixes.py
"""
import argparse, sys
from pathlib import Path

SC = ["server/session_core.py", "android/app/src/ondevice/python/server/session_core.py"]
RP = ["server/replay.py", "android/app/src/ondevice/python/server/replay.py"]
BS = ["server/benchmark_setup.py", "android/app/src/ondevice/python/server/benchmark_setup.py"]
EA = ["server/engine_adapter.py", "android/app/src/ondevice/python/server/engine_adapter.py"]
EXP = "scripts/export_phone_benchmark_bundles.py"
KT_MAIN = "android/app/src/main/java/com/recursiveminds/idr/ui/MainActivity.kt"
KT_MODELS = "android/app/src/main/java/com/recursiveminds/idr/data/Models.kt"
KT_BRIDGE = "android/app/src/ondevice/java/com/recursiveminds/idr/engine/LocalChaquopyEngineBridge.kt"

HELPERS = '''
    # ------------------------------------------------------------------
    # [DEMOFIX] Handoff display wiring (DISPLAY ONLY - never feeds the engine)
    # Blackout is defined by the app state (START/STOP or the benchmark
    # batches), so the FSM is driven by set_blackout(), not by GNSS timeouts.
    # On exit: hold the DR marker until the first valid post-blackout fix,
    # then Hermite-blend the display from the last DR position to that fix.
    # ------------------------------------------------------------------
    def _handoff_ref(self):
        e = self.engine
        return (float(getattr(e, "ref_lat", self.ref_lat) or self.ref_lat),
                float(getattr(e, "ref_lon", self.ref_lon) or self.ref_lon),
                float(getattr(e, "ref_alt", self.ref_alt) or 0.0))

    def _to_enu(self, lat: float, lon: float) -> np.ndarray:
        from sih.data.geo import geodetic_to_enu
        rl, ro, ra = self._handoff_ref()
        return np.asarray(geodetic_to_enu(lat, lon, ra, rl, ro, ra), dtype=np.float64)

    def _handoff_on_blackout(self, active: bool, was_blackout: bool) -> None:
        hm = self.handoff_manager
        if hm is None:
            return
        from sih.handoff.manager import HandoffState
        if active:
            hm.reset()
            hm._state = HandoffState.INS_DEAD_RECKONING
            self._handoff_pending = False
            self._handoff_target_enu = None
        elif was_blackout:
            self._handoff_pending = True            # wait for first valid fix after exit
            hm._state = HandoffState.REACQUISITION_VERIFY

    def _handoff_on_gnss(self, gnss: GNSSSample) -> None:
        hm = self.handoff_manager
        if hm is None or not gnss.is_valid or self.state != "WARMING_UP":
            return
        from sih.handoff.manager import HandoffState
        if self._handoff_pending and self._last_dr_display_enu is not None:
            target = self._to_enu(gnss.latitude_deg, gnss.longitude_deg)
            hm.reconciler.initiate_blend(timestamp_ns=gnss.timestamp_ns,
                                         p_dead_reckoning=self._last_dr_display_enu.copy(),
                                         p_fused=target)
            self._handoff_target_enu = target
            self._handoff_pending = False
            hm._state = HandoffState.REACQUISITION_BLENDING if hm.reconciler.is_blending else HandoffState.GNSS_HEALTHY
        elif hm.reconciler.is_blending:
            self._handoff_target_enu = self._to_enu(gnss.latitude_deg, gnss.longitude_deg)
        elif hm._state in (HandoffState.INITIALIZING, HandoffState.GNSS_DEGRADED, HandoffState.REACQUISITION_VERIFY) and not self._handoff_pending:
            hm._state = HandoffState.GNSS_HEALTHY

    def _handoff_on_fused(self, fused: Any) -> None:
        self._last_dr_display_enu = self._to_enu(fused.latitude_deg, fused.longitude_deg)

    def _handoff_display(self) -> Optional[Dict[str, Any]]:
        hm = self.handoff_manager
        if hm is None or self._handoff_pending or self._handoff_target_enu is None or self._last_imu_ts is None:
            return None
        if not hm.reconciler.is_blending:
            return None
        from sih.handoff.manager import HandoffState
        from sih.data.geo import enu_to_geodetic
        p, alpha = hm.reconciler.get_blended_position(int(self._last_imu_ts), self._handoff_target_enu)
        if not hm.reconciler.is_blending:
            hm._state = HandoffState.GNSS_HEALTHY
            return None
        rl, ro, ra = self._handoff_ref()
        lat, lon, _ = enu_to_geodetic(p[0], p[1], p[2], rl, ro, ra)
        return {"lat": float(lat), "lon": float(lon), "alpha": float(alpha)}

    def _dr_visible(self) -> bool:
        hm = self.handoff_manager
        blending = bool(hm is not None and hm.reconciler.is_blending)
        return bool(self.state == "BLACKOUT" or self._handoff_pending or blending)

    def get_hud(self) -> Dict[str, Any]:'''

EDITS = []
for p in SC:
    EDITS += [
        (p, "S1 handoff config + fields",
         "                self.handoff_manager = SeamlessGNSSHandoffManager(HandoffConfig())\n",
         "                self.handoff_manager = SeamlessGNSSHandoffManager(HandoffConfig(blend_duration_s=3.0))  # [DEMOFIX] visible blend\n"),
        (p, "S2 new fields",
         '        self.map_matching_status_msg: str = "MAP MATCH: ON"\n',
         '        self.map_matching_status_msg: str = "MAP MATCH: ON"\n'
         '        # [DEMOFIX] handoff display + benchmark run state\n'
         '        self._handoff_pending: bool = False\n'
         '        self._handoff_target_enu = None\n'
         '        self._last_dr_display_enu = None\n'
         '        self._last_imu_ts = None\n'
         '        self.benchmark_running: bool = False\n'
         '        self._bundle_cache: Dict[str, Any] = {}\n'),
        (p, "S3 set_blackout hook",
         '        self.state = "BLACKOUT" if active else "WARMING_UP"\n        self.engine.set_blackout(active, entry_gnss=entry_gnss)\n',
         '        was_blackout = (self.state == "BLACKOUT")\n'
         '        self.state = "BLACKOUT" if active else "WARMING_UP"\n        self.engine.set_blackout(active, entry_gnss=entry_gnss)\n'
         '        self._handoff_on_blackout(active, was_blackout)  # [DEMOFIX]\n'),
        (p, "S4 reset clears handoff fields",
         "        self.engine.reset()\n        if self.handoff_manager is not None:\n            self.handoff_manager.reset()\n",
         "        self.engine.reset()\n        if self.handoff_manager is not None:\n            self.handoff_manager.reset()\n"
         "        self._handoff_pending = False  # [DEMOFIX]\n        self._handoff_target_enu = None\n"
         "        self._last_dr_display_enu = None\n        self._last_imu_ts = None\n"),
        (p, "S5 helpers before get_hud",
         "\n    def get_hud(self) -> Dict[str, Any]:", "\n" + HELPERS),
        (p, "S6 reconciled via reconciler",
         '''        reconciled_dict = None
        if self.handoff_manager is not None and getattr(self.handoff_manager.reconciler, "is_active", False):
            try:
                from sih.data.geo import enu_to_geodetic
                p_disp = self.handoff_manager.reconciler.get_reconciled_position(time.time_ns())
                if p_disp is not None and self.ref_lat != 0.0:
                    r_lat, r_lon, _ = enu_to_geodetic(
                        p_disp[0], p_disp[1], p_disp[2],
                        self.ref_lat, self.ref_lon, self.ref_alt
                    )
                    reconciled_dict = {"lat": float(r_lat), "lon": float(r_lon)}
            except Exception:
                pass
''',
         '''        reconciled_dict = self._handoff_display()  # [DEMOFIX] sensor-time Hermite blend
        if self.handoff_manager is not None:
            warmup["handoff_state"] = self.handoff_manager.state.value
        if self.benchmark_active:
            warmup["speed_calib_display"] = "Speed calibration: trip history (bench bundle)"
'''),
        (p, "S7 hud fields",
         '            "reconciled_pos": reconciled_dict,\n            "metrics": self.evaluator.get_summary_dict(),\n        }\n',
         '            "reconciled_pos": reconciled_dict,\n            "metrics": self.evaluator.get_summary_dict(),\n'
         '            "dr_visible": self._dr_visible(),  # [DEMOFIX]\n'
         '            "benchmark_running": bool(self.benchmark_running),  # [DEMOFIX]\n        }\n'),
        (p, "S8 gnss -> handoff",
         "            if self.handoff_manager is not None:\n                try:\n                    self.handoff_manager.on_gnss(gnss)\n                except Exception:\n                    pass\n",
         "            self._handoff_on_gnss(gnss)  # [DEMOFIX] display-only\n"),
        (p, "S9 fused -> handoff + imu ts",
         "            fused = self.engine.on_imu(imu)\n            if fused is not None:\n                self.evaluator.on_dr(fused)\n                if self.handoff_manager is not None:\n                    try:\n                        self.handoff_manager.on_fused_position(fused)\n                    except Exception:\n                        pass\n",
         "            self._last_imu_ts = imu.timestamp_ns  # [DEMOFIX]\n            fused = self.engine.on_imu(imu)\n            if fused is not None:\n                self.evaluator.on_dr(fused)\n                self._handoff_on_fused(fused)  # [DEMOFIX] display-only\n"),
        (p, "S10 bundle cache (parse once)",
         '''        if isinstance(bundle_source, str):
            if not os.path.exists(bundle_source):
                raise FileNotFoundError(f"Benchmark bundle not found: {bundle_source}")
            with gzip.open(bundle_source, "rt", encoding="utf-8") as f:
                bundle = json.load(f)
''',
         '''        if isinstance(bundle_source, str):
            if not os.path.exists(bundle_source):
                raise FileNotFoundError(f"Benchmark bundle not found: {bundle_source}")
            _key = f"{bundle_source}:{os.path.getsize(bundle_source)}"  # [DEMOFIX] parse once, re-setup fast
            if _key not in self._bundle_cache:
                with gzip.open(bundle_source, "rt", encoding="utf-8") as f:
                    self._bundle_cache = {_key: json.load(f)}
            bundle = self._bundle_cache[_key]
'''),
        (p, "S11 no history into evaluator",
         "            for gh in gnss_history:\n                self.evaluator.on_gnss(gh)\n",
         "            pass  # [DEMOFIX] history feeds the engine only; the evaluator (truth display) starts at the warm-up fix\n"),
        (p, "S12 fresh state on every setup",
         '        # Store batches in memory ready for replay\n        self.benchmark_batches = bundle.get("batches", [])\n',
         '        # Store batches in memory ready for replay\n        self.benchmark_batches = bundle.get("batches", [])\n'
         '        self.state = "WARMING_UP"  # [DEMOFIX] every setup starts a clean run\n'
         '        if self.handoff_manager is not None:\n            self.handoff_manager.reset()\n'
         '        self._handoff_pending = False\n        self._handoff_target_enu = None\n'
         '        self._last_dr_display_enu = None\n        self._last_imu_ts = None\n'
         '        self.benchmark_running = True\n'),
        (p, "S13 end of replay flag",
         '        if 0 <= index < len(self.benchmark_batches):\n            return self.push_batch(self.benchmark_batches[index], source="benchmark")\n',
         '        if 0 <= index < len(self.benchmark_batches):\n'
         '            if index >= len(self.benchmark_batches) - 1:\n                self.benchmark_running = False  # [DEMOFIX]\n'
         '            return self.push_batch(self.benchmark_batches[index], source="benchmark")\n'),
    ]
for p in RP:
    EDITS += [
        (p, "R1 tail_s param",
         "def slice_scenario(\n    trip: TripSequence, scenario_id: int, warmup_s: float = 60.0\n",
         "def slice_scenario(\n    trip: TripSequence, scenario_id: int, warmup_s: float = 60.0, tail_s: float = 5.0\n"),
        (p, "R2 tail_s used",
         "        slice_end_ns = bo_end_ns + int(5.0 * 1e9)\n",
         "        slice_end_ns = bo_end_ns + int(tail_s * 1e9)  # [DEMOFIX] default 5 s unchanged\n"),
    ]
for p in BS:
    EDITS += [
        (p, "B1 15 s tail for bundles (handoff visible)",
         "slice_scenario(trip, scenario_id, warmup_s=warmup_s)",
         "slice_scenario(trip, scenario_id, warmup_s=warmup_s, tail_s=15.0)"),
    ]


RESTORE = """    def setup_or_restore_benchmark(self, bundle_path: str) -> Dict[str, Any]:
        \"\"\"[DEMOFIX] First call per bundle: full setup, then snapshot engine+evaluator.
        Later calls (every RUN): restore the snapshot in < 1 s -> each run starts clean.\"\"\"
        import copy
        key = f"{bundle_path}:{os.path.getsize(bundle_path)}"
        snap = getattr(self, "_bench_snapshot", None)
        if snap is None or snap["key"] != key:
            res = self.setup_benchmark_from_bundle(bundle_path)
            memo = {id(self.predictor): self.predictor}
            if self.road_network is not None:
                memo[id(self.road_network)] = self.road_network
            attrs = {k: getattr(self, k, None) for k in self._BENCH_ATTRS}
            self._bench_snapshot = {"key": key, "state": copy.deepcopy((self.engine, self.evaluator), memo),
                                    "attrs": attrs, "res": res}
            return res
        memo = {id(self.predictor): self.predictor}
        if snap["attrs"].get("road_network") is not None:
            memo[id(snap["attrs"]["road_network"])] = snap["attrs"]["road_network"]
        self.engine, self.evaluator = copy.deepcopy(snap["state"], memo)
        for k, v in snap["attrs"].items():
            setattr(self, k, v)
        self.state = "WARMING_UP"
        if self.handoff_manager is not None:
            self.handoff_manager.reset()
        self._handoff_pending = False
        self._handoff_target_enu = None
        self._last_dr_display_enu = None
        self._last_imu_ts = None
        self.benchmark_running = True
        return snap["res"]

    _BENCH_ATTRS = ("ref_lat", "ref_lon", "ref_alt", "saved_alignment", "road_network", "domain",
                    "current_trip_name", "current_benchmark_scenario", "benchmark_active", "is_preloaded_trip",
                    "benchmark_bo_start_ns", "benchmark_bo_end_ns", "norm_mean", "norm_std",
                    "last_valid_gnss", "benchmark_batches", "expected_metrics")

    def get_benchmark_batch_count(self) -> int:"""

for p in SC:
    EDITS += [
        (p, "S14 fast prime from bundle speeds",
         "            self.engine.prime_features(preroll_imu, calib_samples=preroll_calib)\n",
         "            _rp = bundle.get(\"preroll_v_raw\")  # [DEMOFIX] exporter-computed speeds -> seconds, not minutes\n"
         "            self.engine.prime_features(preroll_imu, calib_samples=preroll_calib, raw_preds=_rp)\n"),
        (p, "S15 setup_or_restore_benchmark",
         "    def get_benchmark_batch_count(self) -> int:", RESTORE),
    ]
for p in EA:
    EDITS += [
        (p, "A1 prime_features params",
         "        imu_samples: List[IMUSample],\n        calib_samples: Optional[List[CalibratedSample]] = None,\n    ) -> None:\n        \"\"\"\n        Pre-rolls",
         "        imu_samples: List[IMUSample],\n        calib_samples: Optional[List[CalibratedSample]] = None,\n"
         "        raw_preds: Optional[List[float]] = None,\n        feature_tail: int = 300,\n    ) -> None:\n        \"\"\"\n        Pre-rolls"),
        (p, "A2 fast path",
         "        if self.model is None and self.predictor is None:\n            for c in c_list:\n",
         "        if raw_preds is not None and len(raw_preds) == N:\n"
         "            # [DEMOFIX] speeds precomputed by the exporter: skip N model calls; only the last\n"
         "            # feature_tail samples are needed to rebuild the extractor/IIR state (bit-exact in tests)\n"
         "            for c in c_list[max(0, N - int(feature_tail)):]:\n"
         "                f = self.feature_extractor.push(c)\n"
         "                self.feature_buf.append(f)\n"
         "                if len(self.feature_buf) > 60:\n"
         "                    self.feature_buf.pop(0)\n"
         "            for v_raw in raw_preds:\n"
         "                v_val = float(v_raw)\n"
         "                if self.use_speed_smoother and self.speed_smoother is not None:\n"
         "                    v_val = self.speed_smoother.update(v_val, dt_s=0.1)\n"
         "                self.recent_ai_speeds.append(v_val)\n"
         "            self._last_prime_raw_preds = [float(v) for v in raw_preds]\n"
         "            return\n\n"
         "        if self.model is None and self.predictor is None:\n            for c in c_list:\n"),
        (p, "A3 record speeds",
         "        for v_raw in preds:\n            v_val = float(v_raw)\n",
         "        self._last_prime_raw_preds = [float(v) for v in preds]  # [DEMOFIX] exported into bundles\n"
         "        for v_raw in preds:\n            v_val = float(v_raw)\n"),
    ]
EDITS += [
    (EXP, "X1 capture pre-roll speeds",
     "    adapter.prime_features(sess.preroll_imu, calib_samples=sess.preroll_calib)\n",
     "    adapter.prime_features(sess.preroll_imu, calib_samples=sess.preroll_calib)\n"
     "    preroll_raw = list(getattr(adapter, \"_last_prime_raw_preds\", []) or [])  # [DEMOFIX]\n"),
    (EXP, "X2 return speeds", "    return err, drift\n", "    return err, drift, preroll_raw\n"),
    (EXP, "X3 caller", "        err_m, drift_pct = evaluate_bundle_error(sess, rnet)\n",
     "        err_m, drift_pct, preroll_raw = evaluate_bundle_error(sess, rnet)\n"),
    (EXP, "X4 store in bundle", '            "batches": sess.batches,\n',
     '            "preroll_v_raw": preroll_raw,  # [DEMOFIX] phone skips the pre-roll model calls\n            "batches": sess.batches,\n'),
    (EXP, "X5 ondevice assets only",
     "        with open(out_main, \"wb\") as f:\n            f.write(gz_bytes)\n",
     "        # [DEMOFIX] bundles live only in the ondevice flavor (no duplicate copy in src/main/assets)\n"),
    (KT_MODELS, "M1 new HUD fields",
     '    @SerializedName("metrics") val metrics: LiveMetrics?\n)',
     '    @SerializedName("metrics") val metrics: LiveMetrics?,\n'
     '    @SerializedName("dr_visible") val drVisible: Boolean? = null,          // [DEMOFIX]\n'
     '    @SerializedName("benchmark_running") val benchmarkRunning: Boolean? = null  // [DEMOFIX]\n)'),
    (KT_MAIN, "K1 list only bundled scenarios on device",
     "        return list\n    }\n    private lateinit var scenarioAdapter",
     "        if (packageName.endsWith(\".ondevice\")) {\n"
     "            // [DEMOFIX] the phone can only replay scenarios that have a bench_<id>.bin bundle\n"
     "            val available = assets.list(\"\")?.toSet() ?: emptySet()\n"
     "            list.retainAll { it.isRandom || available.contains(\"bench_${it.id}.bin\") }\n"
     "        }\n"
     "        return list\n    }\n    private lateinit var scenarioAdapter"),
    (KT_MAIN, "K2 RUN disabled for the whole replay",
     "            if (isInBlackout || (latestMetrics != null && latestMetrics?.isBlackout == true)) {\n                tvBenchmarkBadge.text = \"REPLAYING #$scId\"",
     "            if (hud.benchmarkRunning == true || isInBlackout || (latestMetrics != null && latestMetrics?.isBlackout == true)) {  // [DEMOFIX]\n                tvBenchmarkBadge.text = \"REPLAYING #$scId\""),
    (KT_MAIN, "K3 hide DR marker after handoff (benchmark)",
     "            hud.drPos?.let { d ->\n                if (d.lat != 0.0 && d.lon != 0.0) {\n                    val pt = GeoPoint(d.lat, d.lon)\n                    tilePrefetcher?.onMotionUpdate(",
     "            if (hud.drVisible == false) {\n                drMarker?.isEnabled = false  // [DEMOFIX] handoff finished: hide DR marker, keep its trail\n            } else hud.drPos?.let { d ->\n                if (d.lat != 0.0 && d.lon != 0.0) {\n                    val pt = GeoPoint(d.lat, d.lon)\n                    tilePrefetcher?.onMotionUpdate("),
    (KT_MAIN, "K4 live: show handoff blend after STOP",
     "            } else {\n                drMarker?.isEnabled = false\n                reconciledMarker?.isEnabled = false\n            }\n        } else if (currentMode == AppMode.BENCHMARK_EVALUATION) {",
     "            } else {\n"
     "                // [DEMOFIX] after STOP keep the DR marker until the handoff blend ends; show the blend marker\n"
     "                if (hud.drVisible != true) drMarker?.isEnabled = false\n"
     "                val r = hud.reconciledPos\n"
     "                if (r != null && r.lat != 0.0 && r.lon != 0.0) {\n"
     "                    reconciledMarker?.position = GeoPoint(r.lat, r.lon)\n"
     "                    reconciledMarker?.isEnabled = true\n"
     "                } else {\n"
     "                    reconciledMarker?.isEnabled = false\n"
     "                }\n"
     "            }\n        } else if (currentMode == AppMode.BENCHMARK_EVALUATION) {"),
    (KT_MAIN, "K5 on-device wording",
     "Toast.makeText(this, \"Connect to IDR server first!\", Toast.LENGTH_SHORT).show()",
     "Toast.makeText(this, if (packageName.endsWith(\".ondevice\")) \"On-device engine is still starting - try again in a few seconds\" else \"Connect to IDR server first!\", Toast.LENGTH_SHORT).show()  // [DEMOFIX]"),
    (KT_BRIDGE, "B1 prepare uses setup_or_restore",
     "                val setupRes = sessionCore?.callAttr(\"setup_benchmark_from_bundle\", bundleFile.absolutePath)\n",
     "                val setupRes = sessionCore?.callAttr(\"setup_or_restore_benchmark\", bundleFile.absolutePath)  // [DEMOFIX]\n"),
    (KT_BRIDGE, "B2 every RUN starts clean",
     "                var totalBatches = executor.submit(Callable<Int> {\n                    sessionCore?.callAttr(\"get_benchmark_batch_count\")?.toInt() ?: 0\n                }).get()\n\n                if (totalBatches == 0) {\n                    Log.w",
     "                // [DEMOFIX] every RUN starts from a clean engine (snapshot restore, < 1 s; first time = full setup)\n"
     "                val runBundle = extractAssetFile(\"bench_${scenarioId}.bin\")\n"
     "                executor.submit(Callable<Unit> {\n"
     "                    sessionCore?.callAttr(\"setup_or_restore_benchmark\", runBundle.absolutePath)\n"
     "                }).get()\n"
     "                var totalBatches = executor.submit(Callable<Int> {\n                    sessionCore?.callAttr(\"get_benchmark_batch_count\")?.toInt() ?: 0\n                }).get()\n\n                if (totalBatches == 0) {\n                    Log.w"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    files, errs = {}, []
    for path, name, old, new in EDITS:
        if path not in files:
            fp = Path(path)
            if not fp.exists():
                errs.append(f"{name}: missing file {path}"); continue
            raw = fp.read_bytes().decode("utf-8")
            files[path] = {"crlf": "\r\n" in raw, "t": raw.replace("\r\n", "\n"), "n": 0}
        f = files.get(path)
        if f is None: continue
        if new in f["t"]:
            print(f"[skip] {path}: {name}"); continue
        c = f["t"].count(old)
        if c != 1:
            errs.append(f"{name}: anchor found {c}x in {path}"); continue
        f["t"] = f["t"].replace(old, new); f["n"] += 1; print(f"[ok]   {path}: {name}")
    if errs:
        print("\nABORTED - nothing written:"); [print("  -", e) for e in errs]; return 1
    if a.check:
        print("\n--check OK, nothing written"); return 0
    for path, f in files.items():
        if f["n"]:
            out = f["t"].replace("\n", "\r\n") if f["crlf"] else f["t"]
            Path(path).write_bytes(out.encode("utf-8")); print(f"wrote {path} ({f['n']} edits)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
