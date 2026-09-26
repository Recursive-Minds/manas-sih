# Smartphone Intelligent Dead Reckoning (IDR) — Code Reality Audit Report

**Branch:** `audit/code-reality`  
**Evaluation Standard:** Active Codebase as Sole Source of Truth (Zero Trust in Docstrings, Comments, or Readmes)  
**Date:** September 25, 2026  

---

## Executive Summary (10-Line Code Reality)

1. **Android App Execution**: The Android app (`SensorStreamService.kt`, `MainActivity.kt`) runs solely as a thin sensor telemetry collector and display HUD; it streams raw accelerometer, gyroscope, and GNSS samples over WebSocket (`ws://<ip>:8765/ws/stream`) and renders osmdroid map overlays.
2. **Laptop Server Execution**: The Python backend (`server/router.py`, `server/engine_adapter.py`) executes the entire navigation pipeline: mount calibration, 12-channel causal feature extraction, PyTorch neural MoE forward inference, causal speed smoothing, complementary speed filtering, 15-state ES-EKF, and topological HMM map matching.
3. **Edge / On-Device Reality**: Zero ML inference, dead reckoning, or filtering runs on the Android phone today; no ONNX Runtime, TFLite interpreter, C++ NDK wrapper, or PyTorch Mobile runtime is packaged or called in Kotlin.
4. **Warmup & START Gating**: In Kotlin (`MainActivity.kt:676-693`), the START button is unconditionally enabled whenever `!isInBlackout`; users can initiate dead reckoning before mount yaw lock (0/8 turns), before gravity leveling (3s), and before feature buffer warm-up (6s), with only an advisory Toast.
5. **Critical Gap 1 (Heading Seeding Metric)**: `hdg_seed_err` in `dead_reckoning_engine.py:595` is evaluated *after* the entire 30–75s blackout loop finishes, comparing `ekf_pure`'s final integrated heading at blackout *exit* against `g_entry.bearing_deg` at blackout *entry*; on turning maneuvers, it measures total trajectory turn angle plus drift, not entry seeding error.
6. **Critical Gap 2 (Phase 6 Handoff)**: The 6-state handoff FSM (`SeamlessGNSSHandoffManager`) and C^2 Hermite smoothstep reconciliation (`HermiteReconciler`) are implemented and unit-tested in `sih/handoff/`, but are completely uncalled and disconnected from both `server/engine_adapter.py` and `dead_reckoning_engine.py`.
7. **Critical Gap 3 (Missing / Unused Physical Constraints)**: The documented low-speed crawl clamp `v_fwd <= max(v_entry + 1.2, 3.5)` is completely absent from the codebase; the highway straight-line lock (`update_straight_line_lock`) is implemented in `es_ekf.py:731` but never invoked in the runtime stepping loop.
8. **Critical Gap 4 (Corridor Prefetching)**: The speed-adaptive predictive lookahead corridor manager (`PredictiveCorridorManager`) is tested in unit tests but unused by `server/router.py`, which loads static trip GeoJSON networks directly from disk cache.
9. **Critical Gap 5 (Hardcoded Claims & Ghost Features)**: "Sub-lane accuracy", "Simultaneous sub-10% performance", and "Initial Heading Seeding Error PASSED" are hardcoded strings in `benchmarks/run_final_benchmark.py`; claimed hardware features (barometer flyover gating, motorcycle lean-angle frame, NDK `ASensorManager`, INT8 quantization) do not exist in the repository.
10. **Verified Production Core**: The production core pipeline—gravity leveling, centripetal yaw selection, dual-brain Bayesian MoE speed inference (`round1_interval_lam0.5_s42.pt`), rate-adaptive NHC, Lorentzian gyro damping, topological HMM map matching, and Round 1/2 hooks (T7 online calibration, T8 junction snapping, blended speed scale)—is fully implemented and achieves bit-identical 0.0000 m batch vs. streaming parity.

---

## Part 1: How the System Really Runs, End-to-End

### 1. Benchmark Execution Path
The canonical benchmark and held-out evaluation execute through the following chain:

1. **Entry Point (`scripts/evaluate_heldout_seeds.py:25`)**:
   `run_heldout_evaluation()` initializes compute device (`torch.device("cuda" if ... else "cpu")`) and calls `load_precomputed_benchmark_data(device, model_path=None, map_source="osm")` (`evaluate_heldout_seeds.py:36`).
2. **Precomputation Loader (`benchmarks/run_final_benchmark.py:123`)**:
   `load_precomputed_benchmark_data` executes:
   - `sih/models/inference.py:24` (`load_ai_model`): Resolves checkpoint via `sih/round1/model_select.py:24` (`resolve_velocity_checkpoint`) reading `config/round1/production.json` (`"velocity_checkpoint": "models/checkpoints/round1_interval_lam0.5_s42.pt"`).
   - `sih/data/loader.py:120` (`GenericDataLoader.load_file`): Loads trip CSVs (`S-M`, `S-S2`, `S-S1`, `S-S3a`, `S-S4`).
   - `sih/calibration/mount.py:214` (`calibrate_stream`): Runs `MountCalibrator` over the full trip stream.
   - `sih/map/network.py:863` (`load_trip_road_network`): Loads GeoJSON road network from disk cache `data/maps/cache`.
   - `sih/models/inference.py:92` (`predict_velocities`): Extracts 12-channel features via `StreamingFeatureExtractor.batch_extract` (`sih/features/streaming.py:147`) and runs batch forward inference across MoE models, yielding continuous array `v_preds`.
3. **Scenario Iteration (`benchmarks/run_final_benchmark.py:201`)**:
   `evaluate_seed_scenarios(seed, pre)` draws 40 blackout intervals across trips with durations `[30.0, 45.0, 60.0, 75.0]`, calling `run_scenario(...)` at `benchmarks/run_final_benchmark.py:307`.
4. **Scenario Execution Wrapper (`sih/engine/dead_reckoning_engine.py:808`)**:
   `run_dead_reckoning_scenario` instantiates `DeadReckoningEngine(enable_route_matching=False, enable_speed_scale=True)` and invokes `engine.run_scenario(...)` (`dead_reckoning_engine.py:827`).
5. **Steppable Engine Execution (`sih/engine/dead_reckoning_engine.py:429`)**:
   `DeadReckoningEngine.run_scenario` executes:
   - `session = SteppableDeadReckoningEngine(...)` (`dead_reckoning_engine.py:486`)
   - `session.init_from_gnss(warmup_gnss)` (`dead_reckoning_engine.py:498`)
   - Pre-blackout history initialization: `session.r1.set_history(build_pre_blackout_history(...))` (`dead_reckoning_engine.py:502`)
   - GNSS 1 Hz window synthesis: `SteppableDeadReckoningEngine.synthesize_1hz_gnss_window(...)` (`dead_reckoning_engine.py:522`)
   - Warmup replay loop: `session.update_gnss_warmup(g)` (`line 540`) and `session.predict_warmup(cal, v_pred, t)` (`line 557`)
   - Blackout initialization: `session.start_blackout(...)` (`line 547`):
     - Seeds heading: `ekf_pure.seed_pre_blackout_heading(...)` (`line 268`) and `ekf_map.seed_pre_blackout_heading(...)` (`line 274`)
     - Registers entry speeds: `session.r1.on_blackout_start(...)` (`line 283`)
   - Blackout stepping loop (`dead_reckoning_engine.py:559`):
     Calls `step_res = session.step(cal, float(v_preds[j]), t_curr)` (`dead_reckoning_engine.py:285`):
     - Pre-step gyro scale: `self.r1.pre_step_cal(cal)` (`line 288`)
     - Baseline speed scale: `v_ai_cal = float(v_pred) * self.speed_scale` (`line 289`)
     - T7 band speed adjustment: `self.r1.adjust_ai_speed(...)` (`line 292`)
     - Complementary speed observer: `self.speed_obs_pure.update(cal, v_ai_cal)` (`sih/engine/speed_observer.py:61`) (`line 293`)
     - Road kinematics governor: `self.governor.govern_speed(...)` (`sih/map/governor.py:126`) (`line 312`)
     - EKF propagation: `self.ekf_pure.predict(cal, vel_pure)` (`sih/fusion/es_ekf.py:382`) (`line 329`) and `self.ekf_map.predict(cal, vel_map)` (`line 330`)
     - Map matching: `self.matcher.match(fused_map, ekf=self.ekf_map, ...)` (`sih/map/matcher.py:209`) (`line 334`)
     - T8 junction snapping: `self.r1.post_step(self, cal, v_map_fwd, t_curr)` (`line 338`)

---

### 2. Live App Execution Path (Current State)

```
[Android Phone]
  SensorStreamService.kt (SensorManager @ 10-50Hz, LocationListener @ 1Hz)
         │
         ▼  (100ms batched JSON packets)
  WebSocket: ws://<server_ip>:8765/ws/stream
         │
         ▼
[Laptop Python Server]
  server/router.py (ws_stream_handler)
         │
         ▼
  server/engine_adapter.py (EngineAdapterStageB)
         ├── calibrator.update(imu)               [sih/calibration/mount.py]
         ├── _infer_speed()                       [feature_extractor -> MoE model -> speed_smoother]
         └── session.step()                       [SteppableDeadReckoningEngine]
                 ├── speed_obs_pure.update()      [sih/engine/speed_observer.py]
                 ├── governor.govern_speed()      [sih/map/governor.py]
                 ├── ekf_pure / ekf_map.predict() [sih/fusion/es_ekf.py]
                 ├── matcher.match()              [sih/map/matcher.py]
                 └── r1.post_step() (T8 snap)     [sih/round1/junction_anchor.py]
         │
         ▼  (HudUpdate JSON packet)
  WebSocket: ws://<server_ip>:8765/ws/client
         │
         ▼
[Android Phone]
  MainActivity.kt (updateHudUi -> updates osmdroid vehicle marker position & rotation)
```

- **EngineAdapter Selection (`server/router.py:126-146`)**:
  `engine_type` defaults to `"stage_b"` (`router.py:105`). At runtime, `EngineAdapterStageB` is ALWAYS instantiated (`router.py:134, 349, 486`). `EngineAdapterStageA` is ONLY initialized if `engine_type == "stage_a"`, which is never triggered by `router.py`'s CLI (`router.py:725`) and exists only in unit test mocks (`tests/test_app_step2.py`).
- **What runs on the phone**:
  `android/app/src/main/java/com/recursiveminds/idr/`:
  - `service/SensorStreamService.kt`: Listens to Android `SensorManager` and `LocationManager`, batches samples every 100ms, and sends them via OkHttp WebSocket.
  - `ui/MainActivity.kt`: User interface, osmdroid MapView, control buttons, benchmark drawer, and HUD rendering.
  - `ui/MarkerHeading.kt`: Converts compass bearing to counter-clockwise osmdroid marker rotation.
  - `map/SpeedAdaptiveTilePrefetcher.kt`: Prefetches OSM raster image tiles based on speed.
  - `data/Models.kt`: Gson data classes for JSON messages.
- **What runs on the laptop server**:
  All sensor calibration, AI feature extraction, PyTorch model forward inference, causal kinematic speed smoothing, complementary velocity filtering, 15-state EKF propagation, and topological HMM map matching.
- **Does anything run fully on the phone today?**:
  **NO.** Zero dead-reckoning logic, filtering, or neural network inference runs on the phone.

---

### 3. Benchmark Replay Path in App

1. **User Selection**: User opens the "BENCHMARK SUITE" drawer in `MainActivity.kt:274`. `MainActivity.kt:284` triggers `streamService?.prepareBenchmark(scId)`.
2. **Control Message**: `MainActivity.kt:329` (`btnRunBenchmark.setOnClickListener`) calls `s.startBenchmark(scenarioId, speed)`.
3. **WebSocket Transmission**: `SensorStreamService.kt:309-317` sends `ControlMessage(command = "start_benchmark", scenarioId = scenarioId, speed = speed)`.
4. **Server Routing**: `server/router.py:626-629` receives the message in `ws_stream_handler` and calls `await router.start_benchmark(scenario_id=sc_id, speed=spd)` (`router.py:375`).
5. **Replay Engine Execution (`server/router.py:389-438`)**:
   - `server/replay.py:slice_scenario`: Extracts the specific scenario slice and 30s pre-blackout warmup from `self.loaded_trip`.
   - `MountCalibrator`: Runs calibration strictly up to scenario entry (`router.py:396-406`).
   - `self.engine.reset()` (`router.py:410`): Injects aligned mount matrix, resets EKF, and seeds initial fix.
   - `build_sensor_batches` (`router.py:425`): Slices trip into 100 ms telemetry batches.
   - `_run_benchmark_loop` (`router.py:447-474`): Asynchronous loop piping batches through `self.process_batch(batch)` with paced sleep `max(0, 0.10 / speed - elapsed)`. Engine processes batches identically to live driving.

---

### 4. Standalone Logging Path (START REC / STOP / SHARE)

1. **START REC (`MainActivity.kt:253-255`)**:
   Invokes `streamService?.startCsvRecording()`. In `SensorStreamService.kt:376-394`:
   - Checks `isCsvRecording.get()`.
   - Creates file `idr_telemetry_<timestamp>.csv` in `getExternalFilesDir(null)` (or `filesDir`).
   - Writes CSV header: `"type,timestamp_ns,val1,val2,val3,val4,val5,val6\n"` (`SensorStreamService.kt:384`).
   - Sets `isCsvRecording.set(true)`.
   - Appends IMU rows (`logImuToCsv`, line 420): `IMU,<ts>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>`.
   - Appends GNSS rows (`logGnssToCsv`, line 427): `GNSS,<ts>,<lat>,<lon>,<alt>,<spd>,<brg>,<acc>`.
2. **STOP (`MainActivity.kt:257-259`)**:
   Invokes `streamService?.stopCsvRecording()`. In `SensorStreamService.kt:396-410`:
   - Flushes and closes `FileWriter`.
   - Sets `isCsvRecording.set(false)`.
   - Emits callback `onCsvStateChanged(false, file.absolutePath)`.
3. **SHARE (`MainActivity.kt:261-263`)**:
   Invokes `shareCsvFile()` (`MainActivity.kt:963-985`):
   - Flushes active CSV writer.
   - Generates content URI via `FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)`.
   - Launches Android `Intent.ACTION_SEND` with MIME type `"text/csv"` via `Intent.createChooser(...)`.

---

## Part 2: Warm-up and START Gating

### 1. START Button Enablement Conditions (Kotlin)
- **Source Code (`MainActivity.kt:676-693`)**:
  ```kotlin
  private fun updateControlButtons() {
      if (isInBlackout) {
          btnStart.isEnabled = false
          btnStart.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
          btnStop.isEnabled = true
          btnStop.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.accent_rose))
      } else {
          btnStop.isEnabled = false
          btnStop.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
          btnStart.isEnabled = true
          val startColor = if (isEngineReady) {
              ContextCompat.getColor(this, R.color.accent_emerald)
          } else {
              ContextCompat.getColor(this, R.color.accent_amber)
          }
          btnStart.backgroundTintList = ColorStateList.valueOf(startColor)
      }
  }
  ```
- **Button Click Action (`MainActivity.kt:391-403`)**:
  ```kotlin
  btnStart.setOnClickListener {
      val s = streamService ?: return@setOnClickListener
      if (!isEngineReady) {
          Toast.makeText(this, "Starting Dead Reckoning blackout (Warmup still completing...)", Toast.LENGTH_SHORT).show()
      }
      s.startBlackout()
      isInBlackout = true
      ...
  ```
- **Audited Answers**:
  - **Can START be pressed before mount yaw lock (0/8 turns)?**: **YES.** `btnStart.isEnabled = true` is unconditionally active whenever not in blackout.
  - **Can START be pressed before gravity leveling?**: **YES.**
  - **Can START be pressed before the 6s feature buffer is warm?**: **YES.**
  - If `isEngineReady == false`, the button is colored amber instead of emerald and displays a transient Toast message, but execution proceeds immediately without blocking.

---

### 2. Server Behavior If Blackout Starts Before Mount Is Locked
- **Mount Alignment Used (`sih/calibration/mount.py:141-180`)**:
  1. If `len(_accel_buf) >= 30` (>= 3.0s of data): Gravity vector is leveled via Rodrigues rotation (`R_level`). Because `< 8` turn events exist, it executes the variance fallback heuristic at `mount.py:141-147`:
     ```python
     gyros_arr = np.array(self._gyro_buf[:min(len(self._gyro_buf), 200)])
     stds = np.std(gyros_arr, axis=0)
     yaw_idx = int(np.argmax(stds))
     yaw_sign = 1.0
     ```
     Yields a leveled alignment with heuristic yaw axis, assuming positive sign (`+1.0`).
  2. If `len(_accel_buf) < 30` (< 3.0s of data) and no saved alignment exists: `self._alignment` is `None`. `mount.py:172-180` returns an uncalibrated sample with **identity rotation matrix (`np.eye(3)`)** and untouched body accelerometer/gyroscope readings.
  3. If a valid `saved_alignment` was passed to the router (e.g. CLI `--trip`), it reuses the saved alignment.
- **Is `is_calibrated` checked anywhere during blackout?**:
  **NO.** Neither `SteppableDeadReckoningEngine.step` (`sih/engine/dead_reckoning_engine.py:285`) nor `EngineAdapterStageB.on_imu` (`server/engine_adapter.py:829`) checks `cal.is_calibrated` or `calibrator.is_calibrated` during blackout stepping.

---

### 3. "Mount Not Locked / Accuracy Degraded" Warning
- **Audited Finding**: **NOT FOUND IN CODE.**
- Neither `MainActivity.kt` nor `strings.xml` nor `server/router.py` contains the string `"accuracy degraded"` or any modal warning blocking uncalibrated blackout starts. The UI only updates a text label (`tvMountStatusDetail.text`) with `hud.mountStatus` (`"Mount: calibrating X/8"`, `"Mount: locked"`, `"Mount: reused"`).

---

### 4. Mount Turn Event Counting and Lock Thresholds (`sih/calibration/mount.py`)
- **Turn Detection Criteria (`mount.py:73-91`)**:
  - Speed threshold: `gnss.speed_mps >= self.min_speed_mps` (default `2.0 m/s` = 7.2 km/h).
  - Time delta between fixes: `0.2s <= dt_g <= 15.0s` (`line 80`).
  - Heading turn angle: `abs((gnss.bearing_deg - prev_g.bearing_deg + 180.0) % 360.0 - 180.0) >= 2.5 deg` (`line 82`).
  - Gyro integration: Trapezoidal integration of gyro body axes across interval `[t1, t2]`:
    `d_th = np.sum(0.5 * (sub_gy[:-1] + sub_gy[1:]) * dt_imu[:, None], axis=0)` (`line 89`).
  - Appends tuple `(d_bearing_rad, d_th_x, d_th_y, d_th_z)` to `self._turn_events`.
- **Lock Thresholds (`mount.py:120-140`)**:
  - Evaluation begins at: `len(self._turn_events) >= 8`.
  - Computes correlation `corrs = corrcoef(evs[:, 0], evs[:, a + 1])` and energy `energies = mean(|evs[:, a + 1]|)`.
  - Computes score `scores = |corrs| * (energies + 1e-6)`.
  - Determines yaw sign: `slope = np.polyfit(evs[:, best_a + 1], evs[:, 0], 1)[0]`, `yaw_sign = -1.0 if slope > 0 else 1.0` (`line 132`).
  - **Hard Lock Condition (`line 139`)**:
    `is_locked = True` strictly requires:
    1. `len(self._turn_events) >= 15` (NOT 8; 8 only evaluates candidate without lock)
    2. `abs(valid_c[best_a]) >= 0.35`
    3. `separation >= 1.5` (where `separation = sorted_scores[-1] / sorted_scores[-2]`)

---

### 5. Saved Mount Alignment Reuse & Invalidation (Mount Guard)
- **Reuse (`server/engine_adapter.py:522-525`)**:
  If `saved_alignment` is passed and `saved_alignment.is_calibrated` is true, the engine initializes `self.calibrator._alignment = self.saved_alignment`, `self.calibrator._yaw_locked = True`, and `self.mount_reused = True`.
- **Invalidation via Mount Guard (`server/engine_adapter.py:591-615`)**:
  Triggered once on the 30th accelerometer sample (`len(self.initial_accels) == 30` = 3.0s):
  - Computes mean gravity vector `g_curr = np.mean(self.initial_accels[:30], axis=0)`.
  - Calculates angular discrepancy: `angle_deg = np.degrees(arccos(clip(dot(g_curr_norm, g_saved_norm), -1.0, 1.0)))`.
  - If `angle_deg < 5.0 deg`: Retains saved alignment (`mount_reused = True`).
  - If `angle_deg >= 5.0 deg`: Invalidates saved alignment (`self.saved_alignment = None`), calls `self.calibrator.reset()`, and sets `self.mount_changed = True` (`lines 611-614`).
  - (Skipped if `self.lock_saved_alignment == True`, line 592).

---

### 6. T7 Behavior If Blackout Starts < 180s After App Launch
- **History Extraction (`sih/round1/history.py:81-108`)**:
  `build_history_from_buffers` extracts trailing samples within `[bo_start_ns - history_s, bo_start_ns]`. If app has run for only 30s, history contains only 30s of data.
- **Fitting Failure & Fallback (`sih/round1/online_speed_calib.py:47-76`)**:
  - If `len(ts) < 3` or `hist.n_imu < 50`: `fit()` exits immediately with `info = {"reason": "not enough history"}` (`line 48`).
  - If fewer than 3 valid 10-second windows exist (`len(d_g) < 3`): `fit()` exits with `info = {"reason": "too few windows"}` (`line 75`).
  - In either case, `self.fitted` remains `False`.
- **Speed Factor Code Verification**:
  - `BandSpeedCalibrator.factor(v)` (`line 95`):
    ```python
    if not self.fitted:
        return 1.0
    ```
    Returns exact `1.0`.
  - `engine_hooks.py:73-74`: `self.history_ratio` remains `None`.
  - `engine_hooks.py:88-89`: In `scale_level` with `source = "blend"`, if `self.history_ratio is None`, `raw` falls back to `self.scale_raw` (the entry 15s GNSS/AI ratio). If `self.scale_raw` is also None, `engine.speed_scale` is not overwritten (defaults to 1.0).

---

### 7. Cold-Start / No-Yaw-Lock Measurement Records
- **File Location**: `APP_REPORT.md:95-106` (generated by script `server/warmup_benchmark.py`).
- **Real Empirical Measurements**:
  - Evaluated on Seed 541098 across all 40 canonical scenarios:
    - **Full History (Canonical baseline)**: Yaw-Locked: 40/40 (100%), Median Drift: **14.31%**
    - **30.0 s Cold Start**: Yaw-Locked: **0/40 (0.0%)**, Median Drift: **22.77%** (P90: 139.49%, Tier 1: 12/40)
    - **10.0 s Cold Start**: Yaw-Locked: **0/40 (0.0%)**, Median Drift: **29.36%** (P90: 161.96%, Tier 1: 9/40)
    - **5.0 s Cold Start**: Yaw-Locked: **0/40 (0.0%)**, Median Drift: **22.01%** (P90: 102.98%, Tier 1: 12/40)
- **Standalone Artifact Files**: **NOT FOUND IN CODE** (no standalone `.json` or `.csv` exists in `results/`, `artifacts/`, or `logs/`; values were logged in `APP_REPORT.md`).

---

## Part 3: Feature Reality Table

| Feature Category | Specific Item | Status | Evidence (`file:line`) | Actual Parameter Values in Code |
| :--- | :--- | :--- | :--- | :--- |
| **Mount & Features** | Gravity Leveling | IMPLEMENTED+USED | `sih/calibration/mount.py:97-114` | `min_samples = 30`, Rodrigues rotation `R.from_rotvec(axis * angle)` aligning mean accel with `[0, 0, 1]`. |
| | Centripetal Yaw Selection | IMPLEMENTED+USED | `sih/calibration/mount.py:120-126` | Correlation scores `|corr| * (energy + 1e-6)` across 3 axes. Requires `>= 8` turn events. |
| | Sign Lock | IMPLEMENTED+USED | `sih/calibration/mount.py:131-139` | `slope = polyfit(evs[:, best+1], evs[:, 0], 1)[0]`; sign is `-1.0` if slope > 0 else `+1.0`. Lock requires `>= 15` events, `|corr| >= 0.35`, `separation >= 1.5`. |
| | `StreamingFeatureExtractor` | IMPLEMENTED+USED | `sih/features/streaming.py:31-180` | 12 channels: `[f_ax, f_ay, f_az, f_gx, f_gy, f_gz, \|a\|, \|w\|, E_A, E_B, E_ratio, v_proxy]`. 2nd-order Butterworth low-pass filter at `cutoff_hz = 3.5 Hz`. Band A = `(0.1, 1.5) Hz`, Band B = `(1.5, 4.5) Hz`. Max jerk = `15.0 m/s^3`, window = 60 ticks (6.0s), spectral stride = 5 ticks (0.5s). |
| **Speed** | Default Checkpoint Loaded | IMPLEMENTED+USED | `sih/models/inference.py:34`, `sih/round1/model_select.py:24`, `config/round1/production.json:1` | Checkpoint: `models/checkpoints/round1_interval_lam0.5_s42.pt` (promoted s42 interval-tuned MoE). Fallback if unconfigured: `best_moe_velocity_model.pt`. |
| | `CausalSpeedSmoother` | IMPLEMENTED+USED (Both batch and live) | `server/engine_adapter.py:454, 772`, `sih/mobile/causal_stream.py:58, 199`, `sih/models/inference.py:101, 195-197` | `a_max_mps2 = 3.5`, `a_min_mps2 = -5.0`, `tau_s = 0.25 s`. (Historical note: Initially documented as live only; audited in Step 0d confirming that offline benchmark batch `predict_velocities` applies `CausalSpeedSmoother` by default via `apply_smoothing=True`, ensuring identical causal smoothing in both batch and live paths). |
| | `KinematicSpeedObserver` | IMPLEMENTED+USED | `sih/engine/speed_observer.py:23-42, 61-150` | `alpha_base = 0.65`, `alpha_dynamic = 0.82`, `dynamic_accel_thresh = 0.35 m/s^2`, `beta = 0.001`, `max_bias = 0.80 m/s^2`. ZUPT: `a_var < 0.015`, `g_err < 0.35`, `w_norm < 0.05`. |
| | Low-Speed Crawl Clamping | **NOT FOUND IN CODE** | Claimed in docstrings; searched across `sih/` | `v_fwd <= max(v_entry + 1.2, 3.5)` does not exist in `dead_reckoning_engine.py` or anywhere in codebase. |
| | Pre-Blackout Speed Scale | IMPLEMENTED+USED | `sih/engine/dead_reckoning_engine.py:228`, `sih/round1/engine_hooks.py:80-92` | Base engine clips entry 15s GNSS/AI ratio to `[0.85, 1.25]` (1.35 Highway). Under `production.json`, `scale_level` blends 50% entry ratio and 50% 180s GNSS distance ratio (`w_history = 0.5`). |
| **EKF** | 15-State Error-State EKF | IMPLEMENTED+USED | `sih/fusion/es_ekf.py:62-640` | States: position (3), velocity (3), attitude quaternion (3), accel bias (3), gyro bias (3). |
| | Rate-Adaptive NHC | IMPLEMENTED+USED | `sih/fusion/es_ekf.py:567-575` | `dyn_sigma_lat = max(0.1, 1.2 * abs(w_z_corr))`, `R_nhc = diag([dyn_sigma_lat^2, 0.1^2])`. |
| | Physical Rest ZUPT/ZARU | IMPLEMENTED+USED | `sih/fusion/es_ekf.py:82-84, 526-540` | `zupt_accel_var_thresh = 0.04`, `zupt_gyro_norm_thresh = 0.04`, `zupt_gnorm_err_thresh = 0.60`. When `_stat_count > 5`, clamps velocity and updates gyro bias with `r_zupt = (0.001)^2`. |
| | Lorentzian Turn Damping | IMPLEMENTED+USED | `sih/fusion/es_ekf.py:583-592` | `omega_turn_ref = 0.02 rad/s (~1.15 deg/s)`, `turn_damping = 1.0 / (1.0 + (|w_z| / 0.02)^2)`. |
| | Highway Straight-Line Lock | **IMPLEMENTED, NOT USED** | `sih/fusion/es_ekf.py:731-760` | `update_straight_line_lock` is defined but NEVER called in `dead_reckoning_engine.py` or `engine_adapter.py`. |
| **Heading** | Pre-Blackout Heading Seeding | IMPLEMENTED+USED | `sih/fusion/es_ekf.py:268-345` | Speed regime selection (`v >= 2.5 m/s` Doppler bearing, fallback to 2-point displacement if `dt <= 1.5s`, fallback to last stable moving fix if crawling). Road corridor gentle alignment (`diff < 20°`, gain = 0.50). |
| | Seeding Error Metric | **WRONG / MISCOMPUTED** | `sih/engine/dead_reckoning_engine.py:587-595` | `seeded_hdg = float(np.degrees(ekf_pure._heading_rad))` is read AFTER the entire blackout loop finishes. Compares post-blackout heading against entry fix bearing, measuring total turn angle + drift instead of entry seeding error. |
| **Map** | Topological Map Matcher | IMPLEMENTED+USED | `sih/map/matcher.py:20-435` | `sigma_dist_m = 6.0m`, `sigma_heading_deg = 20.0°`. Hard reject: `d_perp > 15.0m` or `h_diff > 40.0°`. Ambiguity ratio: `1.5`. Hysteresis: `2` steps. Max turn gates: 105° (successors), 85° (single corridor), 70° (intent). |
| | Road Snapping Mode | IMPLEMENTED+USED | `sih/map/matcher.py:409-424` | If Urban or `h_diff > 40°` or turning intent: full 2D projection `best_proj`. Else (Highway/Arterial/Mixed): strictly perpendicular lateral projection `p_enu + d_lat * u_norm`. |
| | Kinematics Governor | IMPLEMENTED+USED | `sih/map/governor.py:17-135`, `dead_reckoning_engine.py:97-100, 312` | `a_lat_max = 2.2 m/s^2` Highway, `3.5 m/s^2` Arterial/Urban/Mixed. `v_max = sqrt(a_lat_max / kappa)`. Cap: `33.3 m/s` (120 km/h). |
| | Route Matcher (DFS) | **FLAG OFF** | `sih/engine/dead_reckoning_engine.py:817` | `enable_route_matching = False` by default in all benchmarks and production runs. |
| **Round 1/2** | T7 (Online Speed Calib) | IMPLEMENTED+USED | `sih/round1/online_speed_calib.py:33-98`, `config/round1/production.json:1` | `enabled = true`. Windows = 10.0s, min dist = 20m, band edges = `(0, 5, 10, 15, 22, 60) m/s`, prior = 30.0s, clip bounds = `[0.85, 1.15]`. |
| | T8 (Junction Corner Snap) | IMPLEMENTED+USED | `sih/round1/junction_anchor.py:101-200`, `config/round1/production.json:1` | `enabled = true`. Turn thresholds: on = `0.12 rad/s`, off = `0.05 rad/s`, quiet = `1.0s`, angle = `[50°, 140°]`, max duration = `15.0s`, max speed = `16.0 m/s`, base radius = `25m` (max `80m`), ambiguity = `1.8`, gain = `0.7`, max corr = `40m`. |
| | Blended Speed Scale | IMPLEMENTED+USED | `sih/round1/config.py:85-95`, `config/round1/production.json:1` | `scale_level = {enabled: true, source: blend, w_history: 0.5}`. Blends 50% entry ratio and 50% 180s GNSS distance ratio, clipped to `[0.85, 1.25]` (1.35 Highway). |
| | `entry_doppler_bearing` | IMPLEMENTED+USED | `config/round1/production.json:1`, `sih/round1/entry_bearing.py:41` | Set to `false` in `production.json`. Both batch benchmark and live adapter use geometric bearing. |
| | Flags T3, T4, T5, T9 | **FLAG OFF** | `config/round1/production.json:1` | Not specified in `production.json`; default to `enabled=False` (T3, T4), `mode="ai"` (T5), and `source="both"` (T9). |
| | Flag T10 (`scale_level`) | IMPLEMENTED+USED | `config/round1/production.json:1` | Specified as `{"enabled": true, "source": "blend"}` in `production.json`. |
| | Kill Switch | IMPLEMENTED+USED | `sih/round1/config.py:198-200` | `SIH_ROUND1_CONFIG="off"` disables all hooks and resolves canonical baseline checkpoint. |
| **Handoff** | 6-State FSM (`sih/handoff/manager.py`) | **IMPLEMENTED, NOT USED** | `sih/handoff/manager.py:53` | Implemented and tested in `tests/test_handoff.py`, but NEVER called in `server/router.py`, `server/engine_adapter.py`, or `dead_reckoning_engine.py`. |
| | Hermite Reconciliation | **IMPLEMENTED, NOT USED** | `sih/handoff/reconciliation.py:18` | Defined and unit tested; NEVER imported or called in live adapter or benchmark. |
| **Map Data** | Spatial Disk Cache | IMPLEMENTED+USED | `sih/map/cache.py:18`, `sih/map/network.py:863` | GeoJSON disk caching in `data/maps/cache` used by `load_trip_road_network`. |
| | Predictive Corridor Prefetcher | **IMPLEMENTED, NOT USED** | `sih/map/corridor_manager.py:49` | Tested in `test_map_ingestion.py`, but NOT instantiated or called in `server/router.py`. |
| **Hardware** | Motorcycle Roll / Contact Patch | **NOT FOUND IN CODE** | Searched repo for `contact_patch` / `motorcycle` | Zero engine code exists. Only mentioned in synthetic unit test `test_invariant_features.py:58`. |
| | Barometer Flyover Gating | **NOT FOUND IN CODE** | Searched repo for `barometer` / `TYPE_PRESSURE` | Zero code exists in Python or Android. |
| | Ribbon-Corridor Snapping | **NOT FOUND IN CODE** | Searched repo for `ribbon` | Zero code exists. |
| | NDK `ASensorManager` Capture | **NOT FOUND IN CODE** | Searched repo for `ASensorManager` | Android app uses standard Android Java `SensorManager` (`SensorStreamService.kt:70`). |
| | INT8 / FP16 Quantization | **NOT FOUND IN CODE** | Searched repo for `int8` / `quantiz` | No INT8 model, quantizer, or quantized runtime exists. (FP16 only used during training autocast). |
| **C++ & Exports** | Standalone C++ Engine (`engine/cpp`) | **IMPLEMENTED, NOT USED** | `engine/cpp/src/idr_core.cpp:1-145` | Compiles to `idr_core.dll` (113 KB), but is never invoked by Python or Kotlin. Contains no Round 1/2 logic. |
| | TorchScript Export | IMPLEMENTED, NOT USED | `models/exported/moe_velocity_model.torchscript.pt` | Exported from `round1_interval_lam0.5_s42.pt` (2.66 MB). NOT bundled into Android app `assets/` and NOT loaded by Kotlin. |
| | ONNX / TFLite Files | **NOT FOUND IN CODE** | Searched repo for `*.onnx` and `*.tflite` | Zero `.onnx` and zero `.tflite` files exist in the repository. |
| **Android App** | `MarkerHeading` Pointer Fix | IMPLEMENTED+USED | `MainActivity.kt:637, 830, 853` | Applied on all 3 marker rotation lines: local GNSS (line 637), HUD GNSS (line 830), and HUD DR (line 853). |

---

## Part 4: Names, Commands, and Numbers

### 1. Top-Level Classes and Functions
- **`sih/round1/config.py`**:
  - Classes: `StopDetectorParams` (L23), `GyroScaleParams` (L36), `SpeedModeParams` (L48), `OnlineSpeedCalibParams` (L55), `JunctionAnchorParams` (L66), `ScaleLevelParams` (L85), `SpeedScaleFixParams` (L98), `Round1Config` (L108)
  - Functions: `set_active_config` (L180), `get_active_config` (L189)
- **`sih/round1/engine_hooks.py`**:
  - Class: `Round1EngineHooks` (L33)
- **`sih/round1/entry_bearing.py`**:
  - Functions: `apply_entry_doppler` (L22), `live_override_enabled` (L41)
- **`sih/round1/gyro_scale.py`**:
  - Functions: `_wrap180` (L28), `estimate_gyro_scale` (L32)
- **`sih/round1/history.py`**:
  - Class: `PreBlackoutHistory` (L20)
  - Functions: `build_pre_blackout_history` (L35), `integrate_between` (L68), `build_history_from_buffers` (L81), `slice_history_tail` (L111)
- **`sih/round1/junction_anchor.py`**:
  - Functions: `_unit_from_bearing` (L32), `_axis_diff_deg` (L37), `_line_intersection` (L43), `_dist_point_segment` (L52), `find_corners` (L61)
  - Class: `TurnJunctionAnchor` (L101)
- **`sih/round1/model_select.py`**:
  - Functions: `_abs` (L20), `resolve_velocity_checkpoint` (L24), `load_mean_ensemble` (L36), `_make_module` (L57), `MeanMoEEnsemble` (L81)
- **`sih/round1/online_speed_calib.py`**:
  - Class: `BandSpeedCalibrator` (L33)
- **`sih/round1/stop_detector.py`**:
  - Class: `StopDetector` (L28)
- **`sih/models/interval_dataset.py`**:
  - Class: `IntervalSequenceSampler` (L26)
  - Function: `windows_from_sequence` (L142)
- **`sih/models/interval_loss.py`**:
  - Functions: `emulate_alpha` (L29), `interval_distance_loss` (L46), `interval_error_final` (L66)
- **`sih/models/inference.py`**:
  - Functions: `load_ai_model` (L20), `predict_velocities` (L92)
- **`sih/engine/dead_reckoning_engine.py`**:
  - Classes: `SteppableStepResult` (L27), `SteppableDeadReckoningEngine` (L52), `DeadReckoningEngine` (L353)
  - Function: `run_dead_reckoning_scenario` (L808)
- **`sih/engine/speed_observer.py`**:
  - Class: `KinematicSpeedObserver` (L16)
- **`sih/map/cache.py`**:
  - Class: `SpatialDiskCache` (L18)
- **`sih/map/corridor_manager.py`**:
  - Function: `compute_lookahead_radius` (L19)
  - Class: `PredictiveCorridorManager` (L49)
- **`sih/map/governor.py`**:
  - Classes: `RoadKinematicsGovernor` (L17), `DualRateRoadGovernor` (L138)
  - Function: `update_map_measurement` (L250)
- **`sih/map/hybrid_provider.py`**:
  - Class: `HybridIndiaMapProvider` (L18)
- **`sih/map/local_gis.py`**:
  - Class: `LocalGISProvider` (L19)
- **`sih/map/matcher.py`**:
  - Class: `HMMMapMatcher` (L20)
- **`sih/map/network.py`**:
  - Classes: `RoadSegment` (L20), `RoadNetwork` (L113)
  - Functions: `douglas_peucker_indices` (L72), `build_road_network_from_trip` (L464), `build_road_network_from_osm` (L505), `build_road_network_from_trip_masked` (L633), `compute_road_network_coverage` (L741), `audit_road_network_topology` (L785), `load_trip_road_network` (L863)
- **`sih/map/osm_client.py`**:
  - Class: `OSMOverpassClient` (L20)
- **`sih/map/provider.py`**:
  - Classes: `RoadNetworkMetadata` (L18), `IRoadNetworkProvider` (L31)
- **`sih/map/route_matcher.py`**:
  - Classes: `TurnEvent` (L37), `TurnSequence` (L45), `CandidateRoute` (L52), `RouteMatchResult` (L66), `RouteMatcher` (L495)
  - Functions: `extract_turn_sequence_from_imu` (L81), `extract_turn_sequence_from_route` (L194), `enumerate_routes_dfs` (L253), `score_candidate_route` (L344), `project_by_arclength` (L435)
- **`sih/fusion/es_ekf.py`**:
  - Functions: `skew` (L44), `wrap_pi` (L53)
  - Class: `ErrorStateEKF` (L62)
- **`sih/fusion/handoff.py`**:
  - Classes: `HandoffOutput` (L18), `GNSSHandoffManager` (L28), `GNSSDeficitHandler` (L146)
- **`sih/fusion/naive.py`**:
  - Class: `NaiveDeadReckoningFilter` (L25)
- **`sih/fusion/speed_smoother.py`**:
  - Class: `CausalSpeedSmoother` (L19)

---

### 2. Script CLI Flags from `--help`

- **`scripts/evaluate_heldout_seeds.py`**:
  No `argparse` CLI parser implemented. Directly invokes `run_heldout_evaluation()`.
- **`benchmarks/run_final_benchmark.py`**:
  ```text
  options:
    -h, --help            show this help message and exit
    --single              Run benchmark on a single random or specified seed
    --seed SEED           Specific random seed (default: random when --single is set, or 541098)
    --seeds SEEDS [SEEDS ...]
                          List of custom random seeds for multi-seed mode
    --fixed               Use standard canonical fixed 6 seeds [541098, 75496, 45736, 12345, 987654, 314159]
    --model-path MODEL_PATH
                          Path to custom model checkpoint to benchmark
    --map-source {osm,masked,trip}
                          Map network source for matcher: 'osm' (default, leak-free), 'masked' (leak-free trip), 'trip' (leaked reference)
    --compare-sources     Run 3-way map source comparison on 40 scenarios with fixed seed and write benchmark_map_source_comparison.csv and benchmark_results.json
    --enable-route-matching
                          Enable experimental route-level matching hypothesis evaluation (default: False)
    --disable-route-matching
                          Deprecated flag (route matching is already disabled by default)
  ```
- **`scripts/quick_parity.py`**:
  ```text
  options:
    -h, --help  show this help message and exit
    --raw       Run in raw-input mode (adapter computes mount + features + AI speeds)
  ```
- **`scripts/train_interval_moe.py`**:
  ```text
  options:
    -h, --help            show this help message and exit
    --base BASE
    --lam LAM
    --epochs EPOCHS
    --steps STEPS         optimizer steps per epoch
    --batch BATCH         sequences per step (lower to 2 on CUDA OOM)
    --lr LR
    --seed SEED
    --cal-s CAL_S
    --horizons HORIZONS [HORIZONS ...]
    --pointwise-stride POINTWISE_STRIDE
                          every k-th step feeds the per-sample loss
    --max-rmse-increase MAX_RMSE_INCREASE
                          reject epochs whose val RMSE worsens more than this fraction
    --out OUT
    --overwrite
    --smoke               2 epochs x 3 steps, for a pipeline check only
  ```
- **`scripts/train_can_moe.py`**:
  ```text
  options:
    -h, --help            show this help message and exit
    --epochs EPOCHS       Number of training epochs (default: 15)
    --batch-size BATCH_SIZE
                          Batch size (default: 64)
    --lr LR               Learning rate (default: 1e-3)
    --seed SEED           Fixed random seed (default: 42)
    --checkpoint-path CHECKPOINT_PATH
                          Path to save best checkpoint
  ```
- **`scripts/round1_eval.py`**:
  ```text
  options:
    -h, --help            show this help message and exit
    --configs CONFIGS [CONFIGS ...]
    --tag TAG
    --seeds SEEDS         canonical | heldout | comma list
    --i-have-user-approval
                          required for held-out seeds
    --model-path MODEL_PATH
                          checkpoint, or comma-separated list = mean ensemble
    --map-source {osm,masked,trip}
    --assert-parity ASSERT_PARITY
                          reference *_scenarios.csv; first config must match exactly
  ```
- **`scripts/round1_compare.py`**:
  ```text
  positional arguments:
    a
    b

  options:
    -h, --help       show this help message and exit
    --name-a NAME_A
    --name-b NAME_B
  ```
- **`server/router.py`**:
  ```text
  options:
    -h, --help   show this help message and exit
    --host HOST  Host interface (default 0.0.0.0)
    --port PORT  Port to listen on (default 8765)
    --trip TRIP  Optional preloaded trip (default: None, dynamically loaded per scenario)
  ```

---

### 3. Test Counts & Verification Results
- **`pytest --collect-only -q` Total**: **125 tests**
- **`tests/test_round1.py` Total**: **21 tests**
- **Full Pytest Suite Run**:
  - Passed: **123**
  - Failed: **1** (`tests/test_report_integrity.py::test_benchmark_script_has_no_stale_literals` fails because `benchmarks/run_final_benchmark.py` includes `"11.59% / 35.80% / 17 / pure 26.31%"`, containing string `"/ 35"` which is banned by the test regex)
  - Skipped: **1** (`tests/test_diagnostics_and_sync.py::test_torchscript_model_has_no_recalibration`)
  - Errors: **0**
  - Total Duration: 618.09s (10 min 18s)

---

### 4. Hard-Coded Claims & Statuses in `benchmarks/run_final_benchmark.py`

| File Line Number | Code Snippet | What Is Hardcoded |
| :--- | :--- | :--- |
| `run_final_benchmark.py:1075` | `\| **Legacy Single Model (non-causal, not deployable)** \| **27.33%** \| **11.96%** (P90: 31.39%, Tier-1: 18/40, Beats Pure: 33/40) \| **< 10.0%** \| **Non-Causal Reference** \|` | Entire row of legacy benchmark numbers and status are hardcoded string literals. |
| `run_final_benchmark.py:1079` | `\| **Initial Heading Seeding Error**\| 28.4° (unobservable magnetometer) \| **{mean_hdg_seed_err:.2f}°** (Speed-Regime GPS Vector) \| < 20.0° \| **PASSED** \|` | Status `"**PASSED**"` is hardcoded regardless of computed error value. Baseline error `"28.4°"` is also hardcoded. |
| `run_final_benchmark.py:1099` | `\| **Low Speed / Traffic Crawl** \| ... \| Velocity entry clamping and ZUPT prevent low-speed stationary drift \|` | Text claims "Velocity entry clamping" is active, which is completely absent from code. |
| `run_final_benchmark.py:1220` | `\| **Tier 3: Highway Cruising** \| ... \| **{hwy_drift:.2f}% Median Drift** (Sub-lane accuracy) \| ... \|` | `"(Sub-lane accuracy)"` is hardcoded directly into the template table cell. |
| `run_final_benchmark.py:1238` | ` (Uncalibrated) (SO(3) Rotation Matrix) (Invariant Speed Scaling) (Closed-Loop NHC) (Sub-Lane Precision)` | `"(Sub-Lane Precision)"` is hardcoded as the Stage 5 column header. |
| `run_final_benchmark.py:1309` | `* Phase 4 corner projection and topological snapping maintained sub-lane corridor tracking (**{uc['map_drift_pct']:.2f}% drift**).` | Text `"maintained sub-lane corridor tracking"` is hardcoded. |
| `run_final_benchmark.py:1315` | `#### Spotlight #{pr['scenario_id']:02d}: Sub-Lane Ultra-Precision Outage` | Title `"Sub-Lane Ultra-Precision Outage"` is hardcoded. |
| `run_final_benchmark.py:1370` | ` - Simultaneous sub-10% performance across all disparate environments is definitive proof of structural generalization without overfitting.` | Hardcoded claim asserting simultaneous sub-10% across all domains, even though Highway median is 11.85% and Urban is 13.79% on Seed 541098. |

---

## Part 5: Claims to Check Table

| # | README / Documentation Claim | Code Reality | Status | Evidence (`file:line`) |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **Phase 7 Edge Causal Runtime**: "Direct on-device Kotlin / NDK ONNX Runtime or TFLite execution on phone" | Android app is a pure sensor logger/streamer; no ONNX/TFLite models or NDK inference libraries exist. Zero ML runs on phone. | **NOT IMPLEMENTED** | `android/app/src/main/java/com/recursiveminds/idr/`, zero `.onnx`/`.tflite` files |
| 2 | **Sub-Lane Accuracy**: "Sub-lane precision throughout (< 3.5m lateral offset)" | P90 drift is 32.91% across held-out seeds. Scenarios frequently drift 15m to 50m over 60–75s outages. "Sub-lane" is hardcoded in report strings. | **WRONG** | `benchmarks/run_final_benchmark.py:1220, 1238`, `artifacts/heldout_seed_results.json` |
| 3 | **Simultaneous Sub-10% Generalization**: "Simultaneous sub-10% performance across all disparate environments" | Seed 541098 domain medians: Highway 11.85%, Arterial 6.64%, Urban 13.79%, Mixed 12.38%. Three of four domains exceed 10%. | **WRONG** | `benchmarks/run_final_benchmark.py:1370`, `benchmark_results.json` |
| 4 | **Low-Speed Crawl Clamping**: `v_fwd <= max(v_entry + 1.2 m/s, 3.5 m/s)` during crawl entries | Formula does not exist anywhere in `dead_reckoning_engine.py`, `speed_observer.py`, or `es_ekf.py`. | **NOT FOUND IN CODE** | Search across `sih/` yields zero occurrences |
| 5 | **ZARU Highway Straight-Line Lock**: "Freezes yaw gyro bias when v > 15 m/s and \|omega_z\| < 0.005 rad/s for > 2.0s" | `update_straight_line_lock` is implemented in `es_ekf.py:731`, but NEVER called in `SteppableDeadReckoningEngine.step` or `engine_adapter.py`. | **IMPLEMENTED, NOT USED** | `sih/fusion/es_ekf.py:731`, `dead_reckoning_engine.py:285` |
| 6 | **Phase 6 Seamless Handoff State Machine**: "6-state FSM, Chi-Square NIS gating, C^2 Hermite smoothstep reconciliation" | `SeamlessGNSSHandoffManager` and `HermiteReconciler` exist in `sih/handoff/`, but are NEVER imported or invoked in `server/router.py` or `dead_reckoning_engine.py`. | **IMPLEMENTED, NOT USED** | `sih/handoff/manager.py:53`, `sih/handoff/reconciliation.py:18`, `server/router.py` |
| 7 | **Initial Heading Seeding Error Metric**: "Initial heading seeding error is 17.15°" | `hdg_seed_err` is evaluated after the blackout finishes by comparing `ekf_pure` final exit heading to entry GNSS bearing, measuring turn angle + drift. | **WRONG** | `sih/engine/dead_reckoning_engine.py:587-595` |
| 8 | **Speed-Adaptive Lookahead Corridor Caching**: "Predictive lookahead corridor manager prefetching Overpass OSM road geometry" | `PredictiveCorridorManager` is tested in unit tests, but `server/router.py` loads static GeoJSON files directly via `load_trip_road_network`. | **IMPLEMENTED, NOT USED** | `sih/map/corridor_manager.py:49`, `server/router.py:721` |
| 9 | **Route Matcher (Topological Branch Disambiguation)**: "Route-level hypothesis matching across branching junctions" | Route matching is disabled by default (`enable_route_matching = False`) in all production engines and benchmarks. | **FLAG OFF** | `sih/engine/dead_reckoning_engine.py:817` |
| 10 | **START Button Warm-Up Gating**: "START requires mount yaw lock (8 turn events) and warm buffers" | `btnStart.isEnabled = true` unconditionally whenever `!isInBlackout`. Users can press START at 0/8 turns, before leveling (3s), and before feature warm-up (6s). | **WRONG** | `android/app/src/main/java/com/recursiveminds/idr/ui/MainActivity.kt:676-693` |
| 11 | **NDK ASensorManager, Barometer, Motorcycle Frame**: "NDK ASensorManager capture, Barometer flyover gating, motorcycle lean frame" | Zero code exists in the repository for any of these features. Android uses Java `SensorManager`, no barometer is sampled, motorcycle is only in a synthetic test. | **NOT FOUND IN CODE** | Searches for `ASensorManager`, `TYPE_PRESSURE`, `contact_patch` |
| 12 | **INT8 / FP16 Quantized Model**: "INT8 quantized TorchScript edge model" | Exported model is FP32 TorchScript (2.66 MB). No INT8 model, quantizer script, or quantized runtime exists. | **NOT FOUND IN CODE** | `models/exported/moe_velocity_model.torchscript.pt`, `sih/models/export_onnx.py` |
| 13 | **Production Speed Model**: "T6 speed model round1_interval_lam0.5_s42.pt loaded by default" | Confirmed in code: `production.json` specifies `round1_interval_lam0.5_s42.pt`, which is resolved and loaded by `load_ai_model`. | **MATCH** | `config/round1/production.json:1`, `sih/round1/model_select.py:24`, `sih/models/inference.py:34` |
| 14 | **Round 1/2 T7 Online Speed Calibration**: "Learns speed shape factor from trailing 180s GNSS distance vs AI distance" | Confirmed in code: `BandSpeedCalibrator` fits 10s windows over 180s pre-blackout history with 30s shrinkage prior. Falls back to 1.0 if < 180s. | **MATCH** | `sih/round1/online_speed_calib.py:33-98`, `config/round1/production.json:1` |
| 15 | **Round 1/2 T8 Junction Corner Snapping**: "Snaps position along the outgoing road centerline upon detecting completed turns" | Confirmed in code: `TurnJunctionAnchor` detects turns (`0.12` to `0.05 rad/s`, `50°` to `140°`), finds geometric road corners, and applies 70% along-track correction. | **MATCH** | `sih/round1/junction_anchor.py:101-200`, `config/round1/production.json:1` |
| 16 | **Streaming / Batch Bit Parity**: "Exact 0.0000 m batch vs. streaming parity across canonical scenarios" | Confirmed in code: `quick_parity.py` verifies exact 0.0000 m endpoint and trajectory diff when using pre-blackout history buffer. | **MATCH** | `scripts/quick_parity.py:255`, `server/engine_adapter.py:725` |
| 17 | **`MarkerHeading` Pointer Fix**: "OSM pointer rotation corrected for osmdroid counter-clockwise canvas rotation" | Confirmed in code: `MarkerHeading.toMarkerRotation` is applied on all three marker rotation lines (local GNSS, HUD GNSS, HUD DR). | **MATCH** | `android/app/src/main/java/com/recursiveminds/idr/ui/MarkerHeading.kt:18`, `MainActivity.kt:637, 830, 853` |
