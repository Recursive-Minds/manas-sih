# Technical Audit: Smartphone On-Device Reality vs Claims

## 1. sih/mobile/causal_stream.py Audit

- **Classification**: PARTIAL standalone reimplementation (not a wrapper around production core).
- **Execution**: Runnable in Python, but diverges significantly from `sih/engine/dead_reckoning_engine.py`.

### Call Trace (`MobileDeadReckoningStream`)
- `__init__`:
  - Instantiates `ErrorStateEKF` (`sih/fusion/es_ekf.py:62`).
  - Instantiates `CausalSpeedSmoother(a_max=3.5, a_min=-5.0, tau=0.25)` (`causal_stream.py:20, 58`).
  - Instantiates `HMMMapMatcher(road_network, domain="Highway")` (`causal_stream.py:66`).
  - Instantiates `DualBandSpectralExtractor(sampling_rate=10.0)` (`causal_stream.py:73`).
  - Allocates ring buffers `_window_imu = deque(maxlen=60)`, `feature_buffer = deque(maxlen=60)` (`causal_stream.py:77-78`).
  - Loads TorchScript model via `torch.jit.load("models/exported/moe_velocity_model.torchscript.pt")` and normalization arrays (`causal_stream.py:96-121`).
- `on_gnss_sample`:
  - Calls `self.ekf.update_gnss(pos_enu, vel_enu, accuracy_h_m)` (`causal_stream.py:151`).
  - Sets `is_gnss_healthy = True`, resets `consecutive_outage_samples = 0` (`causal_stream.py:152-153`).
- `on_imu_sample`:
  - Uses raw inputs: `ax_corr = ax, ay_corr = ay, az_corr = az, gx_corr = gx, gy_corr = gy, gz_corr = gz` (`causal_stream.py:174-179`).
  - Appends to `_window_imu` (`causal_stream.py:186`).
  - Calls `self.spectral_extractor.compute_window_features` when `len >= 8` (`causal_stream.py:189`).
  - Pushes 12-dim vector into `feature_buffer` (`causal_stream.py:192-205`).
  - Calls `self._estimate_forward_speed` -> `self.torch_model(t_short, t_long)` when `len(feature_buffer) >= 60` (`causal_stream.py:283-294`).
  - Calls `self.speed_smoother.update(v_raw, dt)` -> `v_smooth` (`causal_stream.py:232`).
  - Packages mock `cal = CalibratedSample(...)` and mock `vel = VelocityEstimate(...)` (`causal_stream.py:234-242`).
  - Calls `fused = self.ekf.predict(cal, vel)` (`causal_stream.py:243`).
  - Calls `self.matcher.match(fused, ekf=self.ekf, domain="Highway", v_fwd=v_smooth)` (`causal_stream.py:247`).
  - **Critical Bug**: Return value of `self.matcher.match` is discarded. Returns `FusedPosition(position_enu_m=fused.position_enu_m)` (`causal_stream.py:247, 265-270`). Map-matched output is never applied.

### Divergences from `dead_reckoning_engine.py`
- **Mount Calibration**:
  - `dead_reckoning_engine.py:161-177`: `MountCalibrator` estimates SO(3) leveling matrix `R_mount` from stationary gravity and GNSS acceleration vectors, rotating accelerations and angular rates into leveled vehicle frame.
  - `causal_stream.py:171-180`: **Skipped**. Identity transform (`ax_corr = ax`). No SO(3) leveling.
- **Model Inference**:
  - `dead_reckoning_engine.py:270-305`: 5-fold LOTO ensemble with fold discounting (`D=0.50`) and gating (`sih/models/ensemble_gating.py`).
  - `causal_stream.py:96-109`: Single-checkpoint TorchScript model (`moe_velocity_model.torchscript.pt`). Zero ensemble gating.
- **Speed Scale Factor (Alpha)**:
  - `dead_reckoning_engine.py:274-297`: Dynamically estimates pre-blackout scale `alpha = sum(v_GNSS) / sum(v_pred)` over preceding 20-25s.
  - `causal_stream.py:56`: **Hardcoded `self.speed_scale = 1.00`**. Never estimated.
- **Speed Observer**:
  - `dead_reckoning_engine.py:19-24, 308-316`: `KinematicSpeedObserver` (`sih/speed/observer.py`) with physical rest detection (`accel_std < 0.08`, `gyro_std < 0.03`), causal forward acceleration integration, zero-velocity clamping, and vibration blending.
  - `causal_stream.py:20-43, 58`: **Missing**. Replaced by naive 4-line exponential slew-rate smoother `CausalSpeedSmoother`.
- **Heading Seeding**:
  - `dead_reckoning_engine.py:215-269`: 2-point GNSS trajectory vector seeding, circular variance consistency check (`circ_std <= 12 deg`), pre-blackout alignment, and decisive straight-line innovation (`gain=0.85`).
  - `causal_stream.py:145`: **Missing**. Unfiltered heading from last raw GNSS bearing or 0.0.
- **Road Kinematics Governor**:
  - `dead_reckoning_engine.py:321-326, 477-493`: `RoadKinematicsGovernor` (`sih/fusion/road_governor.py`) clamping centripetal acceleration (`a_lat <= 1.2 m/s^2`) and curvature.
  - `causal_stream.py`: **Missing**. Zero governor integration.
- **Map Matching**:
  - `dead_reckoning_engine.py:328-348, 497-535`: `RouteMatcher` (`sih/map/route_matcher.py`) with Dijkstra route extraction, topological bonus, candidate pruning, and anti-boundary watchdog.
  - `causal_stream.py:246-248`: `RouteMatcher` missing. Directly invokes `HMMMapMatcher` with hardcoded `domain="Highway"`, and discards matched coordinate output entirely.

---

## 2. Parity Test (Scenario #26, Seed 541098, Trip S-S3a)

- **Scenario Parameters**: Duration 75.0s, Distance 892.8m, Blackout window `1267320000000` to `1342320000000` ns.
- **Execution**: Sample-by-sample incremental feed into `MobileDeadReckoningStream`.
- **Future Sample Usage**: NONE (Verified: `len(feature_buffer) <= 60`, strictly causal ring buffers).

### Empirical Results
- **Batch Engine (`dead_reckoning_engine.py`)**:
  - Pure EKF Endpoint Error: 28.24 m (3.16% drift)
  - Map-Matched Endpoint Error: 4.40 m (0.49% drift)
- **Causal Stream (`causal_stream.py`)**:
  - Pure EKF Endpoint Error: 179.24 m (20.08% drift)
  - Map-Matched Endpoint Error: 179.24 m (20.08% drift)
- **Trajectory Divergence (Stream vs Batch)**:
  - Map Trajectory Maximum Difference: 215.88 m
  - Map Trajectory Endpoint Difference: 211.19 m
  - Pure Trajectory Maximum Difference: 224.34 m
  - Pure Trajectory Endpoint Difference: 211.50 m
- **Reason for 211m Divergence**: `causal_stream.py` omits mount calibration (no gravity leveling), lacks pre-blackout heading seeding and alpha scaling, replaces the kinematic speed observer with a naive smoother, and drops map-matched output.

---

## 3. C++ Engine Audit (`engine/cpp/src/idr_core.cpp`)

- **Compilation**: Compiles with MinGW `g++ -std=c++17` (`scratch/idr_core_test.dll` generated without errors).
- **Architecture**: 145 lines total (`engine/cpp/src/idr_core.cpp:1-145`).
- **Filter Implementation**: **Not an EKF**.
  - Allocates 15x15 covariance array `P` (`idr_core.cpp:26, 38-54`).
  - Euler heading update: `heading -= gz_corr * dt` (`idr_core.cpp:90`).
  - Euler position update: `p += v * dt` (`idr_core.cpp:102-104`).
  - Covariance: strictly adds diagonal `Q * dt` (`idr_core.cpp:111-120`).
  - **Zero Kalman measurement updates**: No `update_gnss`, no `update_nhc`, no `update_zupt`. No Kalman gain `K = P H^T (H P H^T + R)^-1`.
  - Bias states `ba` and `bg` are never updated (remain 0.0 forever).
  - Acceleration inputs `ax, ay, az` and gyro `gx, gy` are completely unused (`idr_core.cpp:74-75`).
- **Model Loading**: **NONE** (`idr_core.cpp:76`). Forward speed is accepted as an external caller-supplied float scalar (`forward_speed_ref`). No LibTorch, ONNX Runtime, TFLite, or NCNN code.
- **JNI Bindings**: **NONE**. Exports plain C functions via `extern "C"` / `IDR_EXPORT` (`idr_core.cpp:58-144`). No `jni.h` included, no `Java_*` symbols anywhere in repository.
- **ARM Execution Evidence**: **NONE**.
  - No Android Gradle/NDK project files (`build.gradle`, `Android.mk`, `CMakeLists.txt` for NDK).
  - Zero ARM binaries in repo. Only pre-compiled Windows x64 PE binaries exist (`engine/cpp/idr_core.dll` [MZ header], `engine/cpp/idr_core.lib`).

---

## 4. Feature Pipeline & Resampling Audit

- **Sampling Rate (`sih/data/spectral.py:18`)**: Band powers are computed at `fs = 10.0` Hz.
- **Physical Validity of 3-8 Hz Band**:
  - Nyquist frequency at 10 Hz is 5.0 Hz (`fs / 2`).
  - Any frequency content from 5.0 to 8.0 Hz cannot exist at 10 Hz without aliasing.
  - In `sih/data/spectral.py:20`, Band B is defined as `(1.5, 4.5)` Hz (clamped below Nyquist).
  - Any docstring or documentation claim citing "3-8 Hz frequency vibration power" (e.g. `CLAUDE.md:86`) is physically impossible at 10 Hz.
- **Raw IO-VNBD Resampling Pipeline**:
  - Raw IO-VNBD dataset CSV files (`data/raw/iovnbd_trips/*.csv`) are ALREADY logged at 10 Hz (timestamps increment by 100 ms: `31321, 31421, 31521 ms`).
  - `GenericDataLoader.load_dataframe` (`sih/data/loader.py:73-150`) performs **zero downsampling or interpolation**; rows are read directly 1:1.
  - **Requirement for 100-200 Hz Phone IMU**:
    - A device streaming at 100-200 Hz MUST pass raw IMU through an anti-aliasing low-pass filter (cutoff <= 4.5 Hz) and decimate to exactly 10.0 Hz (10x or 20x decimation) before feeding `DualBandSpectralExtractor` and the 12-channel neural network.

---

## 5. TorchScript Export Audit

- **Script**: `scripts/export_onnx.py` exports `models/exported/moe_velocity_model.torchscript.pt` (defaults to production checkpoint `models/checkpoints/round1_interval_lam0.5_s42.pt`, with pre-round-1 checkpoint `best_moe_velocity_model.pt` selectable via `--checkpoint`).
- **Numerical Parity**: Verified via independent scratch script (`scratch/test_torchscript_parity.py`) in fresh Python process:
  - Input: random tensor `(1, 12, 20)` and `(1, 12, 60)`.
  - Eager output: `v = 12.397964 m/s, var = 0.496575`.
  - TorchScript output: `v = 12.397964 m/s, var = 0.496575`.
  - Max Absolute Difference (v): `0.000000e+00 m/s`.
  - Max Absolute Difference (var): `0.000000e+00`.
- **Latency Origin ("2.68 ms")**:
  - Measured in `scripts/export_onnx.py:138-153` on host laptop CPU (x86_64, Windows, single-thread PyTorch `torch.set_num_threads(1)`).
  - 500 loop iterations using `time.perf_counter()`.
  - **Never measured on a smartphone, ARM chip, or Android device**.

---

## 6. Unit Test Audit (`tests/test_mobile_stream.py`)

- **Assertion Content**:
  - `test_gnss_ingestion` (`line 21`): Asserts `is_gnss_healthy == True`, `consecutive_outage_samples == 0`.
  - `test_imu_streaming_and_outage` (`line 35`): Feeds 1 dummy GNSS sample + 25 synthetic constant IMU samples (`ax=0, az=9.81`). Asserts `fused is not None`, `not isnan()`, and `fused.is_dead_reckoning == True`.
  - `test_torchscript_auto_load_and_inference` (`line 77`): Feeds 65 dummy IMU samples (`ax=0.2, az=9.81`). Asserts `torch_model is not None`, `len(feature_buffer) >= 60`, and `speed_mps` is float and not NaN.
  - `test_engine_init` (`line 101`): Asserts 2 constructor parameters match input arguments.
- **Mocks**: No `unittest.mock` used. Live code executed, but on 100% trivial synthetic constants (`ax=0, az=9.81`).
- **Omissions**: Zero trajectory assertions, zero real data, zero comparison to batch engine or ground truth, and map matching is explicitly disabled (`enable_map_matching=False`, `line 18`).

---

## 7. Hardcoding Scan (`sih/`)

- **Scenario IDs**: Zero hardcoded scenario runtime branching. Mentions of "Scenario 12" and "Scenario 25" in `sih/map/route_matcher.py:17, 19, 624` are descriptive comments only.
- **Trip Names**:
  - `sih/models/ensemble_gating.py:28-39`: Dictionary registry mapping LOTO checkpoint filenames to trip names (`"S-S1": "Fold 2 (S-S1)"`, etc.).
  - `sih/models/can_dataset.py:24`: `ALLOWED_TRAIN_TRIPS = {"S-M", "S-S1", "S-S2"}`.
  - `sih/data/sequence_dataset.py:240` and `sih/data/loader.py:86`: Special case parsing for `S-S4` sensor column swap.
  - `sih/data/downloader.py:14-26`: Git LFS hash table.
- **Timestamps**: Zero hardcoded absolute timestamp literals in `sih/`.
- **Clock Offsets**: Hardcoded in `config/can_sync.json` (`S-M: +2.3s`, `S-S1: 0.0s`, `S-S2: +8.6s`, `S-S3a: -6.9s`) and enforced via `sih/data/can_sync.py:28-48`. S-S4 permanently excluded from CAN supervision.

---

## Final Verdicts & Android Deployment Effort

| Component | Verdict | Core Deficit |
| :--- | :--- | :--- |
| `sih/mobile/causal_stream.py` | **PARTIAL** | Runs causally in Python, but skips leveling calibration, ensemble gating, pre-blackout alpha/heading seeding, and kinematic speed observer. Discards map matching output. Produces 20.08% drift vs 0.49% batch. |
| `engine/cpp/src/idr_core.cpp` | **SPEC-ONLY** | 145-line 2D Euler vector dead-reckoner. Zero model inference, zero JNI, zero Kalman measurement updates, zero ARM artifacts. |

### Engineering Effort to a Working On-Device Android Loop
- **Estimated Effort**: 3 to 4 Weeks (Full-Time Senior Systems/Mobile Engineer).
- **Required Workstreams**:
  1. **Inference Runtime (3-4 days)**: Integrate ONNX Runtime Mobile or PyTorch Mobile C++ via Android NDK. Wrap `models/exported/moe_velocity_model.torchscript.pt` or `.onnx`.
  2. **Sensor Decimator & Anti-Aliasing (2-3 days)**: C++/Java circular buffer resampling 100-200 Hz Android IMU (`SensorManager.SENSOR_DELAY_FASTEST`) down to 10.0 Hz with low-pass filtering.
  3. **Port Production Core to C++ (7-10 days)**: Reimplement genuine 15-state ES-EKF with NHC/ZUPT, SO(3) leveling (`MountCalibrator`), pre-blackout heading seeding, and Kinematic Speed Observer in C++ with Eigen3.
  4. **JNI Interface & Android App Harness (4-5 days)**: JNI bridge between Kotlin/Java foreground service and C++ core; background threading, GNSS/IMU listener ingestion, and WGS-84 coordinate streaming.
