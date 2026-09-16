# Mobile App Deployment Specification: Smartphone Intelligent Dead Reckoning (IDR)

This document specifies the end-to-end architecture, API integration contracts, computational budgets, and hardware lifecycle protocols for embedding the Smartphone Intelligent Dead Reckoning (IDR) engine into production mobile applications (Android / iOS).

---

## 1. System Architecture Overview

The mobile deployment pipeline transforms uncalibrated raw phone sensors into sub-lane, real-time dead-reckoning trajectory estimates during extended GNSS outages (urban canyons, tunnels, multi-level flyovers, and underpasses).

```
+----------------------------------------------------------------------------------+
|                            MOBILE OPERATING SYSTEM (Android/iOS)                |
|                                                                                  |
|  [Android SensorManager]                [FusedLocationProviderClient]            |
|  TYPE_ACCELEROMETER (50 Hz)             1 Hz GNSS Fixes (Lat, Lon, Alt, Spd)     |
|  TYPE_GYROSCOPE     (50 Hz)             Accuracy & Bearing                       |
+------------------------+------------------------------------+--------------------+
                         |                                    |
                         v                                    v
+----------------------------------------------------------------------------------+
|               MOBILE DEAD RECKONING STREAM (C++ Core / Kotlin / Swift)           |
|                                                                                  |
|  +---------------------------+       +----------------------------------------+  |
|  | Mount Auto-Calibrator     |       | GNSS Handoff Manager                   |  |
|  | Dynamic gravity alignment | <---> | Outage detector (DOP, SNR, std > 5.0m) |  |
|  | Vehicle-to-phone rotation |       | Pre-blackout Doppler / vector seeder   |  |
|  +-------------+-------------+       +-------------------+--------------------+  |
|                |                                         |                       |
|                v                                         v                       |
|  +---------------------------+       +----------------------------------------+  |
|  | AI Forward Speed Engine   |       | 15-State Error-State EKF (ES-EKF)      |  |
|  | PyTorch Mobile / ONNX RT  | ----> | State: [p(3), v(3), theta(3), ba, bg]  |  |
|  | 12-channel feature buffer |       | Non-Holonomic Constraints (NHC)        |  |
|  | Inference: 2.68 ms/step   |       | Zero Velocity Updates (ZUPT)           |  |
|  +---------------------------+       | Straight-Line Heading Lock (ZARU)      |  |
|                                      +-------------------+--------------------+  |
|                                                          |                       |
|                                                          v                       |
|  +----------------------------------------------------------------------------+  |
|  | Topological Road Snapping Engine (Offline Vector Map)                      |  |
|  | Spatial R-Tree candidate search within 15m corridor                        |  |
|  | Heading consistency gating (discrepancy < 65 deg)                         |  |
|  | Multi-hypothesis junction hardening & decisive turn steering               |  |
|  +----------------------------------------------------------------------------+  |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
|                       NAVIGATION UI & TURN-BY-TURN CLIENT                        |
|   Smooth 50 Hz sub-lane vehicle marker, turn guidance, corridor visualizer       |
+----------------------------------------------------------------------------------+
```

---

## 2. Sensor Ingestion & Android Lifecycle

### 2.1 Hardware Sensor Rates & Configuration
- **Accelerometer (`Sensor.TYPE_ACCELEROMETER`)**: Ingested at `SENSOR_DELAY_GAME` (50 Hz nominal, 20 ms period). Provides raw vehicle body acceleration including gravity.
- **Gyroscope (`Sensor.TYPE_GYROSCOPE`)**: Ingested at `SENSOR_DELAY_GAME` (50 Hz nominal, 20 ms period). Measures chassis angular turn rate without software filtering.
- **GNSS (`FusedLocationProviderClient`)**: Ingested at 1 Hz with `PRIORITY_HIGH_ACCURACY`. Used for EKF coordinate grounding, continuous speed scale adaptation, and mount calibration.

### 2.2 Threading & Concurrency Model
To prevent UI stutter and maintain strict 20 ms timing:
1. **Sensor Ingestion Thread**: A dedicated high-priority background handler thread (`Process.THREAD_PRIORITY_URGENT_DISPLAY` or `THREAD_PRIORITY_MORE_FAVORABLE`).
2. **Batching Buffer**: IMU readings are pushed to a lock-free ring buffer (capacity: 256 samples).
3. **Execution Cadence**: 
   - IMU step + EKF propagation: Executes on each IMU tick (latency <= 0.25 ms).
   - AI velocity inference: Evaluated every 10 IMU ticks (100 ms interval, 10 Hz) over a sliding window of 60 steps (latency: 2.68 ms).
   - Map matching: Evaluated on 10 Hz velocity pulses (latency <= 0.40 ms).
   - Total CPU time per 100 ms cycle: **3.33 ms** (3.3% CPU core occupancy).

---

## 3. Edge AI Velocity Model Integration

### 3.1 Model Artifacts
The forward velocity model is compiled into an optimized, self-contained edge binary:
- **File**: `models/exported/moe_velocity_model.torchscript.pt`
- **File Size**: **2.66 MB**
- **Architecture**: Dual-branch Mixture of Experts (Short TCN + Long BiLSTM + Gating Network).
- **Parity**: Exact 0.000000 m/s maximum discrepancy compared to full workstation PyTorch.
- **Latency**: **2.68 ms** average execution on 8-core CPU (373 Hz throughput).
- **Normalization Weights**: `models/exported/normalization_params.npz` (12-channel mean and std vectors).

### 3.2 12-Channel Feature Extraction Buffer
The model requires a sliding temporal buffer of 60 timesteps (6.0 seconds at 10 Hz IMU or decimated 50 Hz):
- Channel 0: Calibrated forward acceleration `f_x` (m/s^2)
- Channel 1: Calibrated lateral acceleration `f_y` (m/s^2)
- Channel 2: Calibrated vertical acceleration `f_z` (m/s^2)
- Channel 3: Angular roll rate `omega_x` (rad/s)
- Channel 4: Angular pitch rate `omega_y` (rad/s)
- Channel 5: Angular yaw rate `omega_z` (rad/s)
- Channel 6: Acceleration norm `norm(f)` (m/s^2)
- Channel 7: Gyroscope norm `norm(omega)` (rad/s)
- Channel 8: Gravity deviation `norm(f) - 9.81` (m/s^2)
- Channel 9: Absolute yaw rate `abs(omega_z)` (rad/s)
- Channel 10: High-frequency spectral power (3-8 Hz road vibration cue)
- Channel 11: Longitudinal-to-vertical ratio `abs(f_x) / (abs(f_z) + 1e-4)`

### 3.3 Android PyTorch Mobile Integration (Kotlin)
Add Gradle dependency:
```groovy
implementation 'org.pytorch:pytorch_android_lite:1.13.1'
implementation 'org.pytorch:pytorch_android_torchvision_lite:1.13.1'
```

Kotlin Inference Stub:
```kotlin
class EdgeVelocityInference(context: Context) {
    private val module: Module = LiteModuleLoader.load(assetFilePath(context, "moe_velocity_model.torchscript.pt"))
    private val normMean = floatArrayOf(...) // 12 elements
    private val normStd  = floatArrayOf(...) // 12 elements

    fun predictSpeed(featureBuffer60x12: Array<FloatArray>): Float {
        // x_long: shape (1, 12, 60), x_short: shape (1, 12, 20)
        val tensorShort = Tensor.fromBlob(flattenAndSlice(featureBuffer60x12, 20), longArrayOf(1, 12, 20))
        val tensorLong  = Tensor.fromBlob(flattenAndSlice(featureBuffer60x12, 60), longArrayOf(1, 12, 60))

        val outputTuple = module.forward(IValue.from(tensorShort), IValue.from(tensorLong)).toTuple()
        val vEst = outputTuple[0].toTensor().dataAsFloatArray[0]
        return maxOf(0.0f, vEst)
    }
}
```

---

## 4. C++ Core Runtime Option (`idr_core.dll` / `libidr_core.so`)

For cross-platform deployment (Android NDK + iOS Objective-C++), the entire mathematical core (Mount Calibrator, Causal Speed Smoother, 15-State ES-EKF, Road Matcher) compiles into a lightweight C++ library without heavy dependencies:
- **Language**: Standard C++17.
- **Dependencies**: Eigen3 (header-only linear algebra).
- **Binary Footprint**: < 1.2 MB compiled shared object (`.so` / `.dylib`).
- **Memory Consumption**: Fixed-allocation static buffers, zero runtime heap allocations in hot loop (< 4.5 MB resident set size).

---

## 5. Offline GIS Vector Road Network Ingestion

Vehicle dead reckoning in deep tunnels requires zero internet connectivity. Map geometry is stored locally in compact, pre-compiled spatial tiles.

### 5.1 Storage Format & Size
- **Format**: FlatGeobuf or binary serialized GeoJSON with spatial R-tree indexing.
- **Coverage**: Metropolitan road network (e.g. Pune / Mumbai highway corridors) occupies ~12 MB compressed.
- **Segment Schema**:
  - `segment_id` (uint32)
  - `geometry` (polyline coordinates in local ENU or relative WGS-84 micro-degrees)
  - `azimuth_deg` (0.0 to 360.0 degrees)
  - `speed_limit_mps` (float)
  - `road_type` (Highway, Arterial, Tunnel, Ramp)
  - `one_way` (bool)

### 5.2 Dynamic Tile Loading & Spatial Query
- The engine queries the local spatial index using a bounding box of radius 500m centered on the vehicle's last known GNSS coordinate.
- Active segments are cached in memory (< 200 road segments in active cache).
- Candidate segments within 15.0m cross-track distance and heading discrepancy < 65 degrees are evaluated at 10 Hz.

---

## 6. Power, Battery, & Thermal Budgeting

Smartphone background navigation must operate continuously without triggering Android OS thermal throttling or excessive battery drain.

| Subsystem | Operational Target | Design Mitigation |
| :--- | :--- | :--- |
| **Battery Drain** | **< 2.8% per hour** | Sensor batching (`maxReportLatencyUs = 20000`), no wake-lock abuse, NEON SIMD vectorization. |
| **CPU Occupancy** | **< 4.0% of single core** | 15-state EKF analytical Jacobians, Joseph-form covariance updates, decimation of heavy AI inferences to 10 Hz. |
| **Memory Footprint** | **< 15 MB RAM** | Fixed circular buffers for features, static matrix allocations, lightweight TorchScript Mobile runtime. |
| **Thermal Profile** | **Cool (< 36 deg C)** | Zero GPU/NPU utilization required; standard low-power CPU cores (LITTLE cluster) handle entire workload. |
| **Wake-Lock Lifecycle**| **PARTIAL_WAKE_LOCK** | Acquired strictly when vehicle is moving (`speed > 1.5 m/s`); automatically released after 3 minutes of rest. |

---

## 7. Android Foreground Service & Permissions

### 7.1 Required Android Manifest Declarations
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- Sensors & Location -->
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
    <uses-permission android:name="android.permission.HIGH_SAMPLING_RATE_SENSORS" />
    
    <!-- Background Execution -->
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_LOCATION" />
    <uses-permission android:name="android.permission.WAKE_LOCK" />

    <application>
        <service
            android:name=".service.DeadReckoningService"
            android:foregroundServiceType="location"
            android:exported="false" />
    </application>
</manifest>
```

### 7.2 Service Lifecycle Protocol
1. **Startup**: When user starts navigation, `DeadReckoningService` starts as a Foreground Service with an ongoing notification ("Dead Reckoning Navigation Active").
2. **Calibration Phase**: The phone listens for 5 to 10 seconds of normal driving. As soon as forward acceleration correlates with GNSS speed, the 3D mount alignment quaternion is locked.
3. **GNSS Outage Event**: When GNSS accuracy exceeds 8.0m or fixes stop arriving, the service transitions seamlessly from `GNSS_AIDED` to `INS_ONLY_BLACKOUT` mode without user interruption.
4. **Handoff & Recovery**: When emerging from a tunnel, the GNSS Handoff Manager smoothly blends GNSS fixes back into the EKF over a 3.0 second exponential ramp to prevent coordinate teleportation jumps.
5. **Shutdown**: When navigation ends or vehicle is parked, sensors are unregistered and wake-locks are released.
