# Smart Dead Reckoning (IDR) Mobile App: Vehicular Drive Test Protocol

## 1. Scope & Objective
This document outlines the standard operational procedure (SOP) for conducting empirical in-vehicle drive tests using the production on-device Android application (`com.recursiveminds.idr.ondevice`) running on a commercial smartphone (Samsung Galaxy F12 / SM-F127G, Android 13, Exynos 850, 4 GB RAM).

The primary objective is to validate real-time dead-reckoning accuracy, on-device causal inference throughput, battery consumption, and seamless GNSS handoff under real Indian transit conditions across the three operational tiers:
- **Tier 1 (Traffic Crawl)**: Speed < 20 km/h, distance < 200 m (frequent stops, idle vibration, bumper-to-bumper congestion).
- **Tier 2 (City Maneuvers)**: Speed 20-50 km/h, distance 200-500 m (frequent 90-degree junction turns, flyover ramps, multi-lane splits).
- **Tier 3 (Highway Cruising)**: Speed > 50 km/h, distance 500 m - 1.2 km (high-speed straightaways, long tunnels, sweeping curves).

Target benchmark criterion: **Total Dead-Reckoning Position Drift < 10% of total distance travelled** during complete GNSS blackout (< 5 m over 50 m, or < 100 m over 1 km).

---

## 2. Test Hardware & Software Configuration

| Component | Specification | Description / Role |
| :--- | :--- | :--- |
| **Test Device** | Samsung Galaxy F12 (`SM-F127G`) | Target low-cost commercial phone (Exynos 850, 8x Cortex-A55, 4 GB RAM) |
| **Operating System** | Android 13 (One UI Core 5.1) | Production unrooted stock firmware |
| **Application Package** | `com.recursiveminds.idr.ondevice` | Standalone on-device APK (Chaquopy + TFLite runtime, ~42 MB) |
| **Sensors Used** | Hardware Accelerometer + Gyroscope + GNSS | 10-50 Hz IMU (triaxial accel + gyro), 1 Hz fused GNSS location |
| **Mounting Fixture** | Rigid Windshield or Dashboard Phone Cradle | Portrait orientation, screen visible to driver/evaluator |
| **Test Vehicle** | Passenger Car / Auto-Rickshaw / Bus | Standard consumer vehicle driven in normal traffic |
| **Ground Truth Reference** | Pre-outage & Post-outage GNSS / Surveyed Landmarks | Ground truth coordinates at tunnel portals / exit points |

---

## 3. Pre-Drive Preparation Checklist

Before starting the test vehicle:

1. **Battery Level**: Verify phone battery charge is >= 50%.
2. **Mount Rigidity**: Secure the smartphone firmly into the dashboard/windshield mount. Ensure no loose rattling or swinging that would inject non-vehicular mechanical vibrations.
3. **Network & Cache Verification**:
   - Ensure mobile data or Wi-Fi is active.
   - Launch the application: `IDR On-Device`.
   - On the map screen, tap **PREFETCH AREA** to ingest OpenStreetMap (OSM) road networks for a 3 km radius around the test area into the local disk cache (`files/maps_cache/`).
   - Verify that the toast and button indicate: `Prefetched N segments` and `MAP MATCH: ON`.
4. **Offline Resilience Check (Optional Pre-Flight)**:
   - Toggle **Airplane Mode** ON to verify the system functions 100% offline from the local spatial cache.
   - Verify that road geometries display properly without active cellular data.
5. **Sensor Stream Verification**:
   - Verify that the HUD shows live IMU rate (~10-50 Hz) and GNSS rate (~1 Hz).
   - Verify that `WARMING_UP` status is displayed.

---

## 4. Operational Drive Test Workflow

### Phase 1: Mount Calibration & Yaw Lock (Warm-Up)
1. Begin driving normally in open-sky conditions.
2. The system executes automated 3D gravity leveling (Rodrigues rotation) and centripetal acceleration correlation to resolve the phone's 3D orientation relative to the vehicle chassis.
3. Perform standard driving maneuvers (straight cruising + 1-2 standard street turns).
4. Observe the HUD: status changes from `WARMING_UP (UNLEVELLED)` -> `LEVELLED` -> `YAW_LOCKED`.
5. Once `YAW_LOCKED` appears with a green badge, the system is fully calibrated and ready for blackout evaluation.

### Phase 2: Open-Sky Baseline Driving
1. Continue driving in normal traffic.
2. The system operates in `GNSS_HEALTHY` mode, logging raw sensor batches and GNSS fixes to the internal session memory and optional CSV log.
3. Speed estimates, heading, and road segment snapped coordinates are continuously tracked with sub-meter accuracy.

### Phase 3: Simulated or Natural GNSS Outage (Blackout)
1. **Natural Outage**: Drive into a covered tunnel, underpass, basement parking structure, or high-rise urban canyon.
2. **Simulated Outage**: If conducting open-road testing without a tunnel, press the **START BLACKOUT** button on the HUD (or tap the floating action button).
3. The system transitions immediately to `INS_DEAD_RECKONING`:
   - Visual indicator displays a red **BLACKOUT (INS ONLY)** badge.
   - GNSS updates are quarantined behind the zero-leak firewall.
   - Forward speed is inferred causally by the on-device Bayesian MoE neural model at 10 Hz.
   - 15-state Error-State Kalman Filter (ES-EKF) integrates orientation on SO(3) with Rate-Adaptive Non-Holonomic Constraints (NHC) and Lorentzian turn damping.
   - Topological Map Matcher snaps positions along candidate road corridors with curvature-governed branch gating.
   - During stops (red lights, traffic congestion), Physical Rest ZUPT clamps forward velocity to exactly 0.0 m/s and ZARU halts gyro bias drift.

### Phase 4: GNSS Re-Acquisition & Seamless Handoff
1. Emerge from the tunnel or press **STOP BLACKOUT**.
2. First GNSS fixes received at the tunnel portal are subjected to Chi-Square Normalized Innovation Squared (NIS) gating:
   - Multipath spikes and degraded fixes (accuracy > 30 m) are rejected.
   - Plausible fixes trigger transition to `REACQUISITION_VERIFY`.
3. Upon 3 consecutive consistent fixes, the system enters `REACQUISITION_BLENDING`:
   - Position and velocity reconciliation executes via C^2 cubic Hermite smoothstep over 3.0 seconds.
   - No visual snapping, teleports, or heading jumps occur on the navigation HUD.
4. The system returns to `GNSS_HEALTHY` and displays the final evaluation summary card.

---

## 5. Empirical Metrics & Data Logging

Upon blackout completion, the HUD automatically presents the quantitative scorecard:

```text
+--------------------------------------------------------------+
|                BLACKOUT EVALUATION SUMMARY                   |
+--------------------------------------------------------------+
| Outage Duration     : 60.0 s                                 |
| Distance Travelled  : 475.1 m                                |
| Final Position Error: 16.28 m                                |
| Total Drift %       : 3.43 %  [PASS: Target < 10.0%]         |
| Along-Track Error   : +12.4 m (speed scale error)            |
| Cross-Track Error   : -10.5 m (turn/heading error)           |
| Speed Regime        : Tier 2 (City Maneuvers, 28.5 km/h avg) |
| Max Real-Time Lag   : 14.2 ms / 100 ms batch (0 backlog)     |
+--------------------------------------------------------------+
```

### Telemetry Export
1. CSV telemetry logs are saved to:
   `/sdcard/Android/data/com.recursiveminds.idr.ondevice/files/logs/`
2. Each run exports:
   - `imu_stream_<timestamp>.csv`: Raw accelerometer and gyroscope data (timestamp_ns, ax, ay, az, gx, gy, gz).
   - `gnss_stream_<timestamp>.csv`: Raw GNSS fixes (timestamp_ns, lat, lon, alt, speed_mps, bearing_deg, accuracy_m).
   - `fused_trajectory_<timestamp>.csv`: Real-time fused and map-matched trajectory (timestamp_ns, lat, lon, speed_mps, heading_deg, mode, segment_id).
   - `session_summary_<timestamp>.json`: Complete quantitative scorecard, drift %, along/cross track errors, and device diagnostics.

---

## 6. Resource Consumption Profiling Protocol

To measure real-world device sustainability during live driving:

1. **Unplugged Battery Drainage**:
   - Disconnect USB cable during the 15-minute live drive run.
   - Measure battery drop (%) and discharge current (mAh) via `adb shell dumpsys batterystats`.
   - Target: Battery drainage < 1.0% per 15 minutes (< 4.0% per hour).
2. **CPU Utilization**:
   - Sample CPU utilization at 1 Hz via `adb shell top -b -n 1 | grep com.recursiveminds.idr`.
   - Target: Sustained CPU usage < 15% across all 8 cores.
3. **RAM Memory Footprint (PSS)**:
   - Sample Proportional Set Size (PSS) at 0, 5, 10, and 15 minutes via `adb shell dumpsys meminfo com.recursiveminds.idr.ondevice`.
   - Target: Total PSS < 250 MB; zero memory leaks across sustained 15-minute runs.
4. **Thermal Stability**:
   - Query device skin and battery temperatures before and after the test via `adb shell dumpsys thermalservice`.
   - Target: Temperature rise < 4.0 degrees C; zero thermal throttling events.

---

## 7. Pass / Fail Acceptance Criteria

A drive test run is classified as **SUCCESSFUL (PASS)** if and only if all of the following conditions are met:

1. **Drift Accuracy**: Overall position drift during GNSS blackout is strictly less than 10.0% of total blackout distance.
2. **Causal Real-Time Execution**: No sensor queue drop or batch backlog accumulation (> 200 ms). Mean processing latency per 100 ms batch is strictly < 25 ms.
3. **Zero Crash Rate**: Zero unhandled exceptions, ANRs (Application Not Responding), or process crashes throughout the entire drive.
4. **Map-Matching Stability**: Vehicle trajectory snaps cleanly to the valid road centerline without oscillation or branch flipping.
5. **Handoff Continuity**: Re-acquisition blending smoothly reconciles dead reckoning with GNSS ground truth with C^2 continuity and zero visual position jumps.
