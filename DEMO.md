# Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion: Live Demonstration Guide

This guide provides step-by-step instructions for running the real-time demonstration of the Smartphone IDR navigation pipeline (SIH Problem Statement 26168).

---

## 1. Demonstration Modes Overview

The system supports three distinct operational demonstration modes:

| Mode | Target Setup | Description |
| :--- | :--- | :--- |
| **Mode 1: USB Scenario Replay (Zero Driving Required)** | Phone connected to PC via USB cable | Replays canonical IO-VNBD benchmark scenarios from the server directly to the smartphone screen, showing the live moving blue dot, real-time map matching, and HUD metric calculations. |
| **Mode 2: Live In-Vehicle Drive (Real Sensors)** | Phone mounted in vehicle cradle | Streams live 50 Hz phone IMU and 1 Hz GNSS data over Wi-Fi/tethering to the dead-reckoning engine, performing real-time cradle calibration, speed inference, and blackout navigation. |
| **Mode 3: Standalone Offline Logging** | Phone operating autonomously (no server) | App logs raw IMU and GNSS fixes to CSV continuously. The user exports logs via Android FileProvider and replays them offline using the server replay CLI. |

---

## 2. Prerequisites & System Setup

### 2.1 Hardware Requirements
- Host PC running Windows, Linux, or macOS.
- Android smartphone (Android 9.0+ / API 28+) with USB debugging enabled.
- USB-C data cable.

### 2.2 Software Environment
1. **Python Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **PyTorch MoE Checkpoint**: Verify the velocity model exists at:
   [models/checkpoints/round1_interval_lam0.5_s42.pt](file:///c:/Users/carpe/SIH/models/checkpoints/round1_interval_lam0.5_s42.pt) (production model, with `best_moe_velocity_model.pt` retained as pre-round-1 backup).
3. **Android Platform Tools (ADB)**:
   Ensure ADB is accessible. Default path on Windows:
   `C:\Users\<username>\AppData\Local\Android\Sdk\platform-tools\adb.exe`
4. **Android APK Installation**:
   Build and install the debug APK onto the device:
   ```powershell
   cd android
   .\gradlew.bat installDebug
   cd ..
   ```
   Or install the pre-built binary:
   ```powershell
   adb install android/app/build/outputs/apk/debug/app-debug.apk
   ```

---

## 3. Demo Mode 1: USB Scenario Replay (Recommended for Jury Evaluation)

This mode allows judges to evaluate the full streaming pipeline and UI responsiveness without driving a car.

### Step 1: Forward WebSocket Port over USB
Connect the Android phone to your PC with USB debugging enabled, then forward port 8765:
```powershell
adb reverse tcp:8765 tcp:8765
```
*(If testing on the PC browser without a phone, navigate to `http://localhost:8765/view`)*.

### Step 2: Launch the Streaming Server
Start the high-performance WebSocket server with the production Stage B dead-reckoning engine:
```powershell
python -m server.router --host 0.0.0.0 --port 8765
```
The server binds to:
- **Phone Ingestion & HUD**: `ws://localhost:8765/ws/stream`
- **Browser Web Dashboard**: `http://localhost:8765/view`
- **Browser HUD Channel**: `ws://localhost:8765/ws/client`

### Step 3: Connect the Phone App
1. Open the **Smart IDR** app on the Android phone.
2. In the header bar, set:
   - **Server**: `localhost` (when using `adb reverse`) or your PC's Wi-Fi LAN IP (e.g. `192.168.1.5`).
   - **Port**: `8765`.
3. Tap **CONNECT**. The connection badge will turn green: `CONNECTED`.

### Step 4: Stream a Short Benchmark Scenario
In a separate terminal on the PC, replay Scenario #30 (or Scenario #22) from the canonical `S-S3a` dataset:
```powershell
python -m server.replay --trip S-S3a --scenario 30 --speed 1.0
```

Available replay arguments:
- `--scenario 30`: Short 244m mixed-domain scenario (35s warm-up + 60s blackout).
- `--scenario 22`: 475m arterial scenario (35s warm-up + 45s blackout).
- `--scenario 26`: Key 892m benchmark scenario (35s warm-up + 75s blackout).
- `--speed 1.0`: Real-time 1.0x playback (use `--speed 2.0` for 2x or `--speed 0.0` for max CPU throughput).
- `--warmup 35.0`: Duration of pre-blackout driving to establish mount lock and speed scaling.

### Step 5: Observe Live Behavior on Phone & Web Dashboard
During playback, observe the following stages in real time:

1. **Warm-Up Phase (0s to 35s)**:
   - The vehicle marker (blue dot) begins driving through Coventry streets.
   - The **Warm-Up Panel** displays active convergence status:
     - `✓ Gravity Leveling` (30/30 samples converged)
     - `✓ Mount Alignment` (turns: 8/8 or Mount: reused)
     - `✓ AI Feature Buffer` (60 samples / 6.0s warm)
     - `✓ Alpha Seeding` (>= 3 moving GNSS fixes)
   - The headline changes from `WAITING FOR READY` to `READY TO START`.
   - The `START` button turns emerald green (`#10B981`).

2. **Blackout Phase (35s to 95s)**:
   - The server triggers the GNSS blackout firewall (all GNSS data is strictly withheld from the engine).
   - The HUD status badge changes to rose red: `BLACKOUT`.
   - The blue dot dead-reckons smoothly along the road network using the Causal MoE velocity estimator, 15-state ES-EKF, and topological map matcher.
   - Live metrics update continuously at 10 Hz:
     - **Drift %**: Stays well within the SIH target `< 10%`.
     - **Along-Track / Cross-Track Error**: Decomposed position error.
     - **Speed & Heading**: Live comparisons between Dead Reckoning and Ground Truth.

3. **Stop & Summary Inspection**:
   - When the scenario completes (or the user taps `STOP`), the **Session Summary Card** appears, reporting:
     - **Final Error**: e.g. `7.04 m` (Scenario #30)
     - **Drift %**: e.g. `2.88% (target < 10%)`
     - **Speed Regime**: `City (20-50 km/h)`
     - **Duration & Distance**: `60.0 s / 244.2 m`
     - **Along-Track & Cross-Track Error**: `-6.9 m / 1.3 m`

---

## 4. Demo Mode 2: In-Vehicle Live Drive Mode (Real Smartphone Sensors)

For field evaluation in a real moving automobile:

1. **Mount Phone**: Secure the smartphone into a car cradle or dashboard mount.
2. **Network Connection**: Connect the phone and laptop to the same Wi-Fi hotspot (or use USB tethering).
3. **Start Router**: On the laptop:
   ```powershell
   python -m server.router --host 0.0.0.0 --port 8765
   ```
4. **Connect App**: In the phone app, enter the laptop's LAN IP address and tap **CONNECT**.
5. **Calibration Drive**: Drive normally for 30 to 60 seconds through 2 to 3 street corners.
   - Gravity levels within 3 seconds.
   - Mount yaw calibrator detects turns and locks cradle alignment.
   - The warm-up indicators will turn green.
6. **Trigger Blackout**:
   - Tap the green **START** button when entering a tunnel, underpass, or parking structure.
   - The app initiates dead reckoning.
   - Tap **STOP** upon exiting the tunnel to view the Session Summary Card.

---

## 5. Demo Mode 3: Standalone Offline Logging & Offline Replay

When driving without a server present:

1. **Standalone Logging**:
   - Open the app (no connection to server required).
   - The app's background service (`SensorStreamService`) continuously records 50 Hz IMU and 1 Hz GNSS fixes into a local CSV file: `idr_telemetry_<timestamp>.csv`.
   - The map displays real-time GNSS location.
2. **Export Telemetry**:
   - Tap the **Share CSV** button in the app header to export the CSV file via Gmail, Google Drive, or USB.
3. **Offline Replay**:
   - Copy the CSV file to the host PC and replay it through the server pipeline:
     ```powershell
     python -m server.replay --trip path/to/idr_telemetry_xxx.csv --blackout-start 60.0 --blackout-duration 45.0 --speed 1.0
     ```

---

## 6. Algorithmic Parity & Unit Test Verification

To mathematically prove zero future leakage and confirm parity between the batch research benchmark and the real-time streaming engine:

### 6.1 Bit-Level Engine Parity (Override Mode)
Confirms 0.0000 m bitwise identity between batch and streaming ES-EKF / Map Matching:
```powershell
python scripts/quick_parity.py
```
*Expected Result*: Endpoint difference = `0.0000 m` across all 5 canonical scenarios.

### 6.2 Causality & Leak-Free Test Suite
Validates strict temporal causality, future independence, and post-blackout NaN-injection invariance:
```powershell
pytest tests/test_causal_streaming.py tests/test_no_future_leak.py -v
```
*Expected Result*: `5 passed in ~5 minutes` (100% success rate).

---

## 7. Architecture Summary & Code References

- **Streaming Server Router**: [server/router.py](file:///c:/Users/carpe/SIH/server/router.py)
- **Production Streaming Adapter (Stage B)**: [server/engine_adapter.py](file:///c:/Users/carpe/SIH/server/engine_adapter.py)
- **Scenario Replay CLI**: [server/replay.py](file:///c:/Users/carpe/SIH/server/replay.py)
- **Web Dashboard**: [server/view/index.html](file:///c:/Users/carpe/SIH/server/view/index.html)
- **Android App Entry Point**: [android/app/src/main/java/com/recursiveminds/idr/ui/MainActivity.kt](file:///c:/Users/carpe/SIH/android/app/src/main/java/com/recursiveminds/idr/ui/MainActivity.kt)
- **Android Background Sensor Service**: [android/app/src/main/java/com/recursiveminds/idr/service/SensorStreamService.kt](file:///c:/Users/carpe/SIH/android/app/src/main/java/com/recursiveminds/idr/service/SensorStreamService.kt)
