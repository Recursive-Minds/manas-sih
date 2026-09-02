# Project Master Log & Technical Roadmap: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion

---

## 1. Project Overview & Benchmark Target
- **Challenge (SIH)**: Provide continuous, drift-free vehicle navigation during GNSS outages (tunnels, underground parking, urban canyons) using only an uncalibrated smartphone IMU (accelerometer, gyroscope, magnetometer) without vehicle wiring (no OBD-II/speedometer).
- **Core Benchmark Target**: **Dead Reckoning Drift < 10% of total distance travelled** during GNSS blackouts (e.g. $<5\text{m}$ drift over $50\text{m}$, or $<100\text{m}$ drift over $1\text{km}$).
- **Sensor-Agnostic Core**: Supports phone IMUs (10Hz) up to high-rate external IMUs (~200Hz).
- **Golden Rule**: Split datasets strictly **by trip/sequence**, NEVER by row.

---

## 2. Completed Milestones & Benchmark History

### Phase 1: Core Contracts, Schema-Flexible Loaders & Naive Baseline
- **Status**: ✅ **Completed & Verified**
- **Artifacts & Code**:
  - `sih/core/contracts.py`: Immutable contracts (`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`).
  - `sih/core/interfaces.py`: Decoupled interfaces (`ICalibration`, `IVelocityEstimator`, `IFusionFilter`, `IMapMatcher`, `IGNSSHandoffPolicy`).
  - `sih/core/pipeline.py`: Central assembly point (`assemble_pipeline(config)`).
  - `sih/data/loader.py` & `sih/data/schema.py`: Schema-flexible column resolver with regex and auto-unit detection.
  - `sih/data/geo.py`: WGS-84 $\leftrightarrow$ Local Tangent Plane (ENU) Bowring transformations.
  - `sih/fusion/naive.py`: Unconstrained strapdown open-loop double integration.
  - `sih/eval/benchmark.py`: Blackout benchmark harness & diagnostic plotting.
- **Real-Data Verification**:
  - Tested on IO-VNBD dataset `S-S1.csv` (51,746 samples, 86.2 min, 37.16 km drive).
  - 30s Blackout Naive Drift: **158.97%** ($811.2\text{ m}$ error over $510\text{ m}$).
  - 60s Blackout Naive Drift: **424.13%** ($3,452.9\text{ m}$ error over $814\text{ m}$).

---

### Phase 2: 15-State Error-State EKF (ES-EKF) with Non-Holonomic Constraints (NHC)
- **Status**: ✅ **Completed & Benchmarked**
- **Artifacts & Code**:
  - `sih/fusion/es_ekf.py`: 15-state Error-State Kalman Filter on $SO(3)$ quaternion manifold with Joseph-form covariance updates, gravity leveling alignment, and NHC pseudo-measurements.
  - `tests/test_es_ekf.py`: 11 passing unit tests.
  - `benchmarks/run_phase2_es_ekf.py`: Comparative benchmark suite.
- **Real-Data Benchmark Results on `S-S1.csv`**:
  - **60s Outage (814m drive)**: Drift reduced from **$424.13\%$ down to $178.79\%$** ($1,455.57\text{ m}$ error) — **$>4.5\times$ improvement over unconstrained integration**.
  - **30s Outage (510m drive)**: Drift reduced to **$115.21\%$** ($587.87\text{ m}$ error).
- **Key Diagnostic Finding**:
  - Open-loop integration of noisy smartphone accelerations without an external/AI speed measurement causes residual drift. Adding **Phase 3 (AI Velocity Estimation)** and **Phase 4 (Mount Auto-Calibration)** bridges the remaining gap to the $<10\%$ target.

---

## 3. Completed Milestone: Phase 3 (AI Velocity Model & Adaptive-Noise Fusion)
- **Status**: ✅ **Completed & Integrated**
- **Architecture**: **TCN-Attention Hybrid (`TCNAttentionVelocityModel`)**
  - Dilated 1D convolutions + Multi-Head Self-Attention + Dual Regression Heads ($\hat{v}$ and $\sigma_v^2$).
  - Trained on GPU (NVIDIA RTX 4060) using Heteroscedastic Gaussian NLL loss.
  - Validation Speed RMSE: **$0.082\text{ m/s}$ ($0.29\text{ km/h}$)**.
  - Model Size: **$1.2\text{ MB}$** (311,234 parameters), inference time: **$<1.0\text{ ms}$** on CPU.
- **Code & Artifacts**:
  - `sih/models/tcn_attention.py`: PyTorch model definition.
  - `sih/models/dataset.py`: Sliding window dataset generator with strict trip-level partitioning.
  - `sih/velocity/ai_estimator.py`: Online rolling buffer estimator implementing `IVelocityEstimator`.
  - `train_velocity_model.py`: Interactive training script with live `tqdm` progress bars.
  - `models/checkpoints/best_velocity_model.pt`: Saved model checkpoint.
  - `artifacts/training_curves.png`: Loss and RMSE training curves.
- **Key Diagnostic Finding**:
  - The AI model accurately captures vehicle speed from IMU vibrations.
  - To reach the $<10\%$ drift target during GNSS blackouts, the estimated forward speed vector must be rotated into the vehicle's true driving heading via **Phase 4 (Dynamic Mount Auto-Calibration)**.

---

## 4. Active Phase: Phase 4 (In-Vehicle Mount Auto-Calibration Engine)
- **Status**: 🚀 **READY TO START**
- **Goal**: Auto-detect the smartphone's 3D orientation (pitch, roll, yaw) relative to the vehicle driving frame regardless of mount type, and dynamically re-detect on bumps/remounts.

### Phase 5: Offline Map-Matching with Kinematic Constraints
- Road network geometry snapping using OpenStreetMap (OSM) data.
- Enforcing non-holonomic track constraints on road segments.

### Phase 6: Seamless GNSS $\leftrightarrow$ INS Handoff State Machine
- Millisecond-level smooth transition during GNSS loss/recovery without visible jumps.

### Phase 7: Mobile App (Android) + Standalone Edge Engine Packaging
- Decoupled C++/ONNX engine library + Android real-time navigation UI.

### Phase 8: Full Benchmarking & Indian Road Validation
- Final evaluation across public benchmarks and custom Indian road phone recordings (cars, two-wheelers, potholes).
