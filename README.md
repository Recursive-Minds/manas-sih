# Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-40%2F40%20Passing-brightgreen.svg)](tests/)
[![SIH Target](https://img.shields.io/badge/SIH%20Target-%3C%2010%25%20Drift-orange.svg)](FINAL_JUDGE_EVALUATION_REPORT.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Smart India Hackathon (SIH 26168)**: Edge-deployable automotive navigation engine running entirely on low-cost consumer smartphone sensors (10 Hz IMU + 1 Hz GNSS). Maintains continuous, sub-lane vehicular localization during prolonged satellite outages (tunnels, urban canyons, dense canopies, underpasses) with **zero vehicle CAN-bus or OBD-II wiring**.

---

## Authoritative Documentation Suite (Single Source of Truth)

To eliminate contradictory metrics across disparate files, the entire project documentation is consolidated into **3 authoritative master documents**:

| Master Document | Scope & Purpose | Link |
| :--- | :--- | :--- |
| **1. System Implementation & Architecture** | **Living Technical Reference**: Mathematical formulations, dynamic coordinate frames, SO(3) 3D leveling, Kinematic Delta-v Speed Observer, multi-source heading fusion, Repaired Road Network Governor, 15-state ES-EKF, active parameters registry, edge C++ NDK engine, and complete codebase inventory. *Updated whenever code, algorithms, or parameters change.* | [SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md](SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md) |
| **2. Problem Statement & Initial Plan** | **Foundational Specification & Evolution**: SIH 26168 challenge definition, Indian transit constraints, 3 operational tiers, initial 5-phase roadmap, key scientific discoveries (why neural heading failed, 9s GPS optical illusion), 20 physical failure modes & hardening record, and production readiness matrix. | [PROBLEM_STATEMENT_AND_INITIAL_PLAN.md](PROBLEM_STATEMENT_AND_INITIAL_PLAN.md) |
| **3. Final Judge Evaluation Report** | **Definitive Empirical Benchmarks**: Multi-seed statistical validation (6 random seeds × 40 scenarios = 240 evaluation runs), official SIH multi-tier scorecard, out-of-sample domain breakdowns (Highway, Urban Grid, Arterial), error decomposition (cross-track vs. along-track), and scenario trajectory plots. *Updated whenever benchmarks are executed.* | [FINAL_JUDGE_EVALUATION_REPORT.md](FINAL_JUDGE_EVALUATION_REPORT.md) |

*(Developer and agent rules are maintained in [CLAUDE.md](CLAUDE.md) and [GEMINI.md](GEMINI.md)).*

---

## 1. Problem Statement in Brief

### The Operational Challenge
In multi-level flyovers, tunnels, dense tree cover, and urban concrete canyons, smartphones experience severe **GNSS signal degradation or total blackout**:
* Standard mobile navigation apps freeze, extrapolate straight into buildings, or suffer chaotic 50–100 m position jumps upon satellite reacquisition.
* Tactical-grade pre-aligned Inertial Navigation Systems (costing > $10,000) and wheel encoders wired via OBD-II/CAN bus are standard in autonomous testbeds but **absent in consumer vehicles**.
* Over 95% of vehicles in India (two-wheelers, auto-rickshaws, commercial trucks, commuter cars) rely entirely on consumer smartphones placed in arbitrary mounts (portrait, landscape, tilted) with high thermal bias and chassis vibration.

### SIH Operational Tiers & Target Metrics
The official challenge target is **overall dead-reckoning drift < 10% of total distance travelled** during complete GNSS blackouts (< 5m over 50m, or < 100m over 1km).
* **Tier 1: Traffic Crawl (< 20 km/h, < 200m)**: Red-light idling and stop-and-go congestion. Target: position error < 10m.
* **Tier 2: City Maneuvers (20–50 km/h, 200m–500m)**: 90° intersection turns and roundabouts. Target: drift < 15% (sub-lane).
* **Tier 3: Highway Cruising (> 50 km/h, 500m–1.2km)**: High-speed tunnel transits at 60–100 km/h. Target: drift < 10% (< 100m over 1km).

*(For complete empirical scorecards across all 240 evaluation runs, see [FINAL_JUDGE_EVALUATION_REPORT.md](FINAL_JUDGE_EVALUATION_REPORT.md)).*

---

## 2. End-to-End System Pipeline

```
[Raw Smartphone IMU] (100 Hz / 10 Hz)
       │
       ▼
[Stage 1: Mount Auto-Calibrator] ──► SO(3) Gravity Leveling & Dynamic Turn Correlation
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[Stage 2: Neural Speed & Delta-v Observer]   [Stage 3: Physical Rest Detector (ZUPT)]
  - Bayesian Mixture-of-Experts (ResNet+TCN)   - Accel Variance: Var(a) < 0.04 m^2/s^4
  - 10 Hz Kinematic Accel Integration          - Zero-Velocity Update (ZUPT)
  - Zero Causal Window Lag (< 1 ms)            - Zero Angular Rate Update (ZARU)
       │                                          │
       ▼                                          ▼
  Forward Velocity Envelope & Variance       Stationary / Driving State
       │                                          │
       └────────────────────┬─────────────────────┘
                            ▼
           [Stage 4: Dynamic Multi-Source Heading Engine]
             - Gyro Yaw Integration (w_z_corr = raw_gyro[2] - b_g)
             - Centripetal Lateral Accel Cross-Validation: a_lat = v * w_z
             - Speed-Regime GPS Vector Initial Seeder (< 0.15° error)
                            │
                            ▼
           [Stage 5: 15-State Error-State EKF]
             - SO(3) Quaternion Error State
             - Closed-Loop Non-Holonomic Constraints (v_lat = 0, v_up = 0)
             - Lorentzian Turn-Damped Gyro Bias Adaptation
                            │
                            ▼
                 Continuous Fused Position
                            │
                            ▼
           [Stage 6: Repaired Road Network Governor]
             - Curvature Kinematics Governor: v <= sqrt(a_lat_max / kappa)
             - Turn-Inflated Gaussian Emission Likelihood (sigma >= 45°)
             - Successor Topology Graph & Anti-Boundary Pinning Watchdog
                            │
                            ▼
                 Matched Road Trajectory
```

---

## 3. Quickstart & Reproduction Guide

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* Git

### Installation
```bash
git clone https://github.com/Recursive-Minds/manas-sih.git
cd manas-sih
pip install -r requirements.txt
```

### Running Unit Tests (40/40 Passing)
```bash
pytest tests/ -v
```

### Running the Standardized Benchmark
By default, the benchmark runner executes across **6 random seeds** (generating 240 randomized blackout scenarios across 5 diverse test sequences):
```bash
# Full multi-seed benchmark (6 seeds x 40 scenarios = 240 runs)
python benchmarks/run_final_benchmark.py

# Fixed reproducible canonical benchmark
python benchmarks/run_final_benchmark.py --fixed

# Quick single-seed benchmark (40 scenarios)
python benchmarks/run_final_benchmark.py --single
```

### Training the Velocity Estimator
```bash
# Train the Dual-Brain Bayesian Mixture-of-Experts with CAN-bus ground truth
python train_velocity_model.py --epochs 30 --batch_size 128
```

---

## 4. Repository Structure

```
SIH/
├── README.md                                  <-- Master navigation entry point
├── SYSTEM_IMPLEMENTATION_AND_ARCHITECTURE.md  <-- [1] Complete Math, Algorithms & Technical Specs
├── PROBLEM_STATEMENT_AND_INITIAL_PLAN.md      <-- [2] Initial Plan, Roadmap, PS & Failure Modes
├── FINAL_JUDGE_EVALUATION_REPORT.md           <-- [3] Definitive Empirical Benchmark Evaluation
├── FINAL_JUDGE_EVALUATION_REPORT.html         <-- Self-contained interactive report with charts
├── CLAUDE.md                                  <-- AI Developer guidelines & architectural rules
├── GEMINI.md                                  <-- Project memory & synchronization protocol
├── benchmarks/
│   ├── run_final_benchmark.py                 <-- Standardized 40-scenario benchmark orchestrator
│   └── run_multi_seed_evaluation.py           <-- Multi-seed statistical evaluator
├── engine/
│   └── cpp/                                   <-- Zero-dependency 200 Hz embedded C++ NDK engine
├── models/
│   ├── checkpoints/                           <-- Trained PyTorch weights (.pt)
│   └── exported/                              <-- Optimized TorchScript mobile models (.pt)
├── sih/                                       <-- Production Algorithmic Core
│   ├── calibration/                           <-- Dynamic 3D mount leveling & seeder
│   ├── core/                                  <-- Immutable contracts & decoupled interfaces
│   ├── data/                                  <-- Schema-flexible loader & geodetic conversions
│   ├── engine/                                <-- Dead reckoning engine & Delta-v speed observer
│   ├── eval/                                  <-- Outage simulators & error metric computation
│   ├── fusion/                                <-- 15-state ES-EKF & speed smoother
│   ├── handoff/                               <-- 6-state GNSS-INS handoff FSM & Hermite reconciliation
│   ├── map/                                   <-- Topological graph, OSM client & road governor
│   ├── mobile/                                <-- Causal real-time mobile streaming pipeline
│   └── models/                                <-- Bayesian MoE, ResNet-1D, TCN-Attention
└── tests/                                     <-- 40 passing unit test suites
```

---

## 5. Scientific Integrity & No-Fabrication Standard

All performance claims in this repository are backed by inspectable, reproducible code executed against real driving datasets (`S-M.csv`, `S-S2.csv`, `S-S1.csv`, `S-S3a.csv`, `S-S4.csv`) with 10 Hz vehicle CAN-bus wheel speed validation.

To review the complete empirical evaluation tables, domain cross-validation distributions, trajectory error decompositions, and individual blackout trajectory maps, consult:
👉 **[FINAL_JUDGE_EVALUATION_REPORT.md](FINAL_JUDGE_EVALUATION_REPORT.md)**
