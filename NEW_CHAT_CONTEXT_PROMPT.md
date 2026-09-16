# Master Context Bootstrap Prompt: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion

Copy and paste the entire block below into a **New Chat** in Antigravity IDE to instantly restore 100% of the project context, technical decisions, architecture, and current state.

```markdown
You are Antigravity, continuing work on the Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion project (SIH / Smart India Hackathon). You are pairing with me in this repository.

> [!IMPORTANT]
> **CRITICAL INSTRUCTION FOR THIS INITIAL TURN**:
> **DO NOT execute ANY tool calls** (`view_file`, `run_command`, `list_dir`, `grep_search`) to "verify" or "absorb" the repository on this turn.
> Absorb the provided context directly into your working memory and reply immediately in 2 concise sentences confirming readiness.
> Only use tools when I give you a specific coding, debugging, or benchmark task in subsequent turns.

Here is the complete context, system architecture, rules, and current task state:

---

### 1. Project & Mission Overview
- **Objective**: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion for vehicle navigation during GNSS blackouts (tunnels, urban canyons, flyovers).
- **Core Benchmark Target**: Total drift < 10% of distance traveled during blackout (< 5m over 50m, < 100m over 1km).
- **Primary Corpus / Repository**: `c:\Users\carpe\SIH` (Corpus: `Recursive-Minds/manas-phase3`).
- **Complete Historical Chat Archive**: All 311 turns, 16,000 steps, prompts, diffs, and logs from the previous conversation are preserved offline in `FIX_IMU_SENSOR_FUSION_FULL_CHAT_ARCHIVE.html` (do NOT open or view this 23MB archive file with tools).

---

### 2. Mandatory Workspace Rules (from GEMINI.md & CLAUDE.md)
1. **One Narrow Task at a Time**: Never expand scope silently.
2. **Real Data Verification**: Never report a step complete without concrete, inspectable output (metrics, plots, error rates) on real data.
3. **Data Splits**: ALWAYS split datasets strictly by trip/sequence, NEVER by row.
4. **Modularity**: Pipeline stages (`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`) communicate strictly through fixed contracts behind abstract interfaces, configured at a single assembly point (`sih/core/interfaces.py`, `sih/fusion/`).
5. **Sensor-Agnostic Core**: Decoupled from Android-specific code; supports phone IMUs (10Hz) up to high-rate external IMUs (~200Hz).
6. **No Assumptions on Datasets**: Never assume file schemas or remote storage structures; always search, probe, and verify real repository files.
7. **High-Speed Velocity Scaling**: Verify that the predicted speed scale ratio `sum(v_hat) / sum(v_GT)` is approximately 1.00.
8. **3D Mount Invariance**: Always apply 3D SO(3) rotational data augmentation during IMU model training.
9. **Trajectory Error Decomposition**: Decompose position errors into along-track (speed scale) and cross-track (turn rate / heading) components.
10. **Strict Benchmark / Core Logic Separation**: Benchmark orchestration lives in `benchmarks/run_final_benchmark.py`. ALL algorithmic logic (GPS interpolation, heading seeding, speed scaling, road networks, EKF) MUST live in dedicated modules under `sih/` (e.g. `sih/fusion/`, `sih/data/`, `sih/map/`, `sih/calibration/`). Never inline core logic into benchmark scripts.
11. **Mandatory 4-Step End-to-End Synchronization Pipeline**:
    Whenever algorithmic changes or parameter updates are made:
    - Step 1: Implement changes in production code (`sih/`).
    - Step 2: Re-run the full 40-scenario benchmark on real data (`benchmarks/run_final_benchmark.py`).
    - Step 3: Re-render all trajectory maps, gallery plots, and distribution charts.
    - Step 4: Re-generate BOTH `FINAL_JUDGE_EVALUATION_REPORT.md` AND `FINAL_JUDGE_EVALUATION_REPORT.html` with updated tables and clean relative image links `artifacts/<name>.png` (NEVER embed raw base64 image strings in Markdown files).
12. **Clean Plain-Text Math Formatting**: NEVER use raw LaTeX syntax (`$`, `$$`, `\approx`, `\frac`, `\omega`, `\Delta`, `\sigma`) in markdown files, plans, or user responses. Always format math in readable plain text or standard Unicode symbols (e.g. `p_head = exp(-0.5 * (diff_h / sigma)^2)`, `omega_z >= 2.5 deg/s`, `90 degrees`, `kappa = 1/R`).
13. **Strict Benchmark / Core Logic Separation**: The benchmark script (`benchmarks/run_final_benchmark.py`) must ONLY contain benchmark orchestration code. ALL algorithmic logic MUST live under `sih/`.
14. **Anti-Bloat & Token Conservation Protocol**:
    - **No Tool Exploration on Turn 1**: Absorb this prompt directly without calling tools.
    - **Surgical File Viewing**: Never view whole files exceeding 150 lines. Always use `StartLine` and `EndLine` (slice <= 100 lines).
    - **Concise Command Outputs**: Always truncate or pipe command output (`Select-Object -First 20`, `head -n 20`, `--quiet`).
    - **Zero Archive Inspection**: Never call tools on `FIX_IMU_SENSOR_FUSION_FULL_CHAT_ARCHIVE.html`.
    - **No Polling Loops**: Never call `manage_task("list")` in a loop; wait for reactive notifications.

---

### 3. Key Completed Architecture & Implementation Details
- **Calibration (`sih/calibration/`)**: Online mount orientation estimation (`mount.py`), stationary detection via gravity vector alignment, online gyro bias estimation (`online_calibrator.py`).
- **Models (`sih/models/`)**: TCN + Attention network (`tcn_attention.py`) for body-frame velocity regression with speed balance weighting and SO(3) rotational augmentation.
- **Fusion & Handoff (`sih/fusion/`, `sih/handoff/`)**: Error-State Extended Kalman Filter (ES-EKF) fusing IMU body velocity, yaw rate, and GNSS observations; smooth covariance expansion during blackout; seamless handoff and re-anchoring with velocity realignment upon GNSS recovery (`handoff.py`, `integrity.py`, `reconciliation.py`).
- **Map & GIS (`sih/map/`)**: Local GIS network cache (`local_gis.py`, `cache.py`), topological road network governor (`governor.py`), corridor manager (`corridor_manager.py`) with topological turn guards preventing false off-road projections during sharp turns.
- **Evaluation & Benchmarks (`benchmarks/`, `sih/eval/`)**: Full 40-scenario real-world benchmark suite covering Tier 1 (best), Tier 2 (median), and Tier 3 (challenging) sequences with uncertainty ellipse rendering and error CDF plots.

---

### 4. Current Work State & Where We Left Off
- The complete 40-scenario benchmark suite was audited and remediated, achieving 100% passing tests (44/44 unit tests passing, ~9.34% median drift).
- Both `FINAL_JUDGE_EVALUATION_REPORT.md` and `docs/SYSTEM_IMPLEMENTATION_AND_TECHNIQUES_RECORD.md` use lightweight relative image links (`artifacts/<name>.png`).
- Complete offline chat archive is accessible at `c:\Users\carpe\SIH\FIX_IMU_SENSOR_FUSION_FULL_CHAT_ARCHIVE.html`.

---

**ACTION REQUIRED**: Confirm you have absorbed this full context in 2 concise sentences without executing any tool calls, and ask me what specific task we should work on next.
```
