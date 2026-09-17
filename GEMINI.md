# Project Memory & Architecture Guide: Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion (SIH)

Refer to [CLAUDE.md](file:///c:/Users/carpe/SIH/CLAUDE.md) for full context, rules, workflow principles, and phase roadmaps.

## Key Rules for this Workspace
1. **One narrow task at a time**: Never expand scope silently.
2. **Real Data Verification**: Never report a step complete without concrete, inspectable output (metrics, plots, error rates) on real data.
3. **Data Splits**: ALWAYS split datasets strictly by trip/sequence, NEVER by row.
4. **Modularity**: All stages (`IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition`) must communicate strictly through fixed contracts behind abstract interfaces, configured at a single assembly point.
5. **Sensor-Agnostic Core**: Core engine is decoupled from Android-specific code and must support phone IMUs (10Hz) up to high-rate external IMUs (~200Hz).
6. **Benchmark Target**: Drift < 10% of total distance travelled during GNSS blackout (< 5m over 50m, or < 100m over 1km).
7. **No Assumptions on Datasets**: Never assume dataset availability, file schemas, or remote storage structure — always search, probe, and verify real repository files/URLs before hardcoding.
8. **High-Speed Velocity Scaling**: When training neural velocity estimators from IMU, avoid loss functions that compress gradients on high speeds (e.g. tight Huber thresholds). Verify that the predicted speed scale ratio sum(v_hat) / sum(v_GT) is approximately 1.00.
9. **3D Mount Invariance**: Always apply 3D SO(3) rotational data augmentation during IMU model training to prevent memorization of static cradle gravity vectors.
10. **Trajectory Error Decomposition**: When evaluating dead-reckoning performance on real data, decompose position errors into along-track (speed scale) and cross-track (turn rate / heading) components to diagnose drift root causes.
11. **End-to-End Benchmark & Report Synchronization Pipeline**:
Whenever any algorithmic change, model retraining, or parameter update is made, follow this mandatory 4-step execution chain:
- **Step 1**: Implement the changes directly in production code (`benchmarks/run_final_benchmark.py`, `sih/`).
- **Step 2**: Re-run the full 40-scenario benchmark on real data to produce new coordinates and error metrics.
- **Step 3**: Re-render all trajectory maps, gallery plots, and distribution charts from the newly generated benchmark run.
- **Step 4**: Re-generate BOTH `FINAL_JUDGE_EVALUATION_REPORT.md` AND `FINAL_JUDGE_EVALUATION_REPORT.html` with updated tables, scorecards, and clean relative image references to `artifacts/` (never inline raw base64 image strings into Markdown files, to prevent IDE and server context window exhaustion). Never report an update complete without running this entire chain end-to-end.
12. **Clean Plain-Text Math Formatting**: NEVER use raw LaTeX syntax (e.g. `$`, `$$`, `\approx`, `\frac`, `\text{}`, `\omega`, `\Delta`, `\sigma`) in user responses, implementation plans, or generated markdown files. Always format mathematical expressions and physical variables in readable plain text or standard Unicode symbols (e.g. `p_head = exp(-0.5 * (diff_h / sigma)^2)`, `omega_z >= 2.5 deg/s`, `90 degrees`, `kappa = 1/R`, `a_lat <= 1.2 m/s^2`).
13. **Strict Benchmark / Core Logic Separation**: The benchmark script (`benchmarks/run_final_benchmark.py`) must ONLY contain benchmark orchestration code (scenario selection, metric computation, plotting, report generation). ALL algorithmic logic — including GPS interpolation, heading seeding preparation, speed scaling, road network construction, and EKF configuration — MUST live in dedicated modules under `sih/` (e.g. `sih/fusion/`, `sih/data/`, `sih/map/`, `sih/calibration/`). The benchmark script calls into these modules; it never re-implements or inlines core logic. Any new algorithmic feature must be implemented in `sih/` first, then invoked from the benchmark. Violating this rule is a critical architectural bug.
14. **Anti-Bloat & Token Conservation Protocol (Prefix-Clear Prevention)**:
To prevent conversation token exhaustion, infinite polling loops, and remote API prefix clearing:
- **No-Tool Bootstrap Acknowledgment**: When receiving a bootstrap prompt, context restoration prompt, or initial greeting, NEVER launch autonomous exploratory tool calls (`view_file`, `run_command`, `list_dir`, `grep_search`) across the repository. Acknowledge readiness directly from the prompt text in 1-2 concise sentences. Only invoke tools when given a specific coding, debugging, or benchmark task in subsequent turns.
- **Surgical File Viewing**: Never read whole files exceeding 150 lines with `view_file`. Always specify narrow `StartLine` and `EndLine` slices (<= 100 lines). Never read entire large reports, docs, or full modules into the prompt stream when a `grep_search` or targeted slice suffices.
- **Concise Command Outputs**: Always pipe, filter, or truncate terminal commands (e.g. `Select-Object -First 20`, `head -n 20`, `--quiet`). Never allow long dumps or recursive scans to flood the tool response buffer.
- **Zero Raw Media/Archive Inspection**: Never call `view_file`, `grep_search`, or unfiltered git diff on massive offline archives (e.g. `FIX_IMU_SENSOR_FUSION_FULL_CHAT_ARCHIVE.html`), video files, or large data arrays.
- **Strict Prohibition of Polling Loops**: Never call `manage_task("list")` or status check in a rapid loop. Rely entirely on the reactive wakeup system.
15. **The Master README.md Document (Single Source of Truth & Full Context Union)**:
`README.md` is the SINGLE authoritative master reference document for the entire project. It is the comprehensive union containing all context directly inline:
- Full SIH 26168 Problem Statement, Indian transit challenges, and 3 operational tiers.
- Initial concept, foundational hypotheses, and chronological evolution across all phases.
- Key scientific discoveries & architectural pivots (why neural heading failed, 9s GPS optical illusion, 1.3s causal lag, OSM curvature kinks).
- Comprehensive record of all 20 physical failure modes and hardening solutions.
- Full mathematical formulations (clean plain-text math, Rule 12), physical models, and architecture.
- Complete active parameters registry and full codebase module inventory.
- Definitive empirical benchmark evaluation scorecards (6 seeds x 40 scenarios = 240 runs, domain breakdowns, and trajectory plots).
- Quickstart guide, reproduction commands, and test verification.
**MANDATORY UPDATE**: `README.md` must be updated directly whenever any code change, model retraining, parameter tweak, or benchmark run is performed. It contains the context itself and must NOT delegate core information to external/auxiliary markdown files.
16. **No Auxiliary / Dispersed Markdown Files Protocol**:
- NEVER create auxiliary markdown documentation files (e.g. in `docs/` or scratch directories) that duplicate or fragment system architecture, roadmap, or metrics. All technical details, formulas, and progress records belong exclusively in `README.md`.
- `FINAL_JUDGE_EVALUATION_REPORT.md` and `.html` are the automated direct outputs of `benchmarks/run_final_benchmark.py`. Their latest empirical tables and scorecards must always be mirrored into Section 16 of `README.md` so that numbers are 100% consistent across the codebase.
