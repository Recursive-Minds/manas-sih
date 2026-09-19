# Technical Audit 2: Ensemble Provenance, Causality, and Route Matcher Status

## 1. sih/models/ensemble_gating.py & Call Site Audit

### a. Runtime Gating Inputs & Trip Identification
- **Input parameters**: `LOTOEnsembleVelocityEstimator.predict_trip` takes `trip_id: str` as an explicit runtime argument (`sih/models/ensemble_gating.py:96`).
- **Code Quote** (`sih/models/ensemble_gating.py:111-113`):
```python
d = float(discount_d if discount_d is not None else self.discount_d)
held_out_fold = TRIP_HELD_OUT_FOLD.get(trip_id, "Fold 5 (S-S4)")
```
- **Code Quote** (`sih/models/ensemble_gating.py:34-40`):
```python
TRIP_HELD_OUT_FOLD = {
    "S-M":   "Fold 1 (S-M)",
    "S-S1":  "Fold 2 (S-S1)",
    "S-S2":  "Fold 3 (S-S2)",
    "S-S3a": "Fold 4 (S-S3a)",
    "S-S4":  "Fold 5 (S-S4)",
}
```
- **Code Quote** at Call Site (`sih/models/inference.py:103-105` invoked from `benchmarks/run_final_benchmark.py:193`):
```python
if model_type == "loto_ensemble":
    v_fused, _ = model.predict_trip(trip_id=trip_id, calib_samples=calib_samples)
    return v_fused.astype(np.float32)
```
- **Finding**: Yes. Gating explicitly reads `trip_id` at runtime to look up `held_out_fold`.

---

### b. Fold Discounting D=0.50 Mechanics & Weight Distribution
- **Formulation** (`sih/models/ensemble_gating.py:190-203`):
```python
raw_weights = np.zeros_like(all_var, dtype=np.float64)
for idx, fname in enumerate(self.fold_names):
    inv_var = 1.0 / np.maximum(all_var[:, idx], 1e-4)
    if fname == held_out_fold:
        raw_weights[:, idx] = inv_var * 1.0
    else:
        raw_weights[:, idx] = inv_var * discount_d
sum_weights = np.sum(raw_weights, axis=1, keepdims=True)
norm_weights = raw_weights / np.maximum(sum_weights, 1e-9)
```
- **Weight allocation for scenario from Trip X (e.g. S-M)**:
  - Fold 1 (S-M held-out): Multiplier = 1.00
  - Fold 2 (S-S1 held-out, trained on S-M): Multiplier = 0.50
  - Fold 3 (S-S2 held-out, trained on S-M): Multiplier = 0.50
  - Fold 4 (S-S3a held-out, trained on S-M): Multiplier = 0.50
  - Fold 5 (S-S4 held-out, trained on S-M): Multiplier = 0.50
- **Non-zero contribution of trained folds**:
  - **YES**. Every fold trained on Trip X receives a non-zero multiplier of 0.50.
  - Across the 4 in-distribution folds, aggregate multiplier sum = 0.50 * 4 = 2.00 vs 1.00 for the held-out fold.
  - Under equal predicted variance, models that saw Trip X in training contribute **66.7%** of the ensemble weight; the held-out model contributes only **33.3%**.

---

### c. LOTO Checkpoint Provenance & Training Configuration
- **Checkpoint metadata inspection** (`models/checkpoints/loto_moe_fold_*.pt`):
  - `loto_moe_fold_1_S-M.pt`: `test_trip: 'S-M'`, `train_trips: ['S-S1', 'S-S2', 'S-S3a', 'S-S4']`, `in_channels: 14`.
  - `loto_moe_fold_2_S-S1.pt`: `test_trip: 'S-S1'`, `train_trips: ['S-M', 'S-S2', 'S-S3a', 'S-S4']`, `in_channels: 14`.
  - `loto_moe_fold_3_S-S2.pt`: `test_trip: 'S-S2'`, `train_trips: ['S-M', 'S-S1', 'S-S3a', 'S-S4']`, `in_channels: 14`.
  - `loto_moe_fold_4_S-S3a.pt`: `test_trip: 'S-S3a'`, `train_trips: ['S-M', 'S-S1', 'S-S2', 'S-S4']`, `in_channels: 14`.
  - `loto_moe_fold_5_S-S4.pt`: `test_trip: 'S-S4'`, `train_trips: ['S-M', 'S-S1', 'S-S2', 'S-S3a']`, `in_channels: 14`.
- **Training partitions**:
  - `SpeedSequenceDataset` and `load_trip` default to `partition="all"` (`sih/data/sequence_dataset.py:293`).
  - Training used the **full duration** of the 4 training trips (Parts 1, 2, and 3), not Part 1 only.
- **Training script provenance**:
  - No training script for `loto_moe_fold_*.pt` exists in the repository.
  - Committed as pre-trained binary weights in commit `bca5885` (`git log -S "loto_moe_fold"`).
  - Existing training script `scripts/train_can_moe.py:42-60` trains only the single dual-expert architecture on S-M, S-S1, S-S2 (Part 1 only).

---

### d. Canonical Benchmark with Strictly Held-Out Checkpoint (D=0.0)
- **Setup**: For trip X, set `discount_d = 0.0`. Multiplier for all 4 in-distribution folds = 0.0; weight for held-out fold = 1.0 (100% pure unseen inference).
- **Results (Seed 541098, 40 Scenarios, OSM Map)**:
  - **Median Drift**: **15.56%** (vs **11.59%** baseline, **+3.97% regression**)
  - **P90 Drift**: **52.36%** (vs **32.56%** baseline, **+19.80% regression**)
  - **Tier-1 Count (<10% drift)**: **12 / 40** (vs **16 / 40** baseline, **-4 scenarios**)
- **Per-Trip Breakdown (Median Drift)**:
  - S-M (Highway): 16.40% -> 22.18% (+5.78%)
  - S-S2 (Arterial): 9.25% -> 24.37% (+15.12%)
  - S-S1 (Urban): 19.19% -> 33.40% (+14.21%)
  - S-S3a (Mixed): 4.51% -> 11.36% (+6.85%)
  - S-S4 (Arterial): 10.98% -> 9.74% (-1.24%)

---

### e. Canonical Benchmark with Single TorchScript Mobile Checkpoint
- **Setup**: `model_path = "models/checkpoints/best_moe_velocity_model.pt"` (the exact single dual-expert MoE exported to `moe_velocity_model.torchscript.pt`).
- **Results (Seed 541098, 40 Scenarios, OSM Map)**:
  - **Median Drift**: **11.96%** (vs **11.59%** baseline, **+0.37% difference**)
  - **P90 Drift**: **31.39%** (vs **32.56%** baseline, **-1.17% improvement**)
  - **Tier-1 Count (<10% drift)**: **18 / 40** (vs **16 / 40** baseline, **+2 scenarios**)
- **Per-Trip Breakdown (Median Drift)**:
  - S-M (Highway): 16.40% -> 13.43% (-2.97%)
  - S-S2 (Arterial): 9.25% -> 17.95% (+8.70%)
  - S-S1 (Urban): 19.19% -> 17.59% (-1.60%)
  - S-S3a (Mixed): 4.51% -> 7.76% (+3.25%)
  - S-S4 (Arterial): 10.98% -> 20.99% (+10.01%)

---

### Comparison Scorecard Summary

| Configuration | Median Drift | P90 Drift | Tier-1 Count (<10%) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Canonical Baseline (LOTO D=0.50)** | **11.59%** | **32.56%** | **16 / 40** | Folds trained on test trip receive 66.7% weight |
| **Q1d: Pure Held-Out Only (LOTO D=0.0)** | **15.56%** | **52.36%** | **12 / 40** | Zero leakage; only unseen fold evaluated |
| **Q1e: Single TorchScript Mobile Model** | **11.96%** | **31.39%** | **18 / 40** | Production edge model (`best_moe_velocity_model.pt`) |

---

## 2. Causality Audit of the Batch Engine

For a scenario with blackout window `[t0, t1]`:

### Item-by-Item Verification
1. **Centered Windows**:
   - **None in model inference**.
   - `sih/models/ensemble_gating.py:155-159`: Prepends `pad = repeat(feats[0], 99)` and slices `sliding_window_view(..., window_shape=100)`. Window ending at index `k` contains indices `[k-99, k]`. Model extracts `s_seq[:, -1]` (current step only).
2. **Zero-Phase Filtering (`filtfilt`)**:
   - **Not in active pipeline**.
   - `sih/velocity/invariant_features.py:78-121` uses one-sided causal Butterworth filter (`scipy.signal.lfilter`) and first-order exponential moving average.
   - `sih/data/vibration.py:52-53` contains `signal.filtfilt`, but is dead/unreferenced code.
3. **Whole-Array Feature Normalization**:
   - **Feature z-scoring is causal**: `ensemble_gating.py:154` uses pre-saved `norm_mean` and `norm_std` from checkpoint tensors, not sequence statistics.
   - **Non-Causal Leak**: `InvariantFeatureExtractor.extract` (`sih/velocity/invariant_features.py:231`):
     ```python
     cal = estimate_gyro_up_axis(gyro, accel_up=np.mean(g_hat, axis=0))
     ```
     `np.mean(g_hat, axis=0)` computes the global mean of gravity over all 24,000 to 105,000 frames of the full trip, including samples beyond `t1`.
4. **Future GNSS Lookahead**:
   - **Strictly Causal**.
   - `sih/engine/dead_reckoning_engine.py:228-233`:
     ```python
     while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
         if g.timestamp_ns <= bo_start_ns:
             ekf_pure.update_gnss(g)
             ekf_map.update_gnss(g)
         gnss_idx += 1
     ```
     GNSS updates are blocked during `t_curr > bo_start_ns`. No future GNSS enters the filter.
5. **Pre-Blackout Scale (Alpha) & Heading Seeding**:
   - **Strictly Causal**.
   - `pre_gnss_window` (`dead_reckoning_engine.py:246-250`): Filters `timestamp_ns <= bo_start_ns` within preceding 25 seconds.
   - `speed_scale` (`dead_reckoning_engine.py:304`): Slices `v_preds[k]` for `k < j` (prior to entry timestep `j`).
   - Heading seeding (`dead_reckoning_engine.py:358-369`): Uses only `pre_gnss_window` and gyro integration up to `bo_start_ns`.
6. **Map Network Construction**:
   - **Non-Causal Spatial Leak**:
   - `sih/map/network.py:935-940` (`load_trip_road_network`):
     ```python
     bbox = (
         min(valid_lats) - pad_lat,
         min(valid_lons) - pad_lon,
         max(valid_lats) + pad_lat,
         max(valid_lons) + pad_lon,
     )
     ```
     Queries OSM Overpass API using a bounding box encompassing the entire trip's GNSS track (including blackout and post-blackout regions).
7. **Post-Scenario Evaluation**:
   - `dead_reckoning_engine.py:542-550`: Uses `bo_gnss` ground truth points to compute endpoint error and along/cross-track decomposition. This runs purely post-hoc after scenario termination and does not feed into filter states.

### Causality Verdict
- **PARTIALLY REFUTED**: State propagation and filter measurement updates during blackout are strictly causal. However, strict sequence causality is broken in two pre-processing steps: (1) `np.mean(g_hat, axis=0)` in `InvariantFeatureExtractor.extract:231`, and (2) full-trip bounding box calculation for OSM road loading in `sih/map/network.py:935`.

---

## 3. RouteMatcher Runtime Status in Canonical Benchmark

- **Runtime Call Site** (`benchmarks/run_final_benchmark.py:314-318`):
```python
res = run_scenario(
    trip, calib_samples, v_preds, road_net, g_cand, dur,
    domain=domain, can_speeds=can_speeds_dict.get(tid),
    enable_speed_scale=enable_speed_scale,
)
```
- `enable_route_matching` is **NOT passed** by `run_final_benchmark.py`.
- **Target Function Signature** (`sih/engine/dead_reckoning_engine.py:651-660`):
```python
def run_dead_reckoning_scenario(
    ...,
    enable_route_matching: bool = False,
    enable_speed_scale: bool = True,
) -> Optional[Dict[str, Any]]:
    engine = DeadReckoningEngine(
        enable_route_matching=enable_route_matching,
        enable_speed_scale=enable_speed_scale,
    )
```
- **Engine Constructor** (`sih/engine/dead_reckoning_engine.py:38, 45`):
```python
enable_route_matching: bool = False,
...
self.enable_route_matching = enable_route_matching
```
- **Execution Guard** (`sih/engine/dead_reckoning_engine.py:446, 462-463`):
```python
if self.enable_route_matching and entry_segment is not None and len(bo_timestamps_ns) >= 5:
    final_route_res = self.route_matcher.match_blackout_route(...)
...
if not self.enable_route_matching:
    route_fallback_reason = "route matching disabled"
```
- **Runtime Flag Value**: **`enable_route_matching = False`**.
- **Verdict**: `RouteMatcher` is **INACTIVE** in the canonical benchmark path. It is never invoked. Map matching in the benchmark is performed exclusively by `HMMMapMatcher`.
