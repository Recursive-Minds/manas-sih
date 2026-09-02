import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner
from sih.data.geo import geodetic_to_enu

# Load trips
print("Loading trips...")
trip_s1 = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

# =========================================================================
# PART 1: FOLLOW-UP A — ISOLATE SCALE-FACTOR CONTRIBUTION ON SCENARIO A
# =========================================================================
print("\n" + "="*70)
print("PART 1: FOLLOW-UP A — ISOLATE SCALE FACTOR CONTRIBUTION (S-S1 30s @ 120s)")
print("="*70)

def run_scenario_with_forced_scale(trip, blackout_cfg, forced_scale=None):
    cfg = PipelineConfig(
        calibration=CalibrationConfig(algorithm="auto"),
        velocity=VelocityEstimatorConfig(
            algorithm="tcn_attention",
            params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}
        ),
        fusion=FusionFilterConfig(
            algorithm="es_ekf_nhc",
            params={"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15}
        )
    )
    runner = BenchmarkRunner(config=cfg)
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(blackout_cfg.start_time_s * 1e9)
    bo_end_ns = bo_start_ns + int(blackout_cfg.duration_s * 1e9)
    
    runner.pipeline.reset(initial_gnss=trip.gnss_samples[0])
    
    gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
    gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples], dtype=np.float64)
    gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)
    
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    
    records = []
    
    for imu in trip.imu_samples:
        t_curr = imu.timestamp_ns
        t_s = (t_curr - t0_ns) * 1e-9
        
        if t_curr > bo_end_ns + int(2e9):
            break
            
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                runner.pipeline.process_gnss(g)
            gnss_idx += 1
            
        # If forced scale is active, overwrite scale right at blackout start
        if forced_scale is not None and t_curr >= bo_start_ns:
            runner.pipeline.fusion_filter._speed_scale = forced_scale
            
        calib, vel, fused, matched = runner.pipeline.process_imu(imu)
        
        gt_e = np.interp(t_curr, gnss_ts, gnss_enu[:, 0])
        gt_n = np.interp(t_curr, gnss_ts, gnss_enu[:, 1])
        err_m = float(np.sqrt((fused.position_enu_m[0] - gt_e)**2 + (fused.position_enu_m[1] - gt_n)**2))
        
        in_blackout = (bo_start_ns <= t_curr <= bo_end_ns)
        
        records.append({
            "time_s": t_s,
            "timestamp_ns": t_curr,
            "in_blackout": in_blackout,
            "error_m": err_m,
            "est_e": fused.position_enu_m[0],
            "est_n": fused.position_enu_m[1],
            "gt_e": gt_e,
            "gt_n": gt_n,
            "est_speed": vel.forward_speed_mps if vel is not None else 0.0,
            "speed_scale": runner.pipeline.fusion_filter._speed_scale,
            "heading_deg": np.degrees(fused.heading_rad),
        })
        
    df = pd.DataFrame(records)
    df_bo = df[df["in_blackout"]].copy()
    
    gt_dist = np.sum(np.sqrt(np.diff(df_bo["gt_e"])**2 + np.diff(df_bo["gt_n"])**2))
    final_err = df_bo["error_m"].iloc[-1]
    drift_pct = (final_err / gt_dist) * 100.0
    
    # Along-track and cross-track decomposition at end of blackout
    # Vector of road tangent: from start of bo to end of bo
    dx_road = df_bo["gt_e"].iloc[-1] - df_bo["gt_e"].iloc[0]
    dy_road = df_bo["gt_n"].iloc[-1] - df_bo["gt_n"].iloc[0]
    road_len = np.sqrt(dx_road**2 + dy_road**2)
    u_along = np.array([dx_road / road_len, dy_road / road_len])
    u_cross = np.array([-u_along[1], u_along[0]])
    
    pos_err_vec = np.array([df_bo["est_e"].iloc[-1] - df_bo["gt_e"].iloc[-1],
                            df_bo["est_n"].iloc[-1] - df_bo["gt_n"].iloc[-1]])
    
    err_along = float(np.dot(pos_err_vec, u_along))
    err_cross = float(np.dot(pos_err_vec, u_cross))
    
    return {
        "df_bo": df_bo,
        "final_error_m": final_err,
        "gt_dist_m": gt_dist,
        "drift_pct": drift_pct,
        "scale_at_start": df_bo["speed_scale"].iloc[0],
        "err_along_m": err_along,
        "err_cross_m": err_cross,
    }

sc_a = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")
res_orig = run_scenario_with_forced_scale(trip_s1, sc_a, forced_scale=None)
res_forced = run_scenario_with_forced_scale(trip_s1, sc_a, forced_scale=1.0309)

print(f"Original Scenario A (s_v = {res_orig['scale_at_start']:.4f}):")
print(f"  Final Position Error: {res_orig['final_error_m']:.4f} m (Drift: {res_orig['drift_pct']:.4f}%)")
print(f"  Along-Track Error:    {res_orig['err_along_m']:.4f} m")
print(f"  Cross-Track Error:    {res_orig['err_cross_m']:.4f} m")

print(f"\nForced Scenario A (s_v = 1.0309 from Scenario B):")
print(f"  Final Position Error: {res_forced['final_error_m']:.4f} m (Drift: {res_forced['drift_pct']:.4f}%)")
print(f"  Along-Track Error:    {res_forced['err_along_m']:.4f} m")
print(f"  Cross-Track Error:    {res_forced['err_cross_m']:.4f} m")

delta_err = res_orig['final_error_m'] - res_forced['final_error_m']
print(f"\nExact Comparison:")
print(f"  Total Error Explained by Scale-Factor Bias: {delta_err:+.4f} m ({abs(delta_err)/res_orig['final_error_m']*100:.2f}%)")
print(f"  Remaining Unexplained Error:               {res_forced['final_error_m']:.4f} m ({res_forced['final_error_m']/res_orig['final_error_m']*100:.2f}%)")

# =========================================================================
# PART 2: FOLLOW-UP B — EXPLAIN THE FLAT ABSOLUTE ERROR (ERROR GROWTH OVER TIME)
# =========================================================================
print("\n" + "="*70)
print("PART 2: FOLLOW-UP B — CUMULATIVE POSITION ERROR OVER TIME DURING BLACKOUT")
print("="*70)

sc_b = BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s")
res_b = run_scenario_with_forced_scale(trip_s1, sc_b, forced_scale=None)

df_bo_a = res_orig["df_bo"].copy()
df_bo_b = res_b["df_bo"].copy()

df_bo_a["rel_time_s"] = df_bo_a["time_s"] - df_bo_a["time_s"].iloc[0]
df_bo_b["rel_time_s"] = df_bo_b["time_s"] - df_bo_b["time_s"].iloc[0]

# Calculate growth rates (linear fit slope)
slope_a, intercept_a = np.polyfit(df_bo_a["rel_time_s"], df_bo_a["error_m"], 1)
slope_b, intercept_b = np.polyfit(df_bo_b["rel_time_s"], df_bo_b["error_m"], 1)

print(f"Scenario A (30s): Error Growth Rate = {slope_a:.4f} m/s (Final Error = {df_bo_a['error_m'].iloc[-1]:.2f} m @ 30s)")
print(f"Scenario B (60s): Error Growth Rate = {slope_b:.4f} m/s (Final Error = {df_bo_b['error_m'].iloc[-1]:.2f} m @ 60s)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Overlay of Error vs Relative Blackout Time
ax1.plot(df_bo_a["rel_time_s"], df_bo_a["error_m"], color="red", lw=2.5, label=f"Scenario A (30s @ 120s) [Rate: {slope_a:.2f} m/s]")
ax1.plot(df_bo_b["rel_time_s"], df_bo_b["error_m"], color="blue", lw=2.5, label=f"Scenario B (60s @ 300s) [Rate: {slope_b:.2f} m/s]")
ax1.plot(df_bo_a["rel_time_s"], slope_a * df_bo_a["rel_time_s"] + intercept_a, color="darkred", linestyle="--", alpha=0.7, label="Fit Line A")
ax1.plot(df_bo_b["rel_time_s"], slope_b * df_bo_b["rel_time_s"] + intercept_b, color="darkblue", linestyle="--", alpha=0.7, label="Fit Line B")
ax1.set_xlabel("Elapsed Time Inside Blackout (s)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Horizontal Position Error (m)", fontsize=11, fontweight="bold")
ax1.set_title("Position Error Growth Dynamics During Blackout", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# Plot 2: Error as % of Distance Travelled vs Time
df_bo_a["cum_dist"] = np.cumsum(np.insert(np.sqrt(np.diff(df_bo_a["gt_e"])**2 + np.diff(df_bo_a["gt_n"])**2), 0, 0))
df_bo_b["cum_dist"] = np.cumsum(np.insert(np.sqrt(np.diff(df_bo_b["gt_e"])**2 + np.diff(df_bo_b["gt_n"])**2), 0, 0))
df_bo_a["inst_drift_pct"] = (df_bo_a["error_m"] / np.maximum(df_bo_a["cum_dist"], 1.0)) * 100
df_bo_b["inst_drift_pct"] = (df_bo_b["error_m"] / np.maximum(df_bo_b["cum_dist"], 1.0)) * 100

ax2.plot(df_bo_a["rel_time_s"], df_bo_a["inst_drift_pct"], color="red", lw=2, label="Scenario A (30s @ 120s)")
ax2.plot(df_bo_b["rel_time_s"], df_bo_b["inst_drift_pct"], color="blue", lw=2, label="Scenario B (60s @ 300s)")
ax2.axhline(10.0, color="green", linestyle=":", lw=2, label="SIH Target (<10%)")
ax2.set_xlabel("Elapsed Time Inside Blackout (s)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Instantaneous Drift (% of Distance)", fontsize=11, fontweight="bold")
ax2.set_title("Instantaneous Drift Percentage Over Time", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 30)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
growth_plot_path = "artifacts/blackout_error_growth_dynamics.png"
plt.savefig(growth_plot_path, dpi=200)
plt.close()

import shutil
shutil.copy(growth_plot_path, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\blackout_error_growth_dynamics.png")
print(f"Saved growth dynamics plot to {growth_plot_path}")

# =========================================================================
# PART 3: EXTENDED RANDOMIZED BLACKOUT SWEEP (25 SAMPLES PER TRIP)
# =========================================================================
print("\n" + "="*70)
print("PART 3: RANDOMIZED BLACKOUT SWEEP ACROSS S-S1 AND S-S2")
print("="*70)

np.random.seed(42)

def generate_random_blackouts(trip, n_samples=25):
    t_start_min = 100.0 # allow initial alignment & driving
    t_total_s = (trip.imu_samples[-1].timestamp_ns - trip.imu_samples[0].timestamp_ns) * 1e-9
    t_start_max = t_total_s - 120.0
    
    # Generate random start times and durations
    starts = np.random.uniform(t_start_min, t_start_max, n_samples)
    durations = np.random.uniform(20.0, 90.0, n_samples)
    
    configs = []
    for i, (st, dur) in enumerate(zip(starts, durations)):
        configs.append(BlackoutConfig(
            start_time_s=float(st),
            duration_s=float(dur),
            name=f"rand_bo_{i+1:02d}_{dur:.0f}s_at_{st:.0f}s"
        ))
    return configs

configs_s1 = generate_random_blackouts(trip_s1, n_samples=25)
configs_s2 = generate_random_blackouts(trip_s2, n_samples=25)

def evaluate_sweep(trip, configs, trip_name):
    cfg = PipelineConfig(
        calibration=CalibrationConfig(algorithm="auto"),
        velocity=VelocityEstimatorConfig(
            algorithm="tcn_attention",
            params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}
        ),
        fusion=FusionFilterConfig(
            algorithm="es_ekf_nhc",
            params={"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15}
        )
    )
    runner = BenchmarkRunner(config=cfg)
    
    results = []
    print(f"Evaluating {len(configs)} randomized blackouts on {trip_name}...")
    
    for idx, bo in enumerate(configs):
        res = runner.run_trip(trip, bo)
        df_bo = res.history_df[res.history_df["in_blackout"]].copy()
        
        gt_dist = np.sum(np.sqrt(np.diff(df_bo["gt_e"])**2 + np.diff(df_bo["gt_n"])**2))
        final_err = res.final_position_error_m
        drift_pct = (final_err / max(gt_dist, 1.0)) * 100.0
        
        # Get s_v at blackout start
        t0_ns = trip.imu_samples[0].timestamp_ns
        bo_start_ns = t0_ns + int(bo.start_time_s * 1e9)
        idx_start = np.searchsorted(df_bo["timestamp_ns"], bo_start_ns)
        idx_start = min(idx_start, len(df_bo) - 1)
        
        # Get speed scale from filter history
        # We can extract it from the pipeline
        s_v_start = runner.pipeline.fusion_filter._speed_scale
        
        results.append({
            "trip": trip_name,
            "scenario": bo.name,
            "start_time_s": bo.start_time_s,
            "duration_s": bo.duration_s,
            "distance_m": gt_dist,
            "final_error_m": final_err,
            "max_error_m": res.max_position_error_m,
            "rmse_error_m": res.rmse_position_error_m,
            "drift_pct": drift_pct,
            "s_v_at_start": s_v_start,
        })
        print(f"  [{idx+1:02d}/25] Dur={bo.duration_s:.1f}s | Dist={gt_dist:.1f}m | Err={final_err:.2f}m | Drift={drift_pct:.2f}% | s_v={s_v_start:.4f}")
        
    return pd.DataFrame(results)

df_res_s1 = evaluate_sweep(trip_s1, configs_s1, "S-S1 (Highway)")
df_res_s2 = evaluate_sweep(trip_s2, configs_s2, "S-S2 (Urban Unseen)")

df_all = pd.concat([df_res_s1, df_res_s2], ignore_index=True)
df_all.to_csv("artifacts/randomized_blackout_sweep_results.csv", index=False)

# Summary statistics
def calc_summary(df, name):
    drifts = df["drift_pct"].values
    errs = df["final_error_m"].values
    return {
        "Trip": name,
        "Samples": len(df),
        "Mean Drift %": np.mean(drifts),
        "Median Drift %": np.median(drifts),
        "90th %ile Drift %": np.percentile(drifts, 90),
        "Worst Drift %": np.max(drifts),
        "Mean Error (m)": np.mean(errs),
        "Median Error (m)": np.median(errs),
        "90th %ile Error (m)": np.percentile(errs, 90),
        "Worst Error (m)": np.max(errs),
    }

stat_s1 = calc_summary(df_res_s1, "S-S1 (Highway)")
stat_s2 = calc_summary(df_res_s2, "S-S2 (Urban Unseen)")
stat_all = calc_summary(df_all, "Combined Overall")

df_stats = pd.DataFrame([stat_s1, stat_s2, stat_all])
print("\n" + "="*70)
print("RANDOMIZED BLACKOUT SWEEP SUMMARY STATISTICS")
print("="*70)
print(df_stats.to_string(index=False))

# =========================================================================
# SCATTER PLOTS & REGRESSION ANALYSIS
# =========================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Final Error vs Duration
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_all[df_all["trip"] == trip_name]
    ax1.scatter(sub["duration_s"], sub["final_error_m"], color=color, marker=marker, s=60, alpha=0.8, label=trip_name)
    m, b = np.polyfit(sub["duration_s"], sub["final_error_m"], 1)
    x_line = np.linspace(20, 90, 100)
    ax1.plot(x_line, m*x_line + b, color=color, linestyle="--", lw=2, label=f"{trip_name} Trend ({m:.2f} m/s)")

ax1.set_xlabel("Blackout Duration (seconds)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Absolute Final Position Error (meters)", fontsize=11, fontweight="bold")
ax1.set_title("(1) Final Position Error vs. Blackout Duration", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# Plot 2: Final Error vs s_v at Blackout Start
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_all[df_all["trip"] == trip_name]
    ax2.scatter(sub["s_v_at_start"], sub["final_error_m"], color=color, marker=marker, s=60, alpha=0.8, label=trip_name)
    m, b = np.polyfit(sub["s_v_at_start"], sub["final_error_m"], 1)
    x_line = np.linspace(sub["s_v_at_start"].min()-0.02, sub["s_v_at_start"].max()+0.02, 100)
    ax2.plot(x_line, m*x_line + b, color=color, linestyle="--", lw=2, label=f"{trip_name} Trend")

ax2.set_xlabel("Speed Scale Factor $s_v$ at Blackout Start", fontsize=11, fontweight="bold")
ax2.set_ylabel("Absolute Final Position Error (meters)", fontsize=11, fontweight="bold")
ax2.set_title("(2) Final Position Error vs. Pre-Blackout $s_v$", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

# Plot 3: Drift % vs Distance Travelled
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_all[df_all["trip"] == trip_name]
    ax3.scatter(sub["distance_m"], sub["drift_pct"], color=color, marker=marker, s=60, alpha=0.8, label=trip_name)

ax3.axhline(10.0, color="red", linestyle=":", lw=2, label="SIH Target (<10%)")
ax3.set_xlabel("Distance Travelled in Blackout (meters)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Drift Percentage (%)", fontsize=11, fontweight="bold")
ax3.set_title("(3) Drift Percentage vs. Distance Travelled", fontsize=12, fontweight="bold")
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(fontsize=10)

# Plot 4: Drift % Distribution (Boxplot / Violin)
box_data = [df_res_s1["drift_pct"], df_res_s2["drift_pct"]]
ax4.boxplot(box_data, tick_labels=["S-S1 (Highway)", "S-S2 (Urban Unseen)"], patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red", lw=2))
ax4.axhline(10.0, color="green", linestyle=":", lw=2, label="SIH Target (<10%)")
ax4.set_ylabel("Drift Percentage (%)", fontsize=11, fontweight="bold")
ax4.set_title("(4) Drift % Distribution Across 50 Blackouts", fontsize=12, fontweight="bold")
ax4.grid(True, linestyle="--", alpha=0.6)
ax4.legend(fontsize=10)

plt.tight_layout()
sweep_plot_path = "artifacts/randomized_blackout_sweep.png"
plt.savefig(sweep_plot_path, dpi=200)
plt.close()

shutil.copy(sweep_plot_path, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\randomized_blackout_sweep.png")
print(f"Saved randomized sweep plot to {sweep_plot_path}")
