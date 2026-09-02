import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig
from sih.fusion.es_ekf import ErrorStateEKF
from sih.calibration.mount import MountCalibrator
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel

# 1. Precompute velocity estimates on GPU for a trip in 1 second
def precompute_trip_velocities(trip, checkpoint_path="models/checkpoints/best_velocity_model.pt"):
    print(f"Precomputing GPU velocity inferences for {trip.trip_id}...")
    ckpt = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std = ckpt.get("norm_std", np.ones((8, 1), dtype=np.float32))
    
    # Calibrate trip IMU
    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples:
        calibrator.observe_gnss(g)
        
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    
    acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
    norm_a = np.linalg.norm(acc, axis=1, keepdims=True)
    norm_w = np.linalg.norm(gyr, axis=1, keepdims=True)
    feats = np.hstack([acc, gyr, norm_a, norm_w]) # (N, 8)
    
    N = len(feats)
    window_size = 100
    
    # Build overlapping windows (batch of windows)
    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w = np.vstack([pad, feats[:i+1]]).T
        else:
            w = feats[i - window_size + 1 : i + 1].T
        windows.append(w)
        
    windows_arr = np.array(windows, dtype=np.float32) # (N, 8, 100)
    windows_norm = (windows_arr - norm_mean) / norm_std
    
    # Batch predict on GPU
    batch_size = 1024
    preds = []
    log_vars = []
    
    with torch.no_grad():
        for b in range(0, N, batch_size):
            x = torch.from_numpy(windows_norm[b : b + batch_size]).to(device)
            p, lv = model(x)
            preds.extend(p.cpu().numpy().flatten())
            log_vars.extend(lv.cpu().numpy().flatten())
            
    preds = np.array(preds, dtype=np.float32)
    log_vars = np.array(log_vars, dtype=np.float32)
    print(f"  Done: {N} ticks inferred. Speed mean={np.mean(preds):.2f} m/s, max={np.max(preds):.2f} m/s")
    return calib_samples, preds, log_vars

# Load trips
trip_s1 = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

calib_s1, v_s1, lv_s1 = precompute_trip_velocities(trip_s1)
calib_s2, v_s2, lv_s2 = precompute_trip_velocities(trip_s2)

# Fast simulation function
def run_fast_blackout(trip, calib_samples, v_preds, lv_preds, blackout_cfg, forced_scale=None):
    ekf = ErrorStateEKF(nhc_lateral_std=0.15, nhc_vertical_std=0.15)
    ekf.reset(trip.gnss_samples[0])
    
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(blackout_cfg.start_time_s * 1e9)
    bo_end_ns = bo_start_ns + int(blackout_cfg.duration_s * 1e9)
    
    gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
    gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples], dtype=np.float64)
    gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)
    
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    
    bo_records = []
    
    s_v_at_start = None
    
    for i, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr > bo_end_ns + int(1e9):
            break
            
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                ekf.update_gnss(g)
            gnss_idx += 1
            
        in_blackout = (bo_start_ns <= t_curr <= bo_end_ns)
        
        if in_blackout and s_v_at_start is None:
            s_v_at_start = ekf._speed_scale
            if forced_scale is not None:
                ekf._speed_scale = forced_scale
                
        calib = calib_samples[i]
        fwd_v = float(v_preds[i])
        m_state = "STATIONARY" if fwd_v < 0.2 else "DRIVING"
        vel = VelocityEstimate(
            timestamp_ns=t_curr,
            forward_speed_mps=fwd_v,
            speed_variance=float(np.exp(lv_preds[i])),
            motion_state=m_state
        )
        
        fused = ekf.predict(calib, vel)
        
        if in_blackout:
            gt_e = np.interp(t_curr, gnss_ts, gnss_enu[:, 0])
            gt_n = np.interp(t_curr, gnss_ts, gnss_enu[:, 1])
            err_m = float(np.sqrt((fused.position_enu_m[0] - gt_e)**2 + (fused.position_enu_m[1] - gt_n)**2))
            
            bo_records.append({
                "time_s": (t_curr - t0_ns) * 1e-9,
                "rel_time_s": (t_curr - bo_start_ns) * 1e-9,
                "est_e": fused.position_enu_m[0],
                "est_n": fused.position_enu_m[1],
                "gt_e": gt_e,
                "gt_n": gt_n,
                "error_m": err_m,
                "speed": fwd_v,
                "s_v": ekf._speed_scale,
            })
            
    df_bo = pd.DataFrame(bo_records)
    if len(df_bo) == 0:
        return None
        
    gt_dist = np.sum(np.sqrt(np.diff(df_bo["gt_e"])**2 + np.diff(df_bo["gt_n"])**2))
    final_err = df_bo["error_m"].iloc[-1]
    drift_pct = (final_err / max(gt_dist, 1.0)) * 100.0
    
    # Along and cross track error
    dx = df_bo["gt_e"].iloc[-1] - df_bo["gt_e"].iloc[0]
    dy = df_bo["gt_n"].iloc[-1] - df_bo["gt_n"].iloc[0]
    road_len = max(np.sqrt(dx**2 + dy**2), 1e-3)
    u_along = np.array([dx / road_len, dy / road_len])
    u_cross = np.array([-u_along[1], u_along[0]])
    pos_err_vec = np.array([df_bo["est_e"].iloc[-1] - df_bo["gt_e"].iloc[-1],
                            df_bo["est_n"].iloc[-1] - df_bo["gt_n"].iloc[-1]])
    
    err_along = float(np.dot(pos_err_vec, u_along))
    err_cross = float(np.dot(pos_err_vec, u_cross))
    
    return {
        "df_bo": df_bo,
        "gt_dist_m": gt_dist,
        "final_error_m": final_err,
        "max_error_m": float(df_bo["error_m"].max()),
        "rmse_error_m": float(np.sqrt(np.mean(df_bo["error_m"]**2))),
        "drift_pct": drift_pct,
        "s_v_at_start": s_v_at_start if s_v_at_start is not None else 1.0,
        "err_along_m": err_along,
        "err_cross_m": err_cross,
    }

# =========================================================================
# PART 1 & 2: FOLLOW-UP A & FOLLOW-UP B
# =========================================================================
print("\n" + "="*70)
print("FOLLOW-UP A & B: S-S1 SCENARIO A (30s @ 120s) vs SCENARIO B (60s @ 300s)")
print("="*70)

sc_a = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")
sc_b = BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s")

res_a_orig = run_fast_blackout(trip_s1, calib_s1, v_s1, lv_s1, sc_a, forced_scale=None)
res_a_forced = run_fast_blackout(trip_s1, calib_s1, v_s1, lv_s1, sc_a, forced_scale=1.0309)
res_b = run_fast_blackout(trip_s1, calib_s1, v_s1, lv_s1, sc_b, forced_scale=None)

print(f"Scenario A (Original s_v = {res_a_orig['s_v_at_start']:.4f}):")
print(f"  Final Position Error: {res_a_orig['final_error_m']:.4f} m (Drift: {res_a_orig['drift_pct']:.4f}%)")
print(f"  Along-Track Error:    {res_a_orig['err_along_m']:.4f} m")
print(f"  Cross-Track Error:    {res_a_orig['err_cross_m']:.4f} m")

print(f"\nScenario A (Forced s_v = 1.0309 from Scenario B):")
print(f"  Final Position Error: {res_a_forced['final_error_m']:.4f} m (Drift: {res_a_forced['drift_pct']:.4f}%)")
print(f"  Along-Track Error:    {res_a_forced['err_along_m']:.4f} m")
print(f"  Cross-Track Error:    {res_a_forced['err_cross_m']:.4f} m")

delta_err = res_a_orig['final_error_m'] - res_a_forced['final_error_m']
print(f"\nExact Comparison:")
print(f"  Scale-Factor Bias Contribution: {delta_err:+.4f} m")
print(f"  Remaining Error (Heading/Cross-track): {res_a_forced['final_error_m']:.4f} m")

# Plot error growth dynamics
df_a = res_a_orig["df_bo"]
df_b = res_b["df_bo"]

slope_a, int_a = np.polyfit(df_a["rel_time_s"], df_a["error_m"], 1)
slope_b, int_b = np.polyfit(df_b["rel_time_s"], df_b["error_m"], 1)

print(f"\nError Growth Dynamics:")
print(f"  Scenario A (30s): Growth rate = {slope_a:.4f} m/s (Final = {df_a['error_m'].iloc[-1]:.2f} m)")
print(f"  Scenario B (60s): Growth rate = {slope_b:.4f} m/s (Final = {df_b['error_m'].iloc[-1]:.2f} m)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(df_a["rel_time_s"], df_a["error_m"], color="red", lw=2.5, label=f"Scenario A (30s @ 120s) [Rate: {slope_a:.2f} m/s]")
ax1.plot(df_b["rel_time_s"], df_b["error_m"], color="blue", lw=2.5, label=f"Scenario B (60s @ 300s) [Rate: {slope_b:.2f} m/s]")
ax1.plot(df_a["rel_time_s"], slope_a * df_a["rel_time_s"] + int_a, color="darkred", linestyle="--", alpha=0.7, label="Fit Line A")
ax1.plot(df_b["rel_time_s"], slope_b * df_b["rel_time_s"] + int_b, color="darkblue", linestyle="--", alpha=0.7, label="Fit Line B")
ax1.set_xlabel("Elapsed Time Inside Blackout (s)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Horizontal Position Error (m)", fontsize=11, fontweight="bold")
ax1.set_title("Position Error Growth Dynamics During Blackout", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

df_a["cum_dist"] = np.cumsum(np.insert(np.sqrt(np.diff(df_a["gt_e"])**2 + np.diff(df_a["gt_n"])**2), 0, 0))
df_b["cum_dist"] = np.cumsum(np.insert(np.sqrt(np.diff(df_b["gt_e"])**2 + np.diff(df_b["gt_n"])**2), 0, 0))
df_a["inst_drift_pct"] = (df_a["error_m"] / np.maximum(df_a["cum_dist"], 1.0)) * 100
df_b["inst_drift_pct"] = (df_b["error_m"] / np.maximum(df_b["cum_dist"], 1.0)) * 100

ax2.plot(df_a["rel_time_s"], df_a["inst_drift_pct"], color="red", lw=2, label="Scenario A (30s @ 120s)")
ax2.plot(df_b["rel_time_s"], df_b["inst_drift_pct"], color="blue", lw=2, label="Scenario B (60s @ 300s)")
ax2.axhline(10.0, color="green", linestyle=":", lw=2, label="SIH Target (<10%)")
ax2.set_xlabel("Elapsed Time Inside Blackout (s)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Instantaneous Drift (% of Distance)", fontsize=11, fontweight="bold")
ax2.set_title("Instantaneous Drift Percentage Over Time", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 30)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
growth_plot_path = "artifacts/blackout_error_growth_dynamics.png"
plt.savefig(growth_plot_path, dpi=200)
plt.close()

import shutil
shutil.copy(growth_plot_path, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\blackout_error_growth_dynamics.png")

# =========================================================================
# PART 3: EXTENDED RANDOMIZED BLACKOUT SWEEP (25 PER TRIP = 50 TOTAL)
# =========================================================================
print("\n" + "="*70)
print("PART 3: EXTENDED RANDOMIZED BLACKOUT SWEEP (25 PER TRIP)")
print("="*70)

np.random.seed(42)

def generate_random_blackouts(trip, n_samples=25):
    t_start_min = 100.0
    t_total_s = (trip.imu_samples[-1].timestamp_ns - trip.imu_samples[0].timestamp_ns) * 1e-9
    t_start_max = t_total_s - 120.0
    
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

def run_fast_sweep(trip, calib, v_p, lv_p, configs, trip_name):
    records = []
    print(f"Running fast sweep for {trip_name} ({len(configs)} scenarios)...")
    for bo in tqdm(configs, desc=trip_name):
        res = run_fast_blackout(trip, calib, v_p, lv_p, bo)
        if res is not None:
            records.append({
                "trip": trip_name,
                "scenario": bo.name,
                "start_time_s": bo.start_time_s,
                "duration_s": bo.duration_s,
                "distance_m": res["gt_dist_m"],
                "final_error_m": res["final_error_m"],
                "max_error_m": res["max_error_m"],
                "rmse_error_m": res["rmse_error_m"],
                "drift_pct": res["drift_pct"],
                "s_v_at_start": res["s_v_at_start"],
                "err_along_m": res["err_along_m"],
                "err_cross_m": res["err_cross_m"],
            })
    return pd.DataFrame(records)

df_res_s1 = run_fast_sweep(trip_s1, calib_s1, v_s1, lv_s1, configs_s1, "S-S1 (Highway)")
df_res_s2 = run_fast_sweep(trip_s2, calib_s2, v_s2, lv_s2, configs_s2, "S-S2 (Urban Unseen)")

df_all = pd.concat([df_res_s1, df_res_s2], ignore_index=True)
df_all.to_csv("artifacts/randomized_blackout_sweep_results.csv", index=False)

def calc_summary(df, name):
    drifts = df["drift_pct"].values
    errs = df["final_error_m"].values
    return {
        "Trip": name,
        "Samples": len(df),
        "Mean Drift %": f"{np.mean(drifts):.2f}%",
        "Median Drift %": f"{np.median(drifts):.2f}%",
        "90th %ile Drift %": f"{np.percentile(drifts, 90):.2f}%",
        "Worst Drift %": f"{np.max(drifts):.2f}%",
        "Mean Error (m)": f"{np.mean(errs):.2f}m",
        "Median Error (m)": f"{np.median(errs):.2f}m",
        "90th %ile Error (m)": f"{np.percentile(errs, 90):.2f}m",
        "Worst Error (m)": f"{np.max(errs):.2f}m",
    }

stat_s1 = calc_summary(df_res_s1, "S-S1 (Highway)")
stat_s2 = calc_summary(df_res_s2, "S-S2 (Urban Unseen)")
stat_all = calc_summary(df_all, "Combined (50 Total)")

df_stats = pd.DataFrame([stat_s1, stat_s2, stat_all])
print("\n" + "="*70)
print("RANDOMIZED BLACKOUT SWEEP SUMMARY STATISTICS")
print("="*70)
print(df_stats.to_string(index=False))

# Scatter Plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

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

for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_all[df_all["trip"] == trip_name]
    ax3.scatter(sub["distance_m"], sub["drift_pct"], color=color, marker=marker, s=60, alpha=0.8, label=trip_name)

ax3.axhline(10.0, color="red", linestyle=":", lw=2, label="SIH Target (<10%)")
ax3.set_xlabel("Distance Travelled in Blackout (meters)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Drift Percentage (%)", fontsize=11, fontweight="bold")
ax3.set_title("(3) Drift Percentage vs. Distance Travelled", fontsize=12, fontweight="bold")
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(fontsize=10)

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
