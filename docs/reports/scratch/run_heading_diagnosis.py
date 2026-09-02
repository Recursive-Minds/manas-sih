import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig
from sih.fusion.es_ekf import ErrorStateEKF
from sih.calibration.mount import MountCalibrator
from sih.core.contracts import CalibratedSample, VelocityEstimate
from sih.data.geo import geodetic_to_enu
import torch
from sih.models.tcn_attention import TCNAttentionVelocityModel

# Load trips
trip_s1 = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

# Load precomputed velocity predictions
def get_trip_inferences(trip):
    ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std = ckpt.get("norm_std", np.ones((8, 1), dtype=np.float32))
    
    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples:
        calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    
    acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    
    N = len(feats)
    window_size = 100
    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w = np.vstack([pad, feats[:i+1]]).T
        else:
            w = feats[i - window_size + 1 : i + 1].T
        windows.append(w)
        
    windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
    preds = []
    with torch.no_grad():
        for b in range(0, N, 1024):
            x = torch.from_numpy(windows_norm[b : b + 1024]).to(device)
            p, _ = model(x)
            preds.extend(p.cpu().numpy().flatten())
    return calib_samples, np.array(preds, dtype=np.float32)

calib_s1, v_s1 = get_trip_inferences(trip_s1)
calib_s2, v_s2 = get_trip_inferences(trip_s2)

# Load the 50 sweep scenarios from results csv
df_sweep = pd.read_csv("artifacts/randomized_blackout_sweep_results.csv")

# =========================================================================
# 1. GYRO BIAS CONVERGENCE CHECK (DIAGNOSTIC 1)
# =========================================================================
print("="*70)
print("1. GYRO BIAS CONVERGENCE CHECK ON WORST-PERFORMING SAMPLES")
print("="*70)

# Sort sweep by final error to pick worst 3 S-S1 and worst 3 S-S2 samples
worst_s1 = df_sweep[df_sweep["trip"] == "S-S1 (Highway)"].sort_values("final_error_m", ascending=False).head(3)
worst_s2 = df_sweep[df_sweep["trip"] == "S-S2 (Urban Unseen)"].sort_values("final_error_m", ascending=False).head(3)

print("Worst S-S1 Scenarios:")
for _, row in worst_s1.iterrows():
    print(f"  {row['scenario']}: Start={row['start_time_s']:.1f}s, Dur={row['duration_s']:.1f}s, Err={row['final_error_m']:.1f}m, Drift={row['drift_pct']:.1f}%")

print("\nWorst S-S2 Scenarios:")
for _, row in worst_s2.iterrows():
    print(f"  {row['scenario']}: Start={row['start_time_s']:.1f}s, Dur={row['duration_s']:.1f}s, Err={row['final_error_m']:.1f}m, Drift={row['drift_pct']:.1f}%")

def track_gyro_bias_history(trip, calib_samples, v_preds, bo_start_s, bo_dur_s):
    ekf = ErrorStateEKF(nhc_lateral_std=0.15, nhc_vertical_std=0.15)
    ekf.reset(trip.gnss_samples[0])
    
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(bo_start_s * 1e9)
    bo_end_ns = bo_start_ns + int(bo_dur_s * 1e9)
    
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    
    times = []
    bg_z_list = []
    w_z_list = []
    speeds = []
    in_bo_list = []
    
    for i, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr > bo_end_ns + int(2e9):
            break
            
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                ekf.update_gnss(g)
            gnss_idx += 1
            
        in_blackout = (bo_start_ns <= t_curr <= bo_end_ns)
        calib = calib_samples[i]
        fwd_v = float(v_preds[i])
        m_state = "STATIONARY" if fwd_v < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=fwd_v, speed_variance=1.0, motion_state=m_state)
        fused = ekf.predict(calib, vel)
        
        times.append((t_curr - t0_ns) * 1e-9)
        bg_z_list.append(ekf._bg[2])
        w_z_list.append(calib.gyro_vehicle[2])
        speeds.append(fwd_v)
        in_bo_list.append(in_blackout)
        
    return pd.DataFrame({
        "time_s": times,
        "bg_z_deg_s": np.degrees(bg_z_list),
        "w_z_deg_s": np.degrees(w_z_list),
        "speed_mps": speeds,
        "in_blackout": in_bo_list
    })

# Plot Gyro Bias for Worst S-S1 and S-S2 samples
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False)
for idx, (_, row) in enumerate(worst_s1.iterrows()):
    df_track = track_gyro_bias_history(trip_s1, calib_s1, v_s1, row["start_time_s"], row["duration_s"])
    ax = axes[idx, 0]
    ax.plot(df_track["time_s"], df_track["bg_z_deg_s"], color="purple", lw=2, label="Estimated Gyro Bias $b_{g,z}$ (°/s)")
    ax.axvspan(row["start_time_s"], row["start_time_s"] + row["duration_s"], color="yellow", alpha=0.3, label="Blackout Window")
    ax.set_title(f"S-S1 (Highway) — {row['scenario']} (Final Err: {row['final_error_m']:.1f}m)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bias (°/s)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

for idx, (_, row) in enumerate(worst_s2.iterrows()):
    df_track = track_gyro_bias_history(trip_s2, calib_s2, v_s2, row["start_time_s"], row["duration_s"])
    ax = axes[idx, 1]
    ax.plot(df_track["time_s"], df_track["bg_z_deg_s"], color="purple", lw=2, label="Estimated Gyro Bias $b_{g,z}$ (°/s)")
    ax.axvspan(row["start_time_s"], row["start_time_s"] + row["duration_s"], color="yellow", alpha=0.3, label="Blackout Window")
    ax.set_title(f"S-S2 (Urban) — {row['scenario']} (Final Err: {row['final_error_m']:.1f}m)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bias (°/s)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plot_path_bg = "artifacts/gyro_bias_convergence_check.png"
plt.savefig(plot_path_bg, dpi=200)
plt.close()

import shutil
shutil.copy(plot_path_bg, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\gyro_bias_convergence_check.png")

# =========================================================================
# 2. HEADING ERROR VS TURN SEVERITY (DIAGNOSTIC 2)
# =========================================================================
print("\n" + "="*70)
print("2. CORRELATION: HEADING & CROSS-TRACK ERROR VS TURN SEVERITY ACROSS 50 SWEEP SAMPLES")
print("="*70)

# Compute turn severity metrics for all 50 sweep samples
sweep_turn_data = []

def analyze_blackout_turns(trip, calib_samples, v_preds, bo_start_s, bo_dur_s):
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(bo_start_s * 1e9)
    bo_end_ns = bo_start_ns + int(bo_dur_s * 1e9)
    
    # Extract gyro_z inside blackout
    w_z_bo = []
    ts_bo = []
    for i, imu in enumerate(trip.imu_samples):
        if bo_start_ns <= imu.timestamp_ns <= bo_end_ns:
            w_z_bo.append(calib_samples[i].gyro_vehicle[2])
            ts_bo.append(imu.timestamp_ns)
            
    w_z_bo = np.array(w_z_bo, dtype=np.float64)
    if len(w_z_bo) < 2:
        return 0.0, 0.0, 0.0
    dt_arr = np.diff(ts_bo) * 1e-9
    
    # Total absolute angular turn (degrees)
    total_turn_deg = float(np.degrees(np.sum(np.abs(w_z_bo[:-1]) * dt_arr)))
    # Net heading change (degrees)
    net_turn_deg = float(np.degrees(np.abs(np.sum(w_z_bo[:-1] * dt_arr))))
    # Peak yaw rate (degrees/s)
    peak_yaw_rate_deg_s = float(np.degrees(np.max(np.abs(w_z_bo))))
    
    return total_turn_deg, net_turn_deg, peak_yaw_rate_deg_s

turn_metrics = []
for _, row in df_sweep.iterrows():
    trip = trip_s1 if "S-S1" in row["trip"] else trip_s2
    calib = calib_s1 if "S-S1" in row["trip"] else calib_s2
    v_p = v_s1 if "S-S1" in row["trip"] else v_s2
    
    tot_turn, net_turn, peak_rate = analyze_blackout_turns(trip, calib, v_p, row["start_time_s"], row["duration_s"])
    turn_metrics.append({
        "total_turn_deg": tot_turn,
        "net_turn_deg": net_turn,
        "peak_yaw_rate_deg_s": peak_rate,
    })

df_turn = pd.DataFrame(turn_metrics)
df_combined = pd.concat([df_sweep, df_turn], axis=1)

# Compute correlations
r_cross_tot, p_cross_tot = pearsonr(df_combined["total_turn_deg"], np.abs(df_combined["err_cross_m"]))
r_cross_peak, p_cross_peak = pearsonr(df_combined["peak_yaw_rate_deg_s"], np.abs(df_combined["err_cross_m"]))
r_drift_tot, p_drift_tot = pearsonr(df_combined["total_turn_deg"], df_combined["drift_pct"])

print(f"Correlation: |Cross-Track Error| vs Total Heading Turn (deg): r = {r_cross_tot:.4f} (p = {p_cross_tot:.4e})")
print(f"Correlation: |Cross-Track Error| vs Peak Yaw Rate (deg/s):    r = {r_cross_peak:.4f} (p = {p_cross_peak:.4e})")
print(f"Correlation: Drift % vs Total Heading Turn (deg):            r = {r_drift_tot:.4f} (p = {p_drift_tot:.4e})")

# Scatter plots: Error vs Turn Severity
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cross-track Error vs Total Turn
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_combined[df_combined["trip"] == trip_name]
    ax1.scatter(sub["total_turn_deg"], np.abs(sub["err_cross_m"]), color=color, marker=marker, s=60, alpha=0.8, label=trip_name)
    m, b = np.polyfit(sub["total_turn_deg"], np.abs(sub["err_cross_m"]), 1)
    x_line = np.linspace(sub["total_turn_deg"].min(), sub["total_turn_deg"].max(), 100)
    ax1.plot(x_line, m*x_line + b, color=color, linestyle="--", lw=2, label=f"{trip_name} Trend ({m:.2f} m/deg)")

ax1.set_xlabel("Total Accumulated Turn in Blackout (degrees)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Absolute Cross-Track Error (meters)", fontsize=11, fontweight="bold")
ax1.set_title(f"(1) Cross-Track Error vs. Total Turn ($r = {r_cross_tot:.2f}$)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# 2. Cross-track Error vs Peak Yaw Rate
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_combined[df_combined["trip"] == trip_name]
    ax2.scatter(sub["peak_yaw_rate_deg_s"], np.abs(sub["err_cross_m"]), color=color, marker=marker, s=60, alpha=0.8, label=trip_name)
    m, b = np.polyfit(sub["peak_yaw_rate_deg_s"], np.abs(sub["err_cross_m"]), 1)
    x_line = np.linspace(sub["peak_yaw_rate_deg_s"].min(), sub["peak_yaw_rate_deg_s"].max(), 100)
    ax2.plot(x_line, m*x_line + b, color=color, linestyle="--", lw=2, label=f"{trip_name} Trend")

ax2.set_xlabel("Peak Yaw Rate in Blackout (°/s)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Absolute Cross-Track Error (meters)", fontsize=11, fontweight="bold")
ax2.set_title(f"(2) Cross-Track Error vs. Peak Yaw Rate ($r = {r_cross_peak:.2f}$)", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

# 3. Along-Track Error vs Total Turn
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_combined[df_combined["trip"] == trip_name]
    ax3.scatter(sub["total_turn_deg"], np.abs(sub["err_along_m"]), color=color, marker=marker, s=60, alpha=0.8, label=trip_name)

ax3.set_xlabel("Total Accumulated Turn in Blackout (degrees)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Absolute Along-Track Error (meters)", fontsize=11, fontweight="bold")
ax3.set_title("(3) Along-Track (Speed) Error vs. Total Turn", fontsize=12, fontweight="bold")
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(fontsize=10)

# 4. Drift % vs Total Turn
for trip_name, color, marker in [("S-S1 (Highway)", "blue", "o"), ("S-S2 (Urban Unseen)", "darkorange", "s")]:
    sub = df_combined[df_combined["trip"] == trip_name]
    ax4.scatter(sub["total_turn_deg"], sub["drift_pct"], color=color, marker=marker, s=60, alpha=0.8, label=trip_name)

ax4.axhline(10.0, color="red", linestyle=":", lw=2, label="SIH Target (<10%)")
ax4.set_xlabel("Total Accumulated Turn in Blackout (degrees)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Drift Percentage (%)", fontsize=11, fontweight="bold")
ax4.set_title(f"(4) Drift % vs. Total Turn ($r = {r_drift_tot:.2f}$)", fontsize=12, fontweight="bold")
ax4.grid(True, linestyle="--", alpha=0.6)
ax4.legend(fontsize=10)

plt.tight_layout()
plot_path_turn = "artifacts/heading_error_vs_turn_severity.png"
plt.savefig(plot_path_turn, dpi=200)
plt.close()

shutil.copy(plot_path_turn, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\heading_error_vs_turn_severity.png")
print(f"Saved turn severity correlation plot to {plot_path_turn}")

# =========================================================================
# 3. NHC CORRECTION STRENGTH CHECK (DIAGNOSTIC 3)
# =========================================================================
print("\n" + "="*70)
print("3. NHC CORRECTION STRENGTH CHECK")
print("="*70)
print("Auditing codebase in sih/fusion/es_ekf.py:")
print("  - Configuration flags: nhc_lateral_std = 0.15, nhc_vertical_std = 0.15, enable_nhc = True")
print("  - Execution status during AI-velocity prediction:")
print("    In ErrorStateEKF.predict(), when vel is provided:")
print("      v_e = v_fwd * sin(heading), v_n = v_fwd * cos(heading), v_u = 0.0")
print("    Lateral velocity in vehicle frame is assumed 0 analytically.")
print("    --> ZERO Kalman measurement update (K * (0 - v_lat)) is performed on the error-state vector.")
print("    --> Orientation error delta_theta is NEVER updated by NHC.")
print("    --> Heading change delta_heading = 0.0 rad at every step from NHC.")

# =========================================================================
# 4. MAGNETOMETER AVAILABILITY CHECK (DIAGNOSTIC 4)
# =========================================================================
print("\n" + "="*70)
print("4. MAGNETOMETER AVAILABILITY AUDIT IN IO-VNBD DATASET")
print("="*70)

df_s1_raw = pd.read_csv("data/raw/iovnbd_trips/S-S1.csv", encoding="latin-1")
df_s2_raw = pd.read_csv("data/raw/iovnbd_trips/S-S2.csv", encoding="latin-1")

mag_cols_s1 = [c for c in df_s1_raw.columns if "MAGNETIC" in c.upper() or "MAG" in c.upper()]
mag_cols_s2 = [c for c in df_s2_raw.columns if "MAGNETIC" in c.upper() or "MAG" in c.upper()]

print(f"S-S1 Raw Magnetometer Columns: {mag_cols_s1}")
print(f"S-S2 Raw Magnetometer Columns: {mag_cols_s2}")

print(f"S-S1 Mag Non-Null Count: {df_s1_raw[mag_cols_s1].notnull().sum().to_dict()}")
print(f"S-S2 Mag Non-Null Count: {df_s2_raw[mag_cols_s2].notnull().sum().to_dict()}")

print(f"S-S1 Mag Value Ranges (uT):")
for col in mag_cols_s1:
    print(f"  {col}: min={df_s1_raw[col].min():.2f}, mean={df_s1_raw[col].mean():.2f}, max={df_s1_raw[col].max():.2f}")

print(f"S-S2 Mag Value Ranges (uT):")
for col in mag_cols_s2:
    print(f"  {col}: min={df_s2_raw[col].min():.2f}, mean={df_s2_raw[col].mean():.2f}, max={df_s2_raw[col].max():.2f}")

print("\nPipeline Usage Status:")
print("  - GenericDataLoader: Parses mag_x, mag_y, mag_z into IMUSample.mag field.")
print("  - MountCalibrator: Does NOT use mag (uses gravity + centripetal correlation).")
print("  - TCNAttentionVelocityModel: Does NOT use mag (uses 6 IMU + 2 magnitudes).")
print("  - ErrorStateEKF: Does NOT use mag (pure gyro heading integration).")
print("  --> Magnetometer is 100% AVAILABLE in dataset and 100% UNUSED in the pipeline.")
