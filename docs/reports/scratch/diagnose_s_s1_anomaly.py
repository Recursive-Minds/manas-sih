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

def run_and_log_convergence(trip, blackout_cfg):
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
    
    # Pre-compute ground truth ENU
    gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
    gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples], dtype=np.float64)
    from sih.data.geo import geodetic_to_enu
    gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)
    
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    
    records = []
    
    for imu in trip.imu_samples:
        t_curr = imu.timestamp_ns
        t_s = (t_curr - t0_ns) * 1e-9
        
        # Stop 5s after blackout
        if t_curr > bo_end_ns + int(5e9):
            break
            
        # Process GNSS
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                runner.pipeline.process_gnss(g)
            gnss_idx += 1
            
        calib, vel, fused, matched = runner.pipeline.process_imu(imu)
        
        gt_e = np.interp(t_curr, gnss_ts, gnss_enu[:, 0])
        gt_n = np.interp(t_curr, gnss_ts, gnss_enu[:, 1])
        err_m = float(np.sqrt((fused.position_enu_m[0] - gt_e)**2 + (fused.position_enu_m[1] - gt_n)**2))
        
        in_blackout = (bo_start_ns <= t_curr <= bo_end_ns)
        
        # Extract calibration state
        is_aligned = runner.pipeline.calibration.is_aligned()
        yaw_axis = getattr(runner.pipeline.calibration, "_yaw_axis", None)
        yaw_sign = getattr(runner.pipeline.calibration, "_yaw_sign", 1.0)
        
        # Extract speed scale and gyro bias from fusion filter
        s_v = runner.pipeline.fusion_filter._speed_scale
        bg_z = runner.pipeline.fusion_filter._bg[2]
        
        records.append({
            "time_s": t_s,
            "timestamp_ns": t_curr,
            "in_blackout": in_blackout,
            "error_m": err_m,
            "est_speed": vel.forward_speed_mps if vel is not None else 0.0,
            "is_aligned": is_aligned,
            "yaw_axis": yaw_axis if yaw_axis is not None else -1,
            "yaw_sign": yaw_sign,
            "speed_scale": s_v,
            "gyro_bias_z": bg_z,
            "heading_deg": np.degrees(fused.heading_rad),
        })
        
    return pd.DataFrame(records)

print("Running Convergence Tracking on S-S1...")
trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

sc30 = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_blackout_at_120s")
sc60 = BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_blackout_at_300s")

df_sc30 = run_and_log_convergence(trip, sc30)
df_sc60 = run_and_log_convergence(trip, sc60)

# Create 2x3 comparison plot
fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex='col')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

datasets = [(df_sc30, "Scenario A: 30s Blackout @ 120s (Drift 13.72%)", 120.0, 150.0, 0),
            (df_sc60, "Scenario B: 60s Blackout @ 300s (Drift 8.69% - PASSED)", 300.0, 360.0, 1)]

for df, title, bo_start, bo_end, col in datasets:
    # 1. Error vs Time
    ax = axes[0, col]
    ax.plot(df["time_s"], df["error_m"], color="red", lw=2, label="Horizontal Error (m)")
    ax.axvspan(bo_start, bo_end, color="yellow", alpha=0.3, label="Blackout Window")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Error (m)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    # 2. Vehicle Speed vs Time
    ax = axes[1, col]
    ax.plot(df["time_s"], df["est_speed"], color="blue", lw=1.5, label="Estimated Speed (m/s)")
    ax.axvspan(bo_start, bo_end, color="yellow", alpha=0.3)
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    # 3. Online Speed Scale Factor s_v(t)
    ax = axes[2, col]
    ax.plot(df["time_s"], df["speed_scale"], color="green", lw=2, label="Scale Factor s_v")
    ax.axvspan(bo_start, bo_end, color="yellow", alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", label="Nominal (1.0)")
    ax.set_ylabel("Speed Scale s_v")
    ax.set_ylim(0.8, 1.3)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    # 4. Gyro Bias State b_g,z(t)
    ax = axes[3, col]
    ax.plot(df["time_s"], np.degrees(df["gyro_bias_z"]), color="purple", lw=2, label="Gyro Bias b_g,z (°/s)")
    ax.axvspan(bo_start, bo_end, color="yellow", alpha=0.3)
    ax.axhline(0.0, color="gray", linestyle=":")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Bias (°/s)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plot_path = "artifacts/s_s1_anomaly_diagnosis.png"
plt.savefig(plot_path, dpi=200)
plt.close()

# Also copy to brain artifact directory
import shutil
shutil.copy(plot_path, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\s_s1_anomaly_diagnosis.png")

print(f"\nSaved diagnosis plot to {plot_path}")

# Print state snapshot at blackout starts
idx30_start = np.searchsorted(df_sc30["time_s"], 120.0)
idx60_start = np.searchsorted(df_sc60["time_s"], 300.0)

print("\n--- PRE-BLACKOUT CONVERGENCE COMPARISON ---")
print(f"Scenario 1 (t = 120.0s):")
print(f"  Driving duration before blackout: 120.0 s (Vehicle actually moving since t=50s -> 70s active driving)")
print(f"  Alignment status:                 Aligned={df_sc30['is_aligned'].iloc[idx30_start]}, Yaw Axis={df_sc30['yaw_axis'].iloc[idx30_start]}, Sign={df_sc30['yaw_sign'].iloc[idx30_start]}")
print(f"  Online Speed Scale s_v:           {df_sc30['speed_scale'].iloc[idx30_start]:.4f}")
print(f"  Gyro Bias b_g,z:                  {np.degrees(df_sc30['gyro_bias_z'].iloc[idx30_start]):.4f} deg/s")
print(f"  Blackout Speed:                   Mean={df_sc30[df_sc30['in_blackout']]['est_speed'].mean():.2f} m/s, Max={df_sc30[df_sc30['in_blackout']]['est_speed'].max():.2f} m/s")
print(f"  Blackout Distance:                510.3 m")
print(f"  Final Error:                      {df_sc30['error_m'].iloc[-1]:.2f} m (Drift: 13.72%)")

print(f"\nScenario 2 (t = 300.0s):")
print(f"  Driving duration before blackout: 300.0 s (250s active driving with multiple turns and speed changes)")
print(f"  Alignment status:                 Aligned={df_sc60['is_aligned'].iloc[idx60_start]}, Yaw Axis={df_sc60['yaw_axis'].iloc[idx60_start]}, Sign={df_sc60['yaw_sign'].iloc[idx60_start]}")
print(f"  Online Speed Scale s_v:           {df_sc60['speed_scale'].iloc[idx60_start]:.4f}")
print(f"  Gyro Bias b_g,z:                  {np.degrees(df_sc60['gyro_bias_z'].iloc[idx60_start]):.4f} deg/s")
print(f"  Blackout Speed:                   Mean={df_sc60[df_sc60['in_blackout']]['est_speed'].mean():.2f} m/s, Max={df_sc60[df_sc60['in_blackout']]['est_speed'].max():.2f} m/s")
print(f"  Blackout Distance:                814.1 m")
print(f"  Final Error:                      {df_sc60['error_m'].iloc[-1]:.2f} m (Drift: 8.69%)")
