import os
import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_PATH   = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def get_trip_inferences(trip, device, model, ckpt):
    norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std  = ckpt.get("norm_std",  np.ones((8, 1),  dtype=np.float32))
    
    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples:
        calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    
    acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    
    N = len(feats)
    window_size = 100
    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w   = np.vstack([pad, feats[:i+1]]).T
        else:
            w   = feats[i - window_size + 1 : i + 1].T
        windows.append(w)
        
    windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
    preds = []
    with torch.no_grad():
        for b in range(0, N, 1024):
            x = torch.from_numpy(windows_norm[b : b + 1024]).to(device)
            p, _ = model(x)
            preds.extend(p.cpu().numpy().flatten())
    return calib_samples, np.array(preds, dtype=np.float32)

def run_single_eval(trip, calib_samples, v_preds, bo_start_s, bo_dur_s):
    ekf = ErrorStateEKF(enable_nhc=True, enable_zupt=True)
    ekf.init_from_gnss(trip.gnss_samples[0])
    
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(bo_start_s * 1e9)
    bo_end_ns   = bo_start_ns + int(bo_dur_s * 1e9)
    
    gnss_idx = 0
    n_gnss   = len(trip.gnss_samples)
    
    est_pts = []
    gt_pts  = []
    nhc_dthetas = []
    bg_z_history = []
    
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
            
        calib  = calib_samples[i]
        fwd_v  = float(v_preds[i])
        m_state = "STATIONARY" if fwd_v < 0.2 else "DRIVING"
        vel    = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=fwd_v, speed_variance=0.3, motion_state=m_state)
        fused  = ekf.predict(calib, vel)
        
        bg_z_history.append((float((t_curr - t0_ns) * 1e-9), float(ekf.b_g[2])))
        
        if bo_start_ns <= t_curr <= bo_end_ns:
            est_pts.append(fused.position_enu_m[:2])
            g_curr = trip.gnss_samples[min(gnss_idx, n_gnss-1)]
            g_enu  = geodetic_to_enu(
                g_curr.latitude_deg, g_curr.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )
            gt_pts.append(g_enu[:2])
            nhc_dthetas.append(ekf.last_nhc_dtheta_deg)
            
    est_pts = np.array(est_pts)
    gt_pts  = np.array(gt_pts)
    
    if len(gt_pts) < 2:
        return None
        
    dists    = np.sqrt(np.sum(np.diff(gt_pts, axis=0)**2, axis=1))
    tot_dist = max(float(np.sum(dists)), 1.0)
    
    err_vec   = est_pts[-1] - gt_pts[-1]
    final_err = float(np.linalg.norm(err_vec))
    drift_pct = (final_err / tot_dist) * 100.0
    
    fwd_dir = gt_pts[-1] - gt_pts[0]
    fwd_len = np.linalg.norm(fwd_dir)
    if fwd_len > 1.0:
        fwd_hat   = fwd_dir / fwd_len
        lat_hat   = np.array([-fwd_hat[1], fwd_hat[0]])
        err_along = float(np.dot(err_vec, fwd_hat))
        err_cross = float(np.dot(err_vec, lat_hat))
    else:
        err_along = float(err_vec[0])
        err_cross = float(err_vec[1])
        
    # Total turn angle
    pre_idx  = max(0, len(bg_z_history) - len(nhc_dthetas))
    gyro_z_vals = np.array([imu.gyro[2] for imu in trip.imu_samples[pre_idx:pre_idx+len(nhc_dthetas)]])
    dt = 0.01
    tot_turn_deg = float(np.sum(np.abs(gyro_z_vals)) * dt * 180.0 / np.pi)
    
    return {
        "final_error_m": final_err,
        "drift_pct":     drift_pct,
        "dist_m":        tot_dist,
        "err_along_m":   err_along,
        "err_cross_m":   err_cross,
        "tot_turn_deg":  tot_turn_deg,
        "nhc_dthetas":   nhc_dthetas,
        "bg_z_history":  bg_z_history,
        "bg_z_at_bo":    bg_z_history[pre_idx][1] if pre_idx < len(bg_z_history) else 0.0,
    }

def main():
    print("Loading datasets...")
    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model  = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("Precomputing inferences...")
    calib_s1, v_s1 = get_trip_inferences(trip_s1, device, model, ckpt)
    calib_s2, v_s2 = get_trip_inferences(trip_s2, device, model, ckpt)
    
    sweep_df = pd.read_csv(SWEEP_CSV)
    print(f"\nEvaluating {len(sweep_df)} scenarios with production fixed EKF...")
    
    results = []
    bg_z_traces = {}
    
    for i, row in sweep_df.iterrows():
        trip_label = row["trip"]
        if "S-S1" in trip_label:
            trip, calib, v_preds = trip_s1, calib_s1, v_s1
        else:
            trip, calib, v_preds = trip_s2, calib_s2, v_s2
            
        t_start  = float(row["start_time_s"])
        duration = float(row["duration_s"])
        
        res = run_single_eval(trip, calib, v_preds, t_start, duration)
        if res is None:
            continue
            
        results.append({
            "trip":           trip_label,
            "scenario":       row["scenario"],
            "start_time_s":   t_start,
            "duration_s":     duration,
            "distance_m":     res["dist_m"],
            "old_error_m":    row["final_error_m"],
            "old_drift_pct":  row["drift_pct"],
            "new_error_m":    res["final_error_m"],
            "new_drift_pct":  res["drift_pct"],
            "err_along_m":    res["err_along_m"],
            "err_cross_m":    res["err_cross_m"],
            "tot_turn_deg":   res["tot_turn_deg"],
            "bg_z_at_bo":     res["bg_z_at_bo"],
        })
        
        bg_z_traces[row["scenario"]] = res["bg_z_history"]
        
        d_err = res["final_error_m"] - row["final_error_m"]
        sign  = "+" if d_err > 0 else ""
        print(f"  [{i+1:2d}/{len(sweep_df)}] {row['scenario'][:35]:35s} old={row['final_error_m']:7.2f}m  new={res['final_error_m']:7.2f}m ({sign}{d_err:+.2f}m)")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(ARTIFACT_DIR, "fixed_sweep_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(ARTIFACT_DIR, 'fixed_sweep_results.csv')}")

    # Summary Table
    print("\n" + "=" * 80)
    print("50-SAMPLE RANDOMIZED SWEEP COMPARISON TABLE (Baseline vs Fixed EKF)")
    print("=" * 80)
    
    for tag in ["S-S1 (Highway)", "S-S2 (Urban Unseen)"]:
        sub = res_df[res_df["trip"] == tag]
        if sub.empty: continue
        label_str = tag.split("(")[0].strip()
        print(f"\n{label_str} Sub-Dataset (N={len(sub)}):")
        print(f"  {'Metric':<25} | {'Baseline (Old)':>16} | {'Fixed EKF (New)':>16} | {'Delta':>10}")
        print(f"  {'-'*25}-|-{'-'*16}-|-{'-'*16}-|-{'-'*10}")
        print(f"  {'Drift % (Mean)':<25} | {sub['old_drift_pct'].mean():16.2f}% | {sub['new_drift_pct'].mean():16.2f}% | {sub['new_drift_pct'].mean()-sub['old_drift_pct'].mean():+10.2f}%")
        print(f"  {'Drift % (Median)':<25} | {sub['old_drift_pct'].median():16.2f}% | {sub['new_drift_pct'].median():16.2f}% | {sub['new_drift_pct'].median()-sub['old_drift_pct'].median():+10.2f}%")
        print(f"  {'Drift % (90th %)':<25} | {sub['old_drift_pct'].quantile(0.9):16.2f}% | {sub['new_drift_pct'].quantile(0.9):16.2f}% | {sub['new_drift_pct'].quantile(0.9)-sub['old_drift_pct'].quantile(0.9):+10.2f}%")
        print(f"  {'Drift % (Worst)':<25} | {sub['old_drift_pct'].max():16.2f}% | {sub['new_drift_pct'].max():16.2f}% | {sub['new_drift_pct'].max()-sub['old_drift_pct'].max():+10.2f}%")
        print(f"  {'Abs Error m (Mean)':<25} | {sub['old_error_m'].mean():16.2f}m | {sub['new_error_m'].mean():16.2f}m | {sub['new_error_m'].mean()-sub['old_error_m'].mean():+10.2f}m")
        print(f"  {'Abs Error m (Median)':<25} | {sub['old_error_m'].median():16.2f}m | {sub['new_error_m'].median():16.2f}m | {sub['new_error_m'].median()-sub['old_error_m'].median():+10.2f}m")
        print(f"  {'Abs Error m (90th %)':<25} | {sub['old_error_m'].quantile(0.9):16.2f}m | {sub['new_error_m'].quantile(0.9):16.2f}m | {sub['new_error_m'].quantile(0.9)-sub['old_error_m'].quantile(0.9):+10.2f}m")
        print(f"  {'Abs Error m (Worst)':<25} | {sub['old_error_m'].max():16.2f}m | {sub['new_error_m'].max():16.2f}m | {sub['new_error_m'].max()-sub['old_error_m'].max():+10.2f}m")

    # Combined Table
    print(f"\nCOMBINED ALL TRIPS (N={len(res_df)}):")
    print(f"  {'Metric':<25} | {'Baseline (Old)':>16} | {'Fixed EKF (New)':>16} | {'Delta':>10}")
    print(f"  {'-'*25}-|-{'-'*16}-|-{'-'*16}-|-{'-'*10}")
    print(f"  {'Drift % (Mean)':<25} | {res_df['old_drift_pct'].mean():16.2f}% | {res_df['new_drift_pct'].mean():16.2f}% | {res_df['new_drift_pct'].mean()-res_df['old_drift_pct'].mean():+10.2f}%")
    print(f"  {'Drift % (Median)':<25} | {res_df['old_drift_pct'].median():16.2f}% | {res_df['new_drift_pct'].median():16.2f}% | {res_df['new_drift_pct'].median()-res_df['old_drift_pct'].median():+10.2f}%")
    print(f"  {'Drift % (90th %)':<25} | {res_df['old_drift_pct'].quantile(0.9):16.2f}% | {res_df['new_drift_pct'].quantile(0.9):16.2f}% | {res_df['new_drift_pct'].quantile(0.9)-res_df['old_drift_pct'].quantile(0.9):+10.2f}%")
    print(f"  {'Drift % (Worst)':<25} | {res_df['old_drift_pct'].max():16.2f}% | {res_df['new_drift_pct'].max():16.2f}% | {res_df['new_drift_pct'].max()-res_df['old_drift_pct'].max():+10.2f}%")

    # ── PLOT 1: Gyro Bias Convergence Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    # Plot S-S1 full trip bg_z
    s1_hist = list(bg_z_traces.values())[0] # first scenario trace
    times_s1, bg_s1 = zip(*s1_hist)
    ax1.plot(times_s1, np.degrees(bg_s1), color="#3A7BD5", lw=1.2, label="S-S1 Gyro Bias bg_z (deg/s)")
    ax1.axhline(0, color="gray", ls="--", alpha=0.5)
    ax1.set_ylabel("bg_z (°/s)", fontsize=11)
    ax1.set_title("Gyro Bias Estimation (b_{g,z}) Over Time — S-S1 (Highway)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot S-S2 full trip bg_z
    s2_hist = [t for sc, t in bg_z_traces.items() if "S-S2" in sc or "Urban" in sc]
    if s2_hist:
        times_s2, bg_s2 = zip(*s2_hist[0])
        ax2.plot(times_s2, np.degrees(bg_s2), color="#D55A3A", lw=1.2, label="S-S2 Gyro Bias bg_z (deg/s)")
        ax2.axhline(0, color="gray", ls="--", alpha=0.5)
        ax2.set_xlabel("Time Since Trip Start (s)", fontsize=11)
        ax2.set_ylabel("bg_z (°/s)", fontsize=11)
        ax2.set_title("Gyro Bias Estimation (b_{g,z}) Over Time — S-S2 (Urban Unseen)", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    bias_png = os.path.join(ARTIFACT_DIR, "fixed_gyro_bias_convergence.png")
    fig.savefig(bias_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {bias_png}")

    # ── PLOT 2: Turn Severity Correlation Scatter Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = {"S-S1 (Highway)": "#3A7BD5", "S-S2 (Urban Unseen)": "#D55A3A"}
    for trip_label, grp in res_df.groupby("trip"):
        c  = cmap.get(trip_label, "gray")
        ls = trip_label.split("(")[0].strip()
        ax1.scatter(grp["tot_turn_deg"], grp["old_error_m"].abs(), c=c, alpha=0.7, edgecolors="white", lw=0.5, label=ls, s=60)
        ax2.scatter(grp["tot_turn_deg"], grp["new_error_m"].abs(), c=c, alpha=0.7, edgecolors="white", lw=0.5, label=ls, s=60)

    # Fits
    r_old, p_old = pearsonr(res_df["tot_turn_deg"], res_df["old_error_m"].abs())
    m_old, b_old = np.polyfit(res_df["tot_turn_deg"], res_df["old_error_m"].abs(), 1)

    r_new, p_new = pearsonr(res_df["tot_turn_deg"], res_df["new_error_m"].abs())
    m_new, b_new = np.polyfit(res_df["tot_turn_deg"], res_df["new_error_m"].abs(), 1)

    xs = np.linspace(res_df["tot_turn_deg"].min(), res_df["tot_turn_deg"].max(), 100)
    ax1.plot(xs, m_old * xs + b_old, "k--", lw=1.5, label=f"r = {r_old:.3f} (p = {p_old:.4f})")
    ax1.set_xlabel("Total Turn Angle During Blackout (°)", fontsize=11)
    ax1.set_ylabel("Final Position Error (m)", fontsize=11)
    ax1.set_title(f"BEFORE Fix (Baseline)\nr = {r_old:.3f}, p = {p_old:.4f}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, m_new * xs + b_new, "k--", lw=1.5, label=f"r = {r_new:.3f} (p = {p_new:.4f})")
    ax2.set_xlabel("Total Turn Angle During Blackout (°)", fontsize=11)
    ax2.set_ylabel("Final Position Error (m)", fontsize=11)
    ax2.set_title(f"AFTER Fix (Closed-Loop EKF)\nr = {r_new:.3f}, p = {p_new:.4f}", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cross-Track Error vs Total Turn Severity — Before vs After Fix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    corr_png = os.path.join(ARTIFACT_DIR, "fixed_heading_error_vs_turn_severity.png")
    fig.savefig(corr_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {corr_png}")
    print("\nSweep Complete!")

if __name__ == "__main__":
    main()
