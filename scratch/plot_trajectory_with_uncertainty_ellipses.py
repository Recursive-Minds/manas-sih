import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
MODEL_V_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def plot_scenario_with_ellipses(trip_path, t_start_s, duration_s, title, out_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)

    ckpt_v = torch.load(MODEL_V_PATH, map_location=device, weights_only=False)
    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device)
    model_v.eval()

    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples: calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    N = len(feats)
    window_size = 100

    norm_mean = ckpt_v.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std  = ckpt_v.get("norm_std",  np.ones((8, 1),  dtype=np.float32))

    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w   = np.vstack([pad, feats[:i+1]]).T
        else:
            w   = feats[i - window_size + 1 : i + 1].T
        windows.append((w - norm_mean) / norm_std)

    preds_v = []
    with torch.no_grad():
        for b in range(0, N, 2048):
            xv = torch.from_numpy(np.squeeze(np.array(windows[b : b + 2048], dtype=np.float32), axis=1) if len(np.array(windows[b : b + 2048]).shape) == 4 else np.array(windows[b : b + 2048], dtype=np.float32)).to(device)
            pv, _ = model_v(xv)
            preds_v.extend(pv.cpu().numpy().flatten())
    v_preds = np.array(preds_v, dtype=np.float32)

    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(t_start_s * 1e9)
    bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

    gt_pts = []
    gt_ts  = []
    for g in trip.gnss_samples:
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        gt_pts.append(enu)
        gt_ts.append(g.timestamp_ns)
    gt_pts = np.array(gt_pts)
    gt_ts  = np.array(gt_ts)

    gt_bo_mask = (gt_ts >= bo_start_ns) & (gt_ts <= bo_end_ns)
    gt_bo_pts  = gt_pts[gt_bo_mask]

    warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
    warmup_gnss = min([g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=trip.gnss_samples[0])

    ekf = ErrorStateEKF(
        turn_threshold_rad_s=np.radians(1.5),
        cooldown_duration_s=0.5,
        max_gyro_bias_rad_s=np.radians(0.1),
        initial_speed_scale=1.00
    )
    ekf.init_from_gnss(warmup_gnss)

    gnss_idx = 0
    n_gnss   = len(trip.gnss_samples)
    est_pts  = []
    ellipses_data = []

    last_ellipse_t_s = -100.0

    for j, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr < warmup_start_ns: continue
        if t_curr > bo_end_ns + int(1e9): break

        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            if g.timestamp_ns < bo_start_ns:
                ekf.update_gnss(g)
                if g.speed_mps is not None and g.speed_mps > 3.0 and g.bearing_deg is not None:
                    ekf._heading_rad = float(np.radians(g.bearing_deg))
            gnss_idx += 1

        cal = calib_samples[j]
        v_fwd = float(v_preds[j])

        m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)
        fused = ekf.predict(cal, vel)

        if bo_start_ns <= t_curr <= bo_end_ns:
            p_pos = fused.position_enu_m[:2]
            est_pts.append(p_pos)

            t_rel_s = (t_curr - bo_start_ns) / 1e9
            if t_rel_s - last_ellipse_t_s >= 8.0 or j == len(trip.imu_samples) - 1:
                last_ellipse_t_s = t_rel_s
                # Extract 2x2 position covariance in East-North
                cov_2d = ekf._P[0:2, 0:2]
                vals, vecs = np.linalg.eigh(cov_2d)
                # 2-sigma 95% confidence bounds
                width  = 2.0 * 2.0 * np.sqrt(max(vals[1], 1e-4)) # major axis diameter
                height = 2.0 * 2.0 * np.sqrt(max(vals[0], 1e-4)) # minor axis diameter
                angle_deg = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
                ellipses_data.append((p_pos[0], p_pos[1], width, height, angle_deg, t_rel_s))

    est_pts = np.array(est_pts)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot Ground Truth (Black)
    if len(gt_bo_pts) > 0:
        ax.plot(gt_bo_pts[:, 0], gt_bo_pts[:, 1], 'k-', linewidth=3.0, label="Ground Truth (GNSS)", zorder=4)
        ax.scatter(gt_bo_pts[0, 0], gt_bo_pts[0, 1], color='black', s=130, zorder=6, label="Blackout Entry")
        ax.scatter(gt_bo_pts[-1, 0], gt_bo_pts[-1, 1], color='black', marker='X', s=150, zorder=6, label="Blackout Exit")

    # Plot EKF Dead Reckoning Track (Blue)
    if len(est_pts) > 0:
        ax.plot(est_pts[:, 0], est_pts[:, 1], color='#0275d8', linewidth=2.5, linestyle='--', label="Our 15-State ES-EKF Track", zorder=5)
        ax.scatter(est_pts[-1, 0], est_pts[-1, 1], color='#0275d8', marker='o', s=120, zorder=7, label="Estimated Exit Position")

    # Render dynamic uncertainty ellipses
    for (ex, ey, ew, eh, eang, t_s) in ellipses_data:
        ell = Ellipse(xy=(ex, ey), width=ew, height=eh, angle=eang,
                      edgecolor='#0275d8', facecolor='#0275d8', alpha=0.12, linewidth=1.5, linestyle=':', zorder=3)
        ax.add_patch(ell)
        ax.text(ex, ey, f"+{t_s:.0f}s", fontsize=8, color='#004085', fontweight='bold', ha='center', va='center', zorder=5)

    # Dummy ellipse for legend
    dummy_ell = Ellipse(xy=(0,0), width=1, height=1, edgecolor='#0275d8', facecolor='#0275d8', alpha=0.25, linewidth=1.5, linestyle=':', label="Live 95% Confidence Ellipse (2σ)")
    ax.add_patch(dummy_ell)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("East Position (meters)", fontsize=12)
    ax.set_ylabel("North Position (meters)", fontsize=12)
    ax.legend(loc="best", fontsize=11, frameon=True, facecolor='white', framealpha=0.95)
    ax.axis("equal")
    plt.tight_layout()

    out_path = os.path.join(ARTIFACT_DIR, out_filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved uncertainty ellipse plot to {out_path}", flush=True)

def main():
    print("=== GENERATING REAL-TIME UNCERTAINTY ELLIPSE PLOTS ===", flush=True)
    # 1. Highway 75s blackout (S-S1)
    plot_scenario_with_ellipses(
        trip_path=r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv",
        t_start_s=1956.0, duration_s=75.0,
        title="SIH Dead Reckoning with Dynamic 95% Confidence Ellipse (Highway 75s Outage)",
        out_filename="trajectory_with_uncertainty_ellipses_highway.png"
    )
    # 2. Urban turn 45s blackout (S-S2)
    plot_scenario_with_ellipses(
        trip_path=r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv",
        t_start_s=8303.0, duration_s=45.0,
        title="SIH Dead Reckoning with Dynamic 95% Confidence Ellipse (Urban 45s Outage)",
        out_filename="trajectory_with_uncertainty_ellipses_urban.png"
    )

if __name__ == "__main__":
    main()
