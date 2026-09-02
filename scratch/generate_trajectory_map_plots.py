import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92"
MODEL_PATH   = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Trajectory Map Plots on {device}...", flush=True)

    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")

    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model  = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    def get_inferences(trip):
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
        norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
        norm_std  = ckpt.get("norm_std",  np.ones((8, 1),  dtype=np.float32))
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
            for b in range(0, N, 2048):
                x = torch.from_numpy(windows_norm[b : b + 2048]).to(device)
                p, _ = model(x)
                preds.extend(p.cpu().numpy().flatten())
        return calib_samples, np.array(preds, dtype=np.float32)

    calib_s1, v_preds_s1 = get_inferences(trip_s1)
    calib_s2, v_preds_s2 = get_inferences(trip_s2)

    def generate_map_plot(trip, calib_samples, v_preds, t_start_s, duration_s, title, out_filename):
        t0_ns = trip.imu_samples[0].timestamp_ns
        bo_start_ns = t0_ns + int(t_start_s * 1e9)
        bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

        # Ground truth ENU trajectory
        gt_pts = []
        gt_ts  = []
        for g in trip.gnss_samples:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
            gt_pts.append(enu)
            gt_ts.append(g.timestamp_ns)
        gt_pts = np.array(gt_pts)
        gt_ts  = np.array(gt_ts)

        # Blackout GT segment
        gt_bo_mask = (gt_ts >= bo_start_ns) & (gt_ts <= bo_end_ns)
        gt_bo_pts  = gt_pts[gt_bo_mask]

        # EKF Execution
        warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
        warmup_gnss = min([g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=trip.gnss_samples[0])

        ekf = ErrorStateEKF(
            turn_threshold_rad_s=np.radians(1.5),
            cooldown_duration_s=0.5,
            max_gyro_bias_rad_s=np.radians(0.5),
            initial_speed_scale=1.00
        )
        ekf.init_from_gnss(warmup_gnss)

        gnss_idx = 0
        n_gnss   = len(trip.gnss_samples)

        pre_bo_est = []
        bo_est     = []

        for j, imu in enumerate(trip.imu_samples):
            t_curr = imu.timestamp_ns
            if t_curr < warmup_start_ns: continue
            if t_curr > bo_end_ns + int(5.0 * 1e9): break

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

            pos = fused.position_enu_m[:2]
            if t_curr < bo_start_ns:
                pre_bo_est.append(pos)
            elif bo_start_ns <= t_curr <= bo_end_ns:
                bo_est.append(pos)

        pre_bo_est = np.array(pre_bo_est)
        bo_est     = np.array(bo_est)

        # Plotting
        plt.figure(figsize=(10, 8))

        # Ground truth path
        plt.plot(gt_pts[:, 0], gt_pts[:, 1], color="#3498db", linestyle="-", linewidth=2.5, label="Ground Truth Path (GNSS)", zorder=1)
        plt.scatter(gt_pts[::10, 0], gt_pts[::10, 1], color="#2980b9", s=20, label="GNSS Fixes (Blue Dots)", zorder=2)

        # Pre-blackout estimated path
        if len(pre_bo_est) > 0:
            plt.plot(pre_bo_est[:, 0], pre_bo_est[:, 1], color="#2ecc71", linestyle="--", linewidth=2.0, label="EKF Pre-Blackout Fix", zorder=3)

        # Blackout outage region (Black dots)
        if len(gt_bo_pts) > 0:
            plt.scatter(gt_bo_pts[:, 0], gt_bo_pts[:, 1], color="black", s=45, label="GNSS Blackout Outage Window (Black Dots)", zorder=4)

        # EKF Estimated Path during Blackout (Red line & red dots)
        if len(bo_est) > 0:
            plt.plot(bo_est[:, 0], bo_est[:, 1], color="#e74c3c", linestyle="-", linewidth=3.0, label="ES-EKF Dead Reckoning Path (Red Line)", zorder=5)
            plt.scatter(bo_est[::15, 0], bo_est[::15, 1], color="#c0392b", s=35, label="Dead Reckoning Fixes (Red Dots)", zorder=6)

            # Start and End Markers
            plt.scatter(bo_est[0, 0], bo_est[0, 1], color="green", marker="^", s=120, label="Blackout Start Point", zorder=7)
            plt.scatter(bo_est[-1, 0], bo_est[-1, 1], color="red", marker="X", s=140, label="Blackout End (Estimated)", zorder=7)

        if len(gt_bo_pts) > 0:
            plt.scatter(gt_bo_pts[-1, 0], gt_bo_pts[-1, 1], color="blue", marker="P", s=140, label="Blackout End (Ground Truth)", zorder=7)

            # Draw drift error vector
            plt.plot([bo_est[-1, 0], gt_bo_pts[-1, 0]], [bo_est[-1, 1], gt_bo_pts[-1, 1]], color="purple", linestyle=":", linewidth=2.5, label="Final Position Error", zorder=8)

        plt.title(title, fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("East Position ENU (meters)", fontsize=11, fontweight="bold")
        plt.ylabel("North Position ENU (meters)", fontsize=11, fontweight="bold")
        plt.legend(loc="best", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)
        plt.axis("equal")
        plt.tight_layout()

        out_path = os.path.join(ARTIFACT_DIR, out_filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated Map Plot: {out_path}", flush=True)

    # 1. S-S1 60s blackout
    generate_map_plot(
        trip_s1, calib_s1, v_preds_s1, t_start_s=300.0, duration_s=60.0,
        title="Trip S-S1: 60s GNSS Blackout Outage Trajectory (Blue GT vs Red EKF vs Black Outage)",
        out_filename="map_s_s1_60s_blackout.png"
    )

    # 2. S-S1 30s blackout
    generate_map_plot(
        trip_s1, calib_s1, v_preds_s1, t_start_s=120.0, duration_s=30.0,
        title="Trip S-S1: 30s GNSS Blackout Outage Trajectory (Blue GT vs Red EKF vs Black Outage)",
        out_filename="map_s_s1_30s_blackout.png"
    )

    # 3. S-S2 30s blackout (Unseen validation trip)
    generate_map_plot(
        trip_s2, calib_s2, v_preds_s2, t_start_s=120.0, duration_s=30.0,
        title="Trip S-S2 (Unseen Validation): 30s GNSS Blackout Trajectory",
        out_filename="map_s_s2_30s_blackout.png"
    )

    # 4. Worst-case Scenario 2 (rand_bo_02_34s_at_4810s)
    generate_map_plot(
        trip_s1, calib_s1, v_preds_s1, t_start_s=4810.0, duration_s=34.0,
        title="Worst-Case Outage (rand_bo_02): 34s GNSS Blackout (Reduced from 806% to 177% Drift)",
        out_filename="map_worst_case_rand_bo_02.png"
    )

if __name__ == "__main__":
    main()
