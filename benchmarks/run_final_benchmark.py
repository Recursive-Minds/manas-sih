"""
Comprehensive Multi-Stage IDR Benchmark Suite.
Evaluates all 4 architectural generations across the 50 standardized blackout scenarios:
  1. Stage 0: Naive Baseline (Open-loop double-integration of raw accelerometer & gyro)
  2. Stage 1 (Phase 2): Classical 15-State ES-EKF + Non-Holonomic Constraints (NHC) (No AI)
  3. Stage 2 (Phase 3): Production 15-State Unified ES-EKF + AI Velocity Fusion (TCN-Attention)
  4. Stage 3 (Phase 4): Phase 4 Map-Matched ES-EKF (Tightly-coupled road network constraints)
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.fusion.es_ekf import ErrorStateEKF
from sih.fusion.naive import NaiveDeadReckoningFilter
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.map.network import RoadNetwork
from scipy.spatial.transform import Rotation as R

SWEEP_CSV = r"C:\Users\carpe\SIH\artifacts\randomized_blackout_sweep_results.csv"
MODEL_PATH = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"
ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"


def run_multi_stage_benchmark():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print("             SMARTPHONE INTELLIGENT DEAD RECKONING (SIH)             ", flush=True)
    print("          COMPREHENSIVE 4-STAGE ARCHITECTURE BENCHMARK SWEEP         ", flush=True)
    print("=" * 85, flush=True)
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load Trips
    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")

    # 2. Build Road Networks for Map-Matching
    print("\nBuilding Topological Road Networks for Phase 4...", flush=True)
    valid_g_s1 = [g for g in trip_s1.gnss_samples if g.is_valid]
    pts_enu_s1 = np.array([geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip_s1.reference_lat_deg, trip_s1.reference_lon_deg, 0.0)[:2] for g in valid_g_s1])
    pts_ll_s1 = np.array([[g.latitude_deg, g.longitude_deg] for g in valid_g_s1])
    road_net_s1 = RoadNetwork.from_polyline_coords(pts_enu_s1, pts_ll_s1, "hwy_s1", 100.0)

    valid_g_s2 = [g for g in trip_s2.gnss_samples if g.is_valid]
    pts_enu_s2 = np.array([geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip_s2.reference_lat_deg, trip_s2.reference_lon_deg, 0.0)[:2] for g in valid_g_s2])
    pts_ll_s2 = np.array([[g.latitude_deg, g.longitude_deg] for g in valid_g_s2])
    road_net_s2 = RoadNetwork.from_polyline_coords(pts_enu_s2, pts_ll_s2, "urb_s2", 100.0)

    # 3. Precompute AI Model Inferences
    print("Precomputing Deep Temporal Attention Velocity on GPU...", flush=True)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    def get_calib_and_inferences(trip):
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
        norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
        norm_std = ckpt.get("norm_std", np.ones((8, 1), dtype=np.float32))
        for i in range(N):
            if i < window_size:
                pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
                w = np.vstack([pad, feats[:i + 1]]).T
            else:
                w = feats[i - window_size + 1 : i + 1].T
            windows.append(w)
        windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
        preds = []
        with torch.no_grad():
            for b in range(0, N, 2048):
                x = torch.from_numpy(windows_norm[b : b + 2048]).to(device)
                p, _ = model(x)
                preds.extend(p.cpu().numpy().flatten())
        return calib_samples, np.array(preds, dtype=np.float32)

    calib_s1, v_s1 = get_calib_and_inferences(trip_s1)
    calib_s2, v_s2 = get_calib_and_inferences(trip_s2)

    # 4. Load Scenarios
    sweep_df = pd.read_csv(SWEEP_CSV)

    def eval_trip_all_stages(trip, calib, v_preds, road_net, sub_df, offset_idx=0):
        results = []
        for idx, (_, row) in enumerate(sub_df.iterrows()):
            t_start = float(row["start_time_s"])
            duration = float(row["duration_s"])
            dist_m = float(row["distance_m"])

            t0_ns = trip.imu_samples[0].timestamp_ns
            bo_start_ns = t0_ns + int(t_start * 1e9)
            bo_end_ns = bo_start_ns + int(duration * 1e9)

            valid_g = [g for g in trip.gnss_samples if g.is_valid]
            gt_start_sample = min(valid_g, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
            gt_end_sample = min(valid_g, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

            gt_start_enu = geodetic_to_enu(
                gt_start_sample.latitude_deg, gt_start_sample.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )[:2]
            gt_end_enu = geodetic_to_enu(
                gt_end_sample.latitude_deg, gt_end_sample.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )[:2]
            gt_disp = gt_end_enu - gt_start_enu

            # Instantiate All 4 Filter Stages
            # Stage 0: Naive Baseline
            naive = NaiveDeadReckoningFilter()
            naive.reset(gt_start_sample)
            naive.ref_lat_deg = trip.reference_lat_deg
            naive.ref_lon_deg = trip.reference_lon_deg
            naive.pos_enu[:2] = gt_start_enu
            if gt_start_sample.bearing_deg is not None:
                b_rad = np.radians(gt_start_sample.bearing_deg)
                naive.rot_body_to_enu = R.from_euler("z", (np.pi / 2.0) - b_rad)
                v_mag = gt_start_sample.speed_mps or 0.0
                naive.vel_enu = np.array([v_mag * np.sin(b_rad), v_mag * np.cos(b_rad), 0.0])

            # Stage 1: Phase 2 ES-EKF + NHC (No AI)
            ekf_p2 = ErrorStateEKF(
                turn_threshold_rad_s=np.radians(1.5),
                cooldown_duration_s=0.5,
                max_gyro_bias_rad_s=np.radians(0.5),
                initial_speed_scale=1.00,
            )
            ekf_p2.init_from_gnss(gt_start_sample, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)

            # Stage 2: Phase 3 ES-EKF + AI Velocity Fusion
            ekf_p3 = ErrorStateEKF(
                turn_threshold_rad_s=np.radians(1.5),
                cooldown_duration_s=0.5,
                max_gyro_bias_rad_s=np.radians(0.5),
                initial_speed_scale=1.00,
            )
            ekf_p3.init_from_gnss(gt_start_sample, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)

            # Stage 3: Phase 4 Map-Matched ES-EKF
            ekf_p4 = ErrorStateEKF(
                turn_threshold_rad_s=np.radians(1.5),
                cooldown_duration_s=0.5,
                max_gyro_bias_rad_s=np.radians(0.5),
                initial_speed_scale=1.00,
            )
            ekf_p4.init_from_gnss(gt_start_sample, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)

            # Robust multi-epoch heading seeding over pre-blackout window
            pre_gnss_window = [g for g in valid_g if bo_start_ns - int(5.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
            init_road_bearing = None
            init_cands = road_net.find_candidates(gt_start_enu, radius_m=35.0)
            for s in init_cands:
                proj, d_p, _ = s.project_point(gt_start_enu)
                if d_p < 20.0:
                    init_road_bearing = s.bearing_deg
                    break
            ekf_p4.seed_pre_blackout_heading(pre_gnss_window, road_bearing_deg=init_road_bearing)

            v_last_gnss = max(gt_start_sample.speed_mps or 10.0, 1.0)
            pts_naive, pts_p2, pts_p3, pts_p4 = [], [], [], []

            for j, imu in enumerate(trip.imu_samples):
                t_curr = imu.timestamp_ns
                if t_curr < bo_start_ns:
                    continue
                if t_curr > bo_end_ns:
                    break

                cal = calib[j]
                v_ai = float(v_preds[j])
                m_state = "STATIONARY" if v_ai < 0.2 else "DRIVING"

                # 1. Stage 0: Naive Baseline
                fn = naive.predict(cal, VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=0.0, speed_variance=1.0, motion_state="UNKNOWN"))
                pts_naive.append(fn.position_enu_m[:2])

                # 2. Stage 1: Phase 2 ES-EKF + NHC (No AI: forward speed holds last known GNSS velocity)
                vel_p2 = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_last_gnss, speed_variance=1.0, motion_state="DRIVING")
                fp2 = ekf_p2.predict(cal, vel_p2)
                pts_p2.append(fp2.position_enu_m[:2])

                # 3. Stage 2: Phase 3 ES-EKF + AI Velocity Fusion
                vel_p3 = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_ai, speed_variance=0.3, motion_state=m_state)
                fp3 = ekf_p3.predict(cal, vel_p3)
                pts_p3.append(fp3.position_enu_m[:2])

                # 4. Stage 3: Phase 4 Map-Matched ES-EKF
                vel_p4 = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_ai, speed_variance=0.3, motion_state=m_state)
                fp4 = ekf_p4.predict(cal, vel_p4)

                # Apply Phase 4 covariance-driven road network constraint with kinematic turn adaptation & fork resolution
                turn_rate_dps = abs(np.degrees(cal.gyro_vehicle[2]))
                is_turning = turn_rate_dps > 2.0

                curr_p = ekf_p4._p[:2]
                curr_h_deg = float(np.degrees(ekf_p4._heading_rad)) % 360.0

                var_yaw = float(ekf_p4._P[8, 8])
                sigma_yaw_deg = float(np.degrees(np.sqrt(max(1e-6, var_yaw))))
                sigma_eff = float(np.sqrt(sigma_yaw_deg**2 + 15.0**2))
                if is_turning:
                    sigma_eff = max(sigma_eff, 45.0)

                cands = road_net.find_candidates(curr_p, radius_m=50.0)
                if len(cands) > 0:
                    valid_cands = []
                    for s in cands:
                        proj, d_perp, frac = s.project_point(curr_p)
                        h_diff = abs((curr_h_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)
                        if d_perp < 35.0:
                            end_factor = 0.05 if frac >= 0.95 else 1.0
                            score = np.exp(-0.5 * (d_perp / 8.0)**2) * np.exp(-0.5 * (h_diff / sigma_eff)**2) * end_factor
                            if score > 1e-4:
                                valid_cands.append((s, proj, d_perp, h_diff, score))
                    if len(valid_cands) > 0:
                        valid_cands.sort(key=lambda x: x[4], reverse=True)
                        top_c = valid_cands[:2]
                        is_fork = False
                        if len(top_c) >= 2:
                            h_sep = abs((top_c[0][0].bearing_deg - top_c[1][0].bearing_deg + 180.0) % 360.0 - 180.0)
                            if h_sep < 15.0 and top_c[1][4] > 0.40 * top_c[0][4]:
                                is_fork = True
                        if is_fork:
                            w0, w1 = top_c[0][4], top_c[1][4]
                            tot_w = w0 + w1
                            best_s = top_c[0][0]
                            best_proj = (w0 * top_c[0][1] + w1 * top_c[1][1]) / tot_w
                        else:
                            best_s, best_proj, _, _, _ = top_c[0]

                        ekf_p4._p[0] = best_proj[0]
                        ekf_p4._p[1] = best_proj[1]

                        if not is_turning and not is_fork:
                            ekf_p4.reanchor_heading(best_s.bearing_deg, confidence=0.8)

                pts_p4.append(ekf_p4._p[:2].copy())

            pts_naive = np.array(pts_naive)
            pts_p2 = np.array(pts_p2)
            pts_p3 = np.array(pts_p3)
            pts_p4 = np.array(pts_p4)

            err_naive = float(np.linalg.norm((pts_naive[-1] - pts_naive[0]) - gt_disp)) if len(pts_naive) > 1 else 9999.0
            err_p2 = float(np.linalg.norm((pts_p2[-1] - pts_p2[0]) - gt_disp)) if len(pts_p2) > 1 else 9999.0
            err_p3 = float(np.linalg.norm((pts_p3[-1] - pts_p3[0]) - gt_disp)) if len(pts_p3) > 1 else 9999.0
            err_p4 = float(np.linalg.norm((pts_p4[-1] - pts_p4[0]) - gt_disp)) if len(pts_p4) > 1 else 9999.0

            d_ref = max(dist_m, 1.0)
            drift_naive = (err_naive / d_ref) * 100.0
            drift_p2 = (err_p2 / d_ref) * 100.0
            drift_p3 = (err_p3 / d_ref) * 100.0
            drift_p4 = (err_p4 / d_ref) * 100.0

            cur_num = offset_idx + idx + 1
            pct = (cur_num / 50.0) * 100.0
            print(f"  [{cur_num:2d}/50] ({pct:3.0f}%) {row['scenario'][:25]:25s} | Naive: {drift_naive:6.1f}% | P2: {drift_p2:5.1f}% | P3: {drift_p3:5.1f}% | P4: {drift_p4:5.1f}%", flush=True)

            results.append({
                "scenario": row["scenario"],
                "trip": row["trip"],
                "duration_s": duration,
                "distance_m": dist_m,
                "drift_naive_pct": drift_naive,
                "error_naive_m": err_naive,
                "drift_p2_pct": drift_p2,
                "error_p2_m": err_p2,
                "drift_p3_pct": drift_p3,
                "error_p3_m": err_p3,
                "drift_p4_pct": drift_p4,
                "error_p4_m": err_p4,
            })
        return results

    sc1_df = sweep_df[sweep_df["trip"].str.contains("S-S1")]
    sc2_df = sweep_df[sweep_df["trip"].str.contains("S-S2")]

    print("\nExecuting Comprehensive 4-Stage Benchmark Sweep across 50 Scenarios...", flush=True)
    r1 = eval_trip_all_stages(trip_s1, calib_s1, v_s1, road_net_s1, sc1_df, offset_idx=0)
    r2 = eval_trip_all_stages(trip_s2, calib_s2, v_s2, road_net_s2, sc2_df, offset_idx=len(r1))

    df_all = pd.DataFrame(r1 + r2)

    # 5. Compute Comprehensive Metrics
    print("\n" + "=" * 95, flush=True)
    print("                     COMPREHENSIVE 4-STAGE BENCHMARK RESULTS MATRIX                      ", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Pipeline Generation':<38} | {'Median Drift':<12} | {'P90 Drift':<10} | {'Worst-Case':<11} | {'Pass Target (<10%)':<18}")
    print("-" * 95, flush=True)

    stages = [
        ("Stage 0: Naive Open-Loop Baseline", "drift_naive_pct"),
        ("Stage 1: Phase 2 ES-EKF + NHC (No AI)", "drift_p2_pct"),
        ("Stage 2: Phase 3 ES-EKF + AI Velocity", "drift_p3_pct"),
        ("Stage 3: Phase 4 Map-Matched ES-EKF", "drift_p4_pct"),
    ]

    summary_stats = []
    for label, col in stages:
        med = float(df_all[col].median())
        p90 = float(df_all[col].quantile(0.90))
        worst = float(df_all[col].max())
        passes = int((df_all[col] < 10.0).sum())
        pass_pct = (passes / len(df_all)) * 100.0
        print(f"{label:<38} | {med:10.2f}% | {p90:8.2f}% | {worst:9.2f}% | {passes:2d}/50 ({pass_pct:4.1f}%)", flush=True)
        summary_stats.append({
            "Stage": label,
            "Median_Drift_Pct": med,
            "P90_Drift_Pct": p90,
            "Worst_Case_Pct": worst,
            "Passed_Count": passes,
            "Pass_Pct": pass_pct,
        })

    print("=" * 95, flush=True)
    print(f"Benchmark Sweep Completed in {time.time() - t_start:.2f} seconds.", flush=True)

    # Save CSV
    out_csv = os.path.join(ARTIFACT_DIR, "comprehensive_multi_stage_benchmark_results.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Detailed 50-scenario per-stage CSV saved to: {out_csv}", flush=True)

    # 6. Generate 4-Stage Comparison Chart
    plt.figure(figsize=(10, 6), dpi=200)
    stage_labels = ["Stage 0\nNaive Baseline", "Stage 1\nPhase 2 (EKF+NHC)", "Stage 2\nPhase 3 (AI EKF)", "Stage 3\nPhase 4 (Map-Matched)"]
    meds = [s["Median_Drift_Pct"] for s in summary_stats]
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#0275d8"]

    bars = plt.bar(stage_labels, meds, color=colors, width=0.55, edgecolor="black", linewidth=1.2, zorder=3)
    plt.axhline(10.0, color="green", linestyle="--", linewidth=2.0, label="SIH Benchmark Target (< 10% Drift)", zorder=4)

    for bar, med in zip(bars, meds):
        y_val = bar.get_height()
        display_text = f"{med:.1f}%" if med < 1000 else f"{med:.0f}%"
        plt.text(bar.get_x() + bar.get_width() / 2.0, min(y_val + 2.0, 160.0), display_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.title("4-Stage Architectural Evolution of Smartphone Dead Reckoning (SIH)\nEvaluated Across 50 Standardized Real-World Driving Scenarios", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Median Trajectory Drift (%)", fontsize=12, fontweight="bold")
    plt.ylim(0, max(min(max(meds) * 1.15, 180.0), 20.0))
    plt.grid(axis="y", linestyle=":", alpha=0.7, zorder=0)
    plt.legend(loc="upper right", frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()

    out_chart = os.path.join(ARTIFACT_DIR, "comprehensive_4stage_benchmark_chart.png")
    plt.savefig(out_chart)
    plt.close()
    print(f"Comparative benchmark visualization saved to: {out_chart}", flush=True)


if __name__ == "__main__":
    run_multi_stage_benchmark()
