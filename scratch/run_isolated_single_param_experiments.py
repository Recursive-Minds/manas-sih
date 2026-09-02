import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\SIH\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
MODEL_PATH   = r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt"

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STRICT ISOLATED SINGLE-PARAMETER BENCHMARK EXPERIMENTS ({device}) ===", flush=True)

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

    sweep_df = pd.read_csv(SWEEP_CSV)

    def run_50_sample_sweep(
        turn_th_deg=1.5,
        max_bg_deg=0.5,
        cool_s=0.5,
        speed_scale_mode="dynamic", # "off", "static_1.0", "static_3.4", "dynamic"
        r_nhc=0.20
    ):
        results = []
        for idx, row in sweep_df.iterrows():
            trip_name = str(row["trip"])
            t_start   = float(row["start_time_s"])
            duration  = float(row["duration_s"])
            dist_m    = float(row["distance_m"])

            if "S-S1" in trip_name:
                trip = trip_s1
                calib = calib_s1
                v_preds = v_preds_s1
            else:
                trip = trip_s2
                calib = calib_s2
                v_preds = v_preds_s2

            t0_ns = trip.imu_samples[0].timestamp_ns
            bo_start_ns = t0_ns + int(t_start * 1e9)
            bo_end_ns   = bo_start_ns + int(duration * 1e9)

            gt_start_sample = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
            gt_end_sample   = min(trip.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_end_ns))

            gt_start_enu = geodetic_to_enu(
                gt_start_sample.latitude_deg, gt_start_sample.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )[:2]
            gt_end_enu = geodetic_to_enu(
                gt_end_sample.latitude_deg, gt_end_sample.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )[:2]
            gt_disp = gt_end_enu - gt_start_enu

            warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
            warmup_gnss = min([g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns], key=lambda g: abs(g.timestamp_ns - warmup_start_ns), default=trip.gnss_samples[0])

            init_scale = 3.40 if speed_scale_mode == "static_3.4" else 1.00

            ekf = ErrorStateEKF(
                turn_threshold_rad_s=np.radians(turn_th_deg),
                cooldown_duration_s=cool_s,
                max_gyro_bias_rad_s=np.radians(max_bg_deg),
                initial_speed_scale=init_scale,
                nhc_lateral_std=r_nhc,
                nhc_vertical_std=r_nhc
            )
            ekf.init_from_gnss(warmup_gnss)

            gnss_idx = 0
            n_gnss   = len(trip.gnss_samples)
            est_pts  = []

            for j, imu in enumerate(trip.imu_samples):
                t_curr = imu.timestamp_ns
                if t_curr < warmup_start_ns: continue
                if t_curr > bo_end_ns + int(1e9): break

                while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                    g = trip.gnss_samples[gnss_idx]
                    if g.timestamp_ns < bo_start_ns:
                        ekf.update_gnss(g)
                        if speed_scale_mode == "off" or speed_scale_mode.startswith("static"):
                            ekf._speed_scale = init_scale
                        if g.speed_mps is not None and g.speed_mps > 3.0 and g.bearing_deg is not None:
                            ekf._heading_rad = float(np.radians(g.bearing_deg))
                    gnss_idx += 1

                cal = calib[j]
                v_fwd = float(v_preds[j])
                m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
                vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)
                fused = ekf.predict(cal, vel)

                if bo_start_ns <= t_curr <= bo_end_ns:
                    est_pts.append(fused.position_enu_m[:2])

            est_pts = np.array(est_pts)
            if len(est_pts) < 2: continue
            est_disp = est_pts[-1] - est_pts[0]
            final_err = float(np.linalg.norm(est_disp - gt_disp))
            drift_pct = (final_err / max(dist_m, 1.0)) * 100.0
            results.append(drift_pct)

        med = float(np.median(results))
        worst = float(np.max(results))
        return med, worst

    # Define baseline reference params
    # Default baseline: turn_th=1.5, max_bg=0.5, cool=0.5, speed_scale_mode="dynamic", r_nhc=0.20
    experiments = []

    # Benchmark Baseline (Pre-experiments reference)
    med_ref, worst_ref = run_50_sample_sweep(1.5, 0.5, 0.5, "dynamic", 0.20)
    experiments.append({
        "exp_id": "Ref (Current Best)",
        "param_changed": "Baseline Reference",
        "turn_th": 1.5, "max_bg": 0.5, "cool": 0.5, "spd_mode": "dynamic", "r_nhc": 0.20,
        "median_pct": med_ref, "worst_pct": worst_ref
    })

    # Group (a): Gyro-bias clip bound finer sweep (0.3°/s to 1.0°/s)
    print("\n--- Group (a): Gyro-bias Clip Bound Sweep ---", flush=True)
    for bg_val in [0.3, 0.4, 0.5, 0.6, 0.75, 1.0]:
        med, worst = run_50_sample_sweep(turn_th_deg=1.5, max_bg_deg=bg_val, cool_s=0.5, speed_scale_mode="dynamic", r_nhc=0.20)
        experiments.append({
            "exp_id": f"Exp A: bg={bg_val}°/s",
            "param_changed": f"max_gyro_bias={bg_val}°/s",
            "turn_th": 1.5, "max_bg": bg_val, "cool": 0.5, "spd_mode": "dynamic", "r_nhc": 0.20,
            "median_pct": med, "worst_pct": worst
        })
        print(f"  max_bg={bg_val:4.2f}°/s -> Median={med:6.2f}%, Worst={worst:6.2f}%", flush=True)

    # Group (b): Speed-scale logic isolation
    print("\n--- Group (b): Speed-Scale Logic Isolation ---", flush=True)
    for spd_mode in ["off", "static_1.0", "static_3.4", "dynamic"]:
        med, worst = run_50_sample_sweep(turn_th_deg=1.5, max_bg_deg=0.5, cool_s=0.5, speed_scale_mode=spd_mode, r_nhc=0.20)
        experiments.append({
            "exp_id": f"Exp B: spd={spd_mode}",
            "param_changed": f"speed_scale_mode={spd_mode}",
            "turn_th": 1.5, "max_bg": 0.5, "cool": 0.5, "spd_mode": spd_mode, "r_nhc": 0.20,
            "median_pct": med, "worst_pct": worst
        })
        print(f"  spd_mode={spd_mode:12s} -> Median={med:6.2f}%, Worst={worst:6.2f}%", flush=True)

    # Group (c): NHC Measurement Noise R_NHC Sweep
    print("\n--- Group (c): R_NHC Measurement Noise Sweep ---", flush=True)
    for r_val in [0.05, 0.10, 0.20, 0.50, 1.00]:
        med, worst = run_50_sample_sweep(turn_th_deg=1.5, max_bg_deg=0.5, cool_s=0.5, speed_scale_mode="dynamic", r_nhc=r_val)
        experiments.append({
            "exp_id": f"Exp C: r_nhc={r_val}",
            "param_changed": f"r_nhc={r_val}",
            "turn_th": 1.5, "max_bg": 0.5, "cool": 0.5, "spd_mode": "dynamic", "r_nhc": r_val,
            "median_pct": med, "worst_pct": worst
        })
        print(f"  r_nhc={r_val:4.2f} -> Median={med:6.2f}%, Worst={worst:6.2f}%", flush=True)

    t_total = time.time() - t0
    print(f"\nCompleted {len(experiments)} isolated experiments in {t_total:.2f} seconds!", flush=True)

    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv(os.path.join(ARTIFACT_DIR, "isolated_single_param_experiments.csv"), index=False)

    print("\n==========================================================================")
    print("                 ISOLATED EXPERIMENTS SUMMARY TABLE                       ")
    print("==========================================================================")
    for e in experiments:
        print(f"{e['exp_id']:25s} | {e['param_changed']:30s} | Med: {e['median_pct']:6.2f}% | Worst: {e['worst_pct']:6.2f}%")
    print("==========================================================================")

if __name__ == "__main__":
    main()
