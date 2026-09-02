import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig
from sih.data.geo import geodetic_to_enu

def run_fast_eval(trip, blackout, pipeline):
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(blackout.start_time_s * 1e9)
    bo_end_ns = bo_start_ns + int(blackout.duration_s * 1e9)

    gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
    gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples], dtype=np.float64)
    gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples], dtype=np.float64)
    gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)

    pipeline.reset(initial_gnss=trip.gnss_samples[0])
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)

    records = []
    for imu in trip.imu_samples:
        t_curr = imu.timestamp_ns
        if t_curr > bo_end_ns + int(2e9):
            break

        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            if not (bo_start_ns <= g.timestamp_ns <= bo_end_ns):
                pipeline.process_gnss(g)
            gnss_idx += 1

        calib, vel, fused, matched = pipeline.process_imu(imu)

        if bo_start_ns <= t_curr <= bo_end_ns:
            gt_e = np.interp(t_curr, gnss_ts, gnss_enu[:, 0])
            gt_n = np.interp(t_curr, gnss_ts, gnss_enu[:, 1])
            err = np.sqrt((fused.position_enu_m[0] - gt_e)**2 + (fused.position_enu_m[1] - gt_n)**2)
            records.append({
                "time_s": (t_curr - t0_ns) * 1e-9,
                "est_e": fused.position_enu_m[0], "est_n": fused.position_enu_m[1],
                "gt_e": gt_e, "gt_n": gt_n,
                "error_m": err,
                "speed": np.linalg.norm(fused.velocity_enu_mps[:2]),
                "heading_deg": np.degrees(fused.heading_rad),
                "ai_speed": vel.forward_speed_mps if vel is not None else 0.0
            })

    df = pd.DataFrame(records)
    gt_d = np.sum(np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2))
    final_err = df["error_m"].iloc[-1]
    drift_pct = (final_err / max(gt_d, 1.0)) * 100.0
    return {
        "dist_m": gt_d,
        "final_err_m": final_err,
        "max_err_m": df["error_m"].max(),
        "rmse_m": np.sqrt(np.mean(df["error_m"]**2)),
        "drift_pct": drift_pct,
        "df": df
    }

def main():
    loader = GenericDataLoader()
    csv_path = download_iovnbd_trip("S-S1")
    trip = loader.load_file(csv_path)

    scenarios = [
        BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_at_120s"),
        BlackoutConfig(start_time_s=300.0, duration_s=60.0, name="60s_at_300s"),
    ]

    for sc in scenarios:
        print(f"\n================ Scenario: {sc.name} ================")
        # 1. Naive
        p_naive = assemble_pipeline(PipelineConfig(fusion=FusionFilterConfig(algorithm="naive_dead_reckoning")))
        r_naive = run_fast_eval(trip, sc, p_naive)
        print(f"Naive:       FinalErr={r_naive['final_err_m']:.1f}m, Drift={r_naive['drift_pct']:.1f}%, RMSE={r_naive['rmse_m']:.1f}m")

        # 2. ES-EKF
        p_ekf = assemble_pipeline(PipelineConfig(fusion=FusionFilterConfig(algorithm="es_ekf")))
        r_ekf = run_fast_eval(trip, sc, p_ekf)
        print(f"ES-EKF:      FinalErr={r_ekf['final_err_m']:.1f}m, Drift={r_ekf['drift_pct']:.1f}%, RMSE={r_ekf['rmse_m']:.1f}m")

        # 3. ES-EKF + NHC
        p_nhc = assemble_pipeline(PipelineConfig(fusion=FusionFilterConfig(algorithm="es_ekf_nhc")))
        r_nhc = run_fast_eval(trip, sc, p_nhc)
        print(f"ES-EKF+NHC:  FinalErr={r_nhc['final_err_m']:.1f}m, Drift={r_nhc['drift_pct']:.1f}%, RMSE={r_nhc['rmse_m']:.1f}m")

        # 4. ES-EKF + AI
        p_ai = assemble_pipeline(PipelineConfig(
            velocity=VelocityEstimatorConfig(algorithm="tcn_attention", params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}),
            fusion=FusionFilterConfig(algorithm="es_ekf_nhc")
        ))
        r_ai = run_fast_eval(trip, sc, p_ai)
        print(f"ES-EKF+AI:   FinalErr={r_ai['final_err_m']:.1f}m, Drift={r_ai['drift_pct']:.1f}%, RMSE={r_ai['rmse_m']:.1f}m")

if __name__ == "__main__":
    main()
