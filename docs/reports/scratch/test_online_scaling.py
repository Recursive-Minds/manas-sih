import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

# Test integrating online speed scale tracking inside the benchmark run
def test_online_scaling(trip_id, start_s, dur_s, name):
    trip = GenericDataLoader().load_file(download_iovnbd_trip(trip_id))
    sc = BlackoutConfig(start_time_s=start_s, duration_s=dur_s, name=name)
    
    cfg = PipelineConfig(
        calibration=CalibrationConfig(algorithm="auto"),
        velocity=VelocityEstimatorConfig(
            algorithm="tcn_attention",
            params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}
        ),
        fusion=FusionFilterConfig(
            algorithm="es_ekf_nhc",
            params={"nhc_lateral_std": 0.15, "nhc_vertical_std": 0.15, "enable_online_speed_scale": True}
        )
    )
    runner = BenchmarkRunner(config=cfg)
    res = runner.run_trip(trip, sc)
    df_bo = res.history_df[res.history_df["in_blackout"]].copy()
    
    gt_dist = np.sum(np.sqrt(np.diff(df_bo["gt_e"])**2 + np.diff(df_bo["gt_n"])**2))
    est_dist = np.sum(np.sqrt(np.diff(df_bo["est_e"])**2 + np.diff(df_bo["est_n"])**2))
    
    print(f"=== {trip_id} - {name} with Online Speed Scaling ===")
    print(f"  GT Distance:     {gt_dist:.2f} m")
    print(f"  Est Distance:    {est_dist:.2f} m (Ratio: {est_dist/gt_dist:.4f})")
    print(f"  Final Pos Error: {res.final_position_error_m:.2f} m")
    print(f"  Drift %:         {res.drift_percentage:.2f}%\n")

