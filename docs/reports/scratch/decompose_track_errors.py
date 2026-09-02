import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

def analyze_track_errors(start_s, dur_s, name):
    sc = BlackoutConfig(start_time_s=start_s, duration_s=dur_s, name=name)
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
    res = runner.run_trip(trip, sc)
    df_bo = res.history_df[res.history_df["in_blackout"]].copy()
    
    # End positions
    est_end = np.array([df_bo["est_e"].iloc[-1], df_bo["est_n"].iloc[-1]])
    gt_end = np.array([df_bo["gt_e"].iloc[-1], df_bo["gt_n"].iloc[-1]])
    
    # Tangent vector of ground truth at the end of the blackout
    gt_tangent = np.array([
        df_bo["gt_e"].iloc[-1] - df_bo["gt_e"].iloc[-10],
        df_bo["gt_n"].iloc[-1] - df_bo["gt_n"].iloc[-10]
    ])
    gt_tangent_unit = gt_tangent / np.linalg.norm(gt_tangent)
    gt_normal_unit = np.array([-gt_tangent_unit[1], gt_tangent_unit[0]])
    
    pos_err_vec = est_end - gt_end
    along_track_err = np.dot(pos_err_vec, gt_tangent_unit)
    cross_track_err = np.dot(pos_err_vec, gt_normal_unit)
    
    total_err = np.linalg.norm(pos_err_vec)
    
    print(f"=== {name} Error Decomposition ===")
    print(f"  Total Euclidean Error:  {total_err:.2f} m ({res.drift_percentage:.2f}% drift)")
    print(f"  Along-Track Error:      {along_track_err:+.2f} m ({'Overshot' if along_track_err > 0 else 'Short by'} {abs(along_track_err):.1f}m along the road)")
    print(f"  Cross-Track Error:      {cross_track_err:+.2f} m ({'Drifted Right' if cross_track_err > 0 else 'Drifted Left'} by {abs(cross_track_err):.1f}m off the road)")
    print(f"  Along-track variance %: {(along_track_err**2 / total_err**2)*100:.1f}%")
    print(f"  Cross-track variance %: {(cross_track_err**2 / total_err**2)*100:.1f}%\n")

analyze_track_errors(120.0, 30.0, "30s_blackout_at_120s")
analyze_track_errors(300.0, 60.0, "60s_blackout_at_300s")
