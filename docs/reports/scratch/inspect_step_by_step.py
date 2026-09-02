import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig
from sih.core.pipeline import assemble_pipeline
from sih.eval.benchmark import BlackoutConfig
sys.path.insert(0, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92\scratch")
from fast_eval import run_fast_eval

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")
sc = BlackoutConfig(start_time_s=120.0, duration_s=30.0, name="30s_at_120s")

# Let's inspect the pipeline with AI velocity
p_ai = assemble_pipeline(PipelineConfig(
    velocity=VelocityEstimatorConfig(algorithm="tcn_attention", params={"checkpoint_path": "models/checkpoints/best_velocity_model.pt"}),
    fusion=FusionFilterConfig(algorithm="es_ekf_nhc")
))

res = run_fast_eval(trip, sc, p_ai)
df = res["df"]

print("Sample rows during blackout (every 5 seconds):")
for t in [120, 125, 130, 135, 140, 145, 150]:
    row = df.iloc[(df['time_s'] - t).abs().argmin()]
    print(f"t={row['time_s']:5.1f}s | EstPos=({row['est_e']:6.1f}, {row['est_n']:6.1f}) | GTPos=({row['gt_e']:6.1f}, {row['gt_n']:6.1f}) | Err={row['error_m']:5.1f}m | EstSpeed={row['speed']:5.2f}m/s | AISpeed={row['ai_speed']:5.2f}m/s | Heading={row['heading_deg']:5.1f} deg")
