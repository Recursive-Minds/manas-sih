import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

def main():
    print("=== RESEARCH: SPLINE-SMOOTHED GROUND TRUTH HEADING TARGETS ===", flush=True)

    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")

    # Extract GNSS ENU coordinates
    g_ts  = []
    g_enu = []
    for g in trip_s1.gnss_samples:
        if g.speed_mps is not None and g.speed_mps > 1.0:
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip_s1.reference_lat_deg, trip_s1.reference_lon_deg, 0.0)[:2]
            g_ts.append(g.timestamp_ns / 1e9) # seconds
            g_enu.append(enu)

    g_ts  = np.array(g_ts)
    g_enu = np.array(g_enu)

    print(f"Loaded {len(g_ts)} valid GNSS points for spline fitting.", flush=True)

    # Fit smooth UnivariateSpline on East and North positions
    spl_x = UnivariateSpline(g_ts, g_enu[:, 0], s=len(g_ts) * 2.0)
    spl_y = UnivariateSpline(g_ts, g_enu[:, 1], s=len(g_ts) * 2.0)

    # Derivative splines for velocity vectors
    vx_spl = spl_x.derivative()
    vy_spl = spl_y.derivative()

    # Compute continuous smooth ground-truth heading angle \psi_GT(t)
    vx_val = vx_spl(g_ts)
    vy_val = vy_spl(g_ts)

    psi_gt_rad = np.unwrap(np.arctan2(vx_val, vy_val)) # Course heading angle (0=North, pi/2=East)

    # Compute 2.0s windowed delta heading \Delta \psi_GT (rad across 2s)
    # Find delta heading over 2s intervals
    dt_win = 2.0 # 2 seconds
    d_psi_gt = []
    for t in g_ts:
        if t + dt_win <= g_ts[-1]:
            psi_t  = np.arctan2(vx_spl(t), vy_spl(t))
            psi_t2 = np.arctan2(vx_spl(t + dt_win), vy_spl(t + dt_win))
            d_psi = (psi_t2 - psi_t + np.pi) % (2.0 * np.pi) - np.pi
            d_psi_gt.append(d_psi)

    d_psi_gt = np.array(d_psi_gt)
    print(f"Successfully generated {len(d_psi_gt)} clean spline-smoothed delta-heading targets!", flush=True)
    print(f"Delta Heading (2s window): Mean = {np.degrees(np.mean(d_psi_gt)):.4f}°, Std = {np.degrees(np.std(d_psi_gt)):.4f}°", flush=True)

if __name__ == "__main__":
    main()
