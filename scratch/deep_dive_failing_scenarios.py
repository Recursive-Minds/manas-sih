"""
Deep dive analysis into failing scenarios #20, #31, #18, #14 on unseen S-M trip.
Examines:
1. Trajectory and kinematics in the 10 seconds before blackout entry and during blackout
2. Gyro angular rates (omega_z) around blackout entry
3. GNSS bearings vs displacement course vs true trajectory heading
4. Road network candidates around entry and during outage
5. Why #20 and #31 fail: heading error vs map topology (acute fork ambiguity)
"""

import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.data.geo import geodetic_to_enu
from sih.map.network import RoadNetwork

DATA_PATH = r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv"

def main():
    loader = GenericDataLoader()
    trip = loader.load_file(DATA_PATH)
    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    
    calibrator = MountCalibrator(window_size=100)
    if "mount_pitch_deg" in trip.metadata and "mount_roll_deg" in trip.metadata:
        calibrator.calibrate_from_mount_angles(trip.metadata["mount_pitch_deg"], trip.metadata["mount_roll_deg"])
    else:
        for g in trip.gnss_samples: calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]

    times_s = np.array([(g.timestamp_ns - valid_gnss[0].timestamp_ns) * 1e-9 for g in valid_gnss])
    enus = np.array([geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2] for g in valid_gnss])
    cs_e = CubicSpline(times_s, enus[:, 0])
    cs_n = CubicSpline(times_s, enus[:, 1])
    t0_ns = trip.imu_samples[0].timestamp_ns
    t_gnss0_ns = valid_gnss[0].timestamp_ns

    results_csv = r"C:\Users\carpe\SIH\artifacts\phase4_unseen_sm_benchmark_results.csv"
    df = pd.read_csv(results_csv)

    pts_enu = []
    pts_lat_lon = []
    last_p = None
    for g in trip.gnss_samples:
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        if last_p is None or np.linalg.norm(enu - last_p) >= 20.0:
            pts_enu.append(enu)
            pts_lat_lon.append((g.latitude_deg, g.longitude_deg))
            last_p = enu
    road_net = RoadNetwork.from_polyline_coords(np.array(pts_enu), pts_lat_lon, road_id_prefix="sm", road_type="primary", cell_size_m=100.0)

    for sc_id in [20, 31, 18, 14]:
        row = df[df["scenario_id"] == sc_id].iloc[0]
        bo_start_s = float(row["start_time_s"])
        duration_s = float(row["duration_s"])
        dist_m = float(row["distance_m"])
        bo_start_ns = t0_ns + int(bo_start_s * 1e9)
        bo_end_ns   = bo_start_ns + int(duration_s * 1e9)

        # True instantaneous heading at bo_start_ns
        t_bo_rel = (bo_start_ns - t_gnss0_ns) * 1e-9
        v_e = float(cs_e.derivative(1)(t_bo_rel))
        v_n = float(cs_n.derivative(1)(t_bo_rel))
        true_hdg_entry = float(np.degrees(np.arctan2(v_e, v_n))) % 360.0

        # Pre-blackout GNSS
        pre_gnss = [g for g in valid_gnss if bo_start_ns - int(5.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
        
        # Gyro rates in 5 seconds before bo_start_ns
        pre_gyros = [np.degrees(calib_samples[j].gyro_vehicle[2]) for j, imu in enumerate(trip.imu_samples) 
                     if bo_start_ns - int(5.0 * 1e9) <= imu.timestamp_ns <= bo_start_ns]
        mean_pre_wz = np.mean(pre_gyros) if pre_gyros else 0.0
        max_pre_wz  = np.max(np.abs(pre_gyros)) if pre_gyros else 0.0

        # Gyros during blackout
        bo_gyros = [np.degrees(calib_samples[j].gyro_vehicle[2]) for j, imu in enumerate(trip.imu_samples)
                    if bo_start_ns <= imu.timestamp_ns <= bo_end_ns]
        mean_bo_wz = np.mean(bo_gyros) if bo_gyros else 0.0
        max_bo_wz  = np.max(np.abs(bo_gyros)) if bo_gyros else 0.0

        # Road candidates at entry
        entry_enu = [float(cs_e(t_bo_rel)), float(cs_n(t_bo_rel))]
        cands = road_net.find_candidates(np.array(entry_enu), radius_m=50.0)

        print(f"\n{'='*70}")
        print(f"SCENARIO #{sc_id}: Dist={dist_m:.1f}m, Dur={duration_s:.1f}s, Pure Drift={row['pure_drift_pct']:.1f}%, Phase4 Drift={row['map_drift_pct']:.1f}%")
        print(f"{'='*70}")
        print(f"True Heading at Entry: {true_hdg_entry:.2f}°")
        print(f"Pre-Blackout GNSS Fixes (last 5s): N={len(pre_gnss)}")
        for g in pre_gnss[-3:]:
            dt_to_bo = (g.timestamp_ns - bo_start_ns) * 1e-9
            print(f"  Fix at {dt_to_bo:+.2f}s: bearing={g.bearing_deg}, speed={g.speed_mps:.2f} m/s")
        print(f"Pre-Blackout Yaw Rate: Mean={mean_pre_wz:+.2f}°/s, Max=|{max_pre_wz:.2f}|°/s")
        print(f"Blackout Yaw Rate:     Mean={mean_bo_wz:+.2f}°/s, Max=|{max_bo_wz:.2f}|°/s")
        print(f"Road Candidates within 50m at Entry (N={len(cands)}):")
        for c in cands[:4]:
            proj, d_perp, frac = c.project_point(np.array(entry_enu))
            h_diff = (c.bearing_deg - true_hdg_entry + 180.0) % 360.0 - 180.0
            print(f"  Segment {c.segment_id}: bearing={c.bearing_deg:.1f}°, d_perp={d_perp:.1f}m, hdg_diff={h_diff:+.1f}°")

if __name__ == "__main__":
    main()
