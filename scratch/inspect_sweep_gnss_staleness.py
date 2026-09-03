"""
Inspect staleness of GNSS heading in the 50 standardized scenarios (SWEEP_CSV).
Calculates:
1. Difference between bo_start_ns and the GNSS fix used for initialization
2. Heading offset between stale GNSS bearing and true ground truth bearing at bo_start_ns
3. Impact of gyro extrapolation: last_gnss.bearing_deg + integral(omega_z dt)
"""

import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.data.geo import geodetic_to_enu

SWEEP_CSV = r"C:\Users\carpe\SIH\artifacts\randomized_blackout_sweep_results.csv"

def analyze_trip_scenarios(trip, sub_df):
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

    results = []
    for idx, row in sub_df.iterrows():
        t_start = float(row["start_time_s"])
        duration = float(row["duration_s"])
        bo_start_ns = t0_ns + int(t_start * 1e9)

        # Ground truth true bearing at exact bo_start_ns
        t_bo_rel = (bo_start_ns - t_gnss0_ns) * 1e-9
        v_e = float(cs_e.derivative(1)(t_bo_rel))
        v_n = float(cs_n.derivative(1)(t_bo_rel))
        true_hdg_entry = float(np.degrees(np.arctan2(v_e, v_n))) % 360.0

        # The GNSS sample picked in run_final_benchmark.py
        gt_start_sample = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
        stale_s = (bo_start_ns - gt_start_sample.timestamp_ns) * 1e-9
        stale_hdg = gt_start_sample.bearing_deg if gt_start_sample.bearing_deg is not None else np.nan

        # The last strictly PRE-blackout GNSS sample
        pre_gnss = [g for g in valid_gnss if g.timestamp_ns <= bo_start_ns]
        last_pre = pre_gnss[-1] if pre_gnss else valid_gnss[0]
        last_pre_stale_s = (bo_start_ns - last_pre.timestamp_ns) * 1e-9
        last_pre_hdg = last_pre.bearing_deg if last_pre.bearing_deg is not None else np.nan

        # Gyro extrapolation from last_pre to bo_start_ns
        accum_wz_deg = 0.0
        for j, imu in enumerate(trip.imu_samples):
            if last_pre.timestamp_ns < imu.timestamp_ns <= bo_start_ns:
                dt = (imu.timestamp_ns - trip.imu_samples[j-1].timestamp_ns) * 1e-9
                accum_wz_deg += np.degrees(calib_samples[j].gyro_vehicle[2] * dt)

        extrap_hdg = (last_pre_hdg + accum_wz_deg) % 360.0 if not np.isnan(last_pre_hdg) else np.nan

        def diff_ang(a, b):
            return (a - b + 180.0) % 360.0 - 180.0

        off_stale = diff_ang(stale_hdg, true_hdg_entry)
        off_last_pre = diff_ang(last_pre_hdg, true_hdg_entry)
        off_extrap = diff_ang(extrap_hdg, true_hdg_entry)

        results.append({
            "start_time_s": t_start,
            "duration_s": duration,
            "dist_m": float(row["distance_m"]),
            "stale_s": stale_s,
            "last_pre_stale_s": last_pre_stale_s,
            "true_hdg": true_hdg_entry,
            "stale_hdg": stale_hdg,
            "extrap_hdg": extrap_hdg,
            "off_stale": off_stale,
            "off_last_pre": off_last_pre,
            "off_extrap": off_extrap,
        })
    return pd.DataFrame(results)

def main():
    loader = GenericDataLoader()
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")
    sweep_df = pd.read_csv(SWEEP_CSV)

    df_s1 = sweep_df[sweep_df["trip"].str.contains("S-S1")]
    df_s2 = sweep_df[sweep_df["trip"].str.contains("S-S2")]

    res_s1 = analyze_trip_scenarios(trip_s1, df_s1)
    res_s2 = analyze_trip_scenarios(trip_s2, df_s2)
    res_all = pd.concat([res_s1, res_s2])

    print("=" * 80)
    print("GNSS STALENESS & HEADING OFFSET IN 50 STANDARDIZED SCENARIOS")
    print("=" * 80)
    print(f"Staleness of GNSS sample used (abs): Mean={res_all['stale_s'].abs().mean():.2f}s, Max={res_all['stale_s'].abs().max():.2f}s")
    print(f"Staleness of last pre-GNSS fix:      Mean={res_all['last_pre_stale_s'].mean():.2f}s, Max={res_all['last_pre_stale_s'].max():.2f}s")
    print("-" * 80)
    print(f"Offset (Stale GNSS Bearing vs True GT):     AbsMean={res_all['off_stale'].abs().mean():.2f}°, MaxAbs={res_all['off_stale'].abs().max():.2f}°")
    print(f"Offset (Last Pre-GNSS Bearing vs True GT): AbsMean={res_all['off_last_pre'].abs().mean():.2f}°, MaxAbs={res_all['off_last_pre'].abs().max():.2f}°")
    print(f"Offset (Gyro-Extrapolated vs True GT):      AbsMean={res_all['off_extrap'].abs().mean():.2f}°, MaxAbs={res_all['off_extrap'].abs().max():.2f}°")
    print("-" * 80)

if __name__ == "__main__":
    main()
