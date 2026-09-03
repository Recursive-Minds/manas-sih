"""
Compare scenario mix between S-M (35 scenarios) and 50-Scenario Benchmark (S-S1 & S-S2).
Analyzes:
1. Turn density: integrated |omega_z| dt per km, sharp turn frequency (>2 deg/s)
2. Distance distribution: min, 25%, median, 75%, max
3. Duration distribution: min, median, max
4. Speed profile: average speed, stationary fraction (<0.5 m/s), highway fraction (>15 m/s)
5. GNSS staleness at blackout entry (0s vs 3.34s avg)
6. Road candidate complexity / branchiness
"""

import sys
sys.path.insert(0, r"C:\Users\carpe\SIH")

import numpy as np
import pandas as pd

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.data.geo import geodetic_to_enu

def analyze_dataset_scenarios(trip, df, is_sm=False):
    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    calibrator = MountCalibrator(window_size=100)
    if "mount_pitch_deg" in trip.metadata and "mount_roll_deg" in trip.metadata:
        calibrator.calibrate_from_mount_angles(trip.metadata["mount_pitch_deg"], trip.metadata["mount_roll_deg"])
    else:
        for g in trip.gnss_samples: calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]

    t0_ns = trip.imu_samples[0].timestamp_ns
    results = []

    for idx, row in df.iterrows():
        t_start = float(row["start_time_s"])
        duration = float(row["duration_s"])
        dist_m = float(row["distance_m"])
        bo_start_ns = t0_ns + int(t_start * 1e9)
        bo_end_ns   = bo_start_ns + int(duration * 1e9)

        # IMU samples in blackout
        bo_imus = [imu for imu in trip.imu_samples if bo_start_ns <= imu.timestamp_ns <= bo_end_ns]
        bo_calibs = [calib_samples[j] for j, imu in enumerate(trip.imu_samples) if bo_start_ns <= imu.timestamp_ns <= bo_end_ns]

        if len(bo_calibs) < 2: continue

        w_zs = np.array([np.degrees(c.gyro_vehicle[2]) for c in bo_calibs])
        dts = np.diff([imu.timestamp_ns for imu in bo_imus]) * 1e-9
        dts = np.append(dts, 0.1)

        total_turn_deg = float(np.sum(np.abs(w_zs) * dts))
        dist_km = max(dist_m * 1e-3, 0.01)
        turn_density_deg_per_km = total_turn_deg / dist_km
        sharp_turn_sec = float(np.sum((np.abs(w_zs) > 2.0) * dts))
        sharp_turn_frac = sharp_turn_sec / duration

        # GNSS staleness
        gt_start_sample = min(valid_gnss, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
        stale_s = abs(bo_start_ns - gt_start_sample.timestamp_ns) * 1e-9

        # Speed
        avg_speed_mps = dist_m / max(duration, 1.0)

        drift_pct = float(row["map_drift_pct"]) if "map_drift_pct" in row else float(row["drift_pct"])

        results.append({
            "dist_m": dist_m,
            "duration_s": duration,
            "avg_speed_mps": avg_speed_mps,
            "stale_s": stale_s,
            "mean_abs_wz": float(np.mean(np.abs(w_zs))),
            "max_abs_wz": float(np.max(np.abs(w_zs))),
            "total_turn_deg": total_turn_deg,
            "turn_density_deg_per_km": turn_density_deg_per_km,
            "sharp_turn_sec": sharp_turn_sec,
            "sharp_turn_frac": sharp_turn_frac,
            "drift_pct": drift_pct,
        })
    return pd.DataFrame(results)

def main():
    loader = GenericDataLoader()
    trip_sm = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv")
    trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")
    trip_s2 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S2.csv")

    df_sm = pd.read_csv(r"C:\Users\carpe\SIH\artifacts\phase4_unseen_sm_benchmark_results.csv")
    df_50_res = pd.read_csv(r"C:\Users\carpe\SIH\artifacts\phase4_map_matching_50_scenarios_results.csv")
    df_50_sweep = pd.read_csv(r"C:\Users\carpe\SIH\artifacts\randomized_blackout_sweep_results.csv")
    
    # Merge
    df_50 = df_50_sweep.copy()
    df_50["drift_pct"] = df_50_res["matched_drift_pct"].values

    res_sm = analyze_dataset_scenarios(trip_sm, df_sm, is_sm=True)

    df_s1 = df_50[df_50["trip"].str.contains("S-S1")]
    df_s2 = df_50[df_50["trip"].str.contains("S-S2")]
    res_s1 = analyze_dataset_scenarios(trip_s1, df_s1)
    res_s2 = analyze_dataset_scenarios(trip_s2, df_s2)
    res_50 = pd.concat([res_s1, res_s2])

    print("=" * 85)
    print("COMPARATIVE SCENARIO MIX ANALYSIS: S-M (35 SCENARIOS) vs 50-SCENARIO BENCHMARK")
    print("=" * 85)

    def print_metric_row(name, sm_val, s50_val, s1_val=None, s2_val=None):
        extra = f" [S1={s1_val}, S2={s2_val}]" if s1_val is not None else ""
        print(f"  {name:<38}: S-M={sm_val:<15} | 50-Sweep={s50_val:<15}{extra}")

    print("\n1. PERFORMANCE MEDIANS & DRIFT:")
    print("-" * 75)
    print_metric_row("Phase 4 Median Drift", f"{res_sm['drift_pct'].median():.2f}%", f"{res_50['drift_pct'].median():.2f}%", f"{res_s1['drift_pct'].median():.2f}%", f"{res_s2['drift_pct'].median():.2f}%")
    print_metric_row("Phase 4 P90 Drift", f"{res_sm['drift_pct'].quantile(0.90):.2f}%", f"{res_50['drift_pct'].quantile(0.90):.2f}%")
    print_metric_row("Phase 4 Pass Rate (<10%)", f"{(res_sm['drift_pct'] < 10.0).mean()*100:.1f}%", f"{(res_50['drift_pct'] < 10.0).mean()*100:.1f}%")

    print("\n2. TURN DENSITY & DYNAMICS:")
    print("-" * 75)
    print_metric_row("Turn Density (deg/km) Median", f"{res_sm['turn_density_deg_per_km'].median():.1f} °/km", f"{res_50['turn_density_deg_per_km'].median():.1f} °/km", f"{res_s1['turn_density_deg_per_km'].median():.1f}", f"{res_s2['turn_density_deg_per_km'].median():.1f}")
    print_metric_row("Turn Density (deg/km) Mean", f"{res_sm['turn_density_deg_per_km'].mean():.1f} °/km", f"{res_50['turn_density_deg_per_km'].mean():.1f} °/km")
    print_metric_row("Sharp Turn Time Fraction (>2°/s)", f"{res_sm['sharp_turn_frac'].mean()*100:.1f}%", f"{res_50['sharp_turn_frac'].mean()*100:.1f}%", f"{res_s1['sharp_turn_frac'].mean()*100:.1f}%", f"{res_s2['sharp_turn_frac'].mean()*100:.1f}%")
    print_metric_row("Mean |omega_z| during outage", f"{res_sm['mean_abs_wz'].mean():.2f} °/s", f"{res_50['mean_abs_wz'].mean():.2f} °/s")

    print("\n3. DISTANCE & OUTAGE DURATION:")
    print("-" * 75)
    print_metric_row("Median Outage Distance", f"{res_sm['dist_m'].median():.1f} m", f"{res_50['dist_m'].median():.1f} m")
    print_metric_row("Mean Outage Distance", f"{res_sm['dist_m'].mean():.1f} m", f"{res_50['dist_m'].mean():.1f} m")
    print_metric_row("Short Outages (<100m) Fraction", f"{(res_sm['dist_m'] < 100).mean()*100:.1f}%", f"{(res_50['dist_m'] < 100).mean()*100:.1f}%")
    print_metric_row("Median Outage Duration", f"{res_sm['duration_s'].median():.1f} s", f"{res_50['duration_s'].median():.1f} s")

    print("\n4. SPEED & STATIONARY PROFILE:")
    print("-" * 75)
    print_metric_row("Average Speed", f"{res_sm['avg_speed_mps'].mean()*3.6:.1f} km/h", f"{res_50['avg_speed_mps'].mean()*3.6:.1f} km/h")
    print_metric_row("Low Speed (<3 m/s) Fraction", f"{(res_sm['avg_speed_mps'] < 3.0).mean()*100:.1f}%", f"{(res_50['avg_speed_mps'] < 3.0).mean()*100:.1f}%")

    print("\n5. GNSS STALENESS AT BLACKOUT ENTRY:")
    print("-" * 75)
    print_metric_row("GNSS Staleness (Mean)", f"{res_sm['stale_s'].mean():.2f} s", f"{res_50['stale_s'].mean():.2f} s")
    print_metric_row("GNSS Staleness (Max)", f"{res_sm['stale_s'].max():.2f} s", f"{res_50['stale_s'].max():.2f} s")
    print("=" * 85)

if __name__ == "__main__":
    main()
