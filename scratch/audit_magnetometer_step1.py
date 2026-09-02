import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.data.geo import geodetic_to_enu

ARTIFACT_DIR = r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92"

def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    print("=== STEP 1: MAGNETOMETER RELIABILITY AUDIT ===", flush=True)

    file_path = r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv"
    df_raw = pd.read_csv(file_path, encoding="latin-1")
    print(f"Loaded {file_path}. Columns in dataset:", flush=True)
    print(df_raw.columns.tolist())

    mag_cols = [c for c in df_raw.columns if "mag" in c.lower() or "mx" in c.lower() or "my" in c.lower() or "mz" in c.lower()]
    print(f"Detected Magnetometer Columns: {mag_cols}", flush=True)

    loader = GenericDataLoader()
    trip = loader.load_file(file_path)

    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples:
        calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]

    # Check if mag_uncalibrated exists in imu samples or raw dataframe
    has_mag = hasattr(trip.imu_samples[0], "mag_x") and trip.imu_samples[0].mag_x is not None

    if not has_mag and len(mag_cols) > 0:
        print("Extracting magnetometer columns directly from DataFrame...", flush=True)
        # Check raw dataframe column names
        mx_col = next((c for c in mag_cols if "x" in c.lower()), None)
        my_col = next((c for c in mag_cols if "y" in c.lower()), None)
        mz_col = next((c for c in mag_cols if "z" in c.lower()), None)
        
        if mx_col and my_col and mz_col:
            m_x = df_raw[mx_col].values
            m_y = df_raw[my_col].values
            m_z = df_raw[mz_col].values
            has_mag = True
            print(f"Found mag signals: {mx_col}, {my_col}, {mz_col}", flush=True)

    if not has_mag:
        print("\nERROR: No magnetometer data available in S-S1.csv!", flush=True)
        return

    # Check 1: Straight-Line Highway Segment
    # Find a long straight segment: speed > 15 m/s, angular rate < 1 deg/s for at least 30s
    gnss_df = pd.DataFrame([{
        "ts": g.timestamp_ns,
        "lat": g.latitude_deg,
        "lon": g.longitude_deg,
        "speed": g.speed_mps,
        "bearing": g.bearing_deg
    } for g in trip.gnss_samples if g.speed_mps is not None and g.speed_mps > 15.0 and g.bearing_deg is not None])

    print(f"\nTotal valid GNSS samples with speed > 15 m/s: {len(gnss_df)}", flush=True)

    mag_headings_deg = []
    gnss_bearings_deg = []
    diffs_deg = []
    straight_diffs_deg = []
    turning_diffs_deg  = []
    mag_headings_deg   = []
    gnss_bearings_deg  = []

    mx_col = [c for c in df_raw.columns if "MAGNETIC" in c and "X" in c][0]
    my_col = [c for c in df_raw.columns if "MAGNETIC" in c and "Y" in c][0]
    mz_col = [c for c in df_raw.columns if "MAGNETIC" in c and "Z" in c][0]

    raw_m_x = pd.to_numeric(df_raw[mx_col], errors='coerce').fillna(0).values
    raw_m_y = pd.to_numeric(df_raw[my_col], errors='coerce').fillna(0).values
    raw_m_z = pd.to_numeric(df_raw[mz_col], errors='coerce').fillna(0).values

    lev_m_x = []
    lev_m_y = []

    n_samples = len(trip.imu_samples)
    R_mount = calibrator.alignment.R_phone_to_vehicle.as_matrix() if calibrator.is_calibrated else np.eye(3)

    for i in range(min(n_samples, len(raw_m_x))):
        imu = trip.imu_samples[i]
        cal = calib_samples[i]
        m_raw = np.array([raw_m_x[i], raw_m_y[i], raw_m_z[i]], dtype=np.float64)

        # Vehicle frame leveling
        m_veh = R_mount @ m_raw
        lev_m_x.append(m_veh[0])
        lev_m_y.append(m_veh[1])

        # Mag heading (atan2(my, mx)) in vehicle frame
        mag_hdg_rad = np.arctan2(m_veh[1], m_veh[0])
        mag_hdg_deg = (np.degrees(mag_hdg_rad) + 360.0) % 360.0

        t_curr = imu.timestamp_ns
        w_z = abs(cal.gyro_vehicle[2]) # rad/s

        matching_g = [g for g in trip.gnss_samples if abs(g.timestamp_ns - t_curr) < 2e8 and g.speed_mps is not None and g.speed_mps > 3.0 and g.bearing_deg is not None]
        if len(matching_g) > 0:
            g = matching_g[0]
            gnss_b = g.bearing_deg % 360.0
            diff = (mag_hdg_deg - gnss_b + 180.0) % 360.0 - 180.0

            if w_z < np.radians(1.0): # Straight segment
                straight_diffs_deg.append(diff)
            elif w_z > np.radians(3.0): # Turning segment
                turning_diffs_deg.append(diff)

            mag_headings_deg.append(mag_hdg_deg)
            gnss_bearings_deg.append(gnss_b)

    straight_diffs_deg = np.array(straight_diffs_deg)
    turning_diffs_deg  = np.array(turning_diffs_deg)

    print("\n--------------------------------------------------------------------------", flush=True)
    print(" 1. STRAIGHT-LINE HIGHWAY COMPARISON (Mag Heading vs GNSS Bearing)", flush=True)
    print("--------------------------------------------------------------------------", flush=True)
    print(f" Matched Straight Samples: {len(straight_diffs_deg)}", flush=True)
    if len(straight_diffs_deg) > 0:
        s_mean = float(np.mean(straight_diffs_deg))
        s_std  = float(np.std(straight_diffs_deg))
        print(f" Mean Difference (Mag - GNSS): {s_mean:6.2f}°", flush=True)
        print(f" Std Deviation:               {s_std:6.2f}°", flush=True)
    else:
        s_mean, s_std = 0.0, 0.0

    print("\n--------------------------------------------------------------------------", flush=True)
    print(" 2. TURNING SEGMENT COMPARISON (Mag Heading vs GNSS Bearing)", flush=True)
    print("--------------------------------------------------------------------------", flush=True)
    print(f" Matched Turning Samples:  {len(turning_diffs_deg)}", flush=True)
    if len(turning_diffs_deg) > 0:
        t_mean = float(np.mean(turning_diffs_deg))
        t_std  = float(np.std(turning_diffs_deg))
        print(f" Mean Difference during Turns: {t_mean:6.2f}°", flush=True)
        print(f" Std Deviation during Turns:   {t_std:6.2f}°", flush=True)
    else:
        t_mean, t_std = 0.0, 0.0

    # Check 3: Hard/Soft-Iron Distortion Plot (Raw Magnetometer X vs Y)
    raw_m_x = np.array(raw_m_x)
    raw_m_y = np.array(raw_m_y)
    lev_m_x = np.array(lev_m_x)
    lev_m_y = np.array(lev_m_y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(raw_m_x[::5], raw_m_y[::5], c="blue", alpha=0.3, s=10, label="Raw Mag (Mx vs My)")
    ax1.set_title("Raw Magnetometer X-Y Scatter (Sensor Frame)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mag X (uT)", fontsize=11)
    ax1.set_ylabel("Mag Y (uT)", fontsize=11)
    ax1.axhline(0, color="gray", linestyle="--")
    ax1.axvline(0, color="gray", linestyle="--")
    ax1.axis("equal")

    ax2.scatter(lev_m_x[::5], lev_m_y[::5], c="red", alpha=0.3, s=10, label="Leveled Mag (Veh X vs Y)")
    ax2.set_title("Leveled Magnetometer X-Y Scatter (Vehicle Frame)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Vehicle Mag X (uT)", fontsize=11)
    ax2.set_ylabel("Vehicle Mag Y (uT)", fontsize=11)
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.axvline(0, color="gray", linestyle="--")
    ax2.axis("equal")

    plt.tight_layout()
    out_img = os.path.join(ARTIFACT_DIR, "magnetometer_distortion_audit.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"\nSaved distortion plot: {out_img}", flush=True)

    # Evaluate Center Offset (Hard-Iron distortion indicator)
    center_x = (np.max(raw_m_x) + np.min(raw_m_x)) / 2.0
    center_y = (np.max(raw_m_y) + np.min(raw_m_y)) / 2.0
    radius_x = (np.max(raw_m_x) - np.min(raw_m_x)) / 2.0
    radius_y = (np.max(raw_m_y) - np.min(raw_m_y)) / 2.0

    print(f"\n--- Check 3: Distortion Analysis ---", flush=True)
    print(f"Magnetometer Center Offset: Center X = {center_x:6.2f} uT, Center Y = {center_y:6.2f} uT", flush=True)
    print(f"Magnetometer Span: Radius X = {radius_x:6.2f} uT, Radius Y = {radius_y:6.2f} uT", flush=True)

    # Determine Verdict
    # If std > 25 deg or center offset > 0.3 * radius, distortion is severe
    is_usable = (s_std < 25.0) and (abs(center_x) < 0.3 * radius_x) and (abs(center_y) < 0.3 * radius_y)
    verdict_str = "PASS" if is_usable else "FAIL (UNUSABLE)"

    print(f"\n==========================================================================", flush=True)
    print(f"         STEP 1 MAGNETOMETER AUDIT VERDICT: [{verdict_str}]              ", flush=True)
    print(f"==========================================================================", flush=True)

if __name__ == "__main__":
    main()
