"""
Bootstrap: writes the full sweep script and executes it.
Run from C:\\Users\\carpe\\SIH with: python scratch\\bootstrap_sweep.py
"""
import os

SWEEP_SCRIPT = r"C:\Users\carpe\SIH\scratch\run_fixed_sweep.py"

CODE = """\
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, r"C:\\\\Users\\\\carpe\\\\SIH")
from sih.data.loader import GenericDataLoader
from sih.core.contracts import CalibratedSample, VelocityEstimate, GNSSSample
from sih.data.geo import geodetic_to_enu, enu_to_geodetic
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.calibration.mount import MountCalibrator
from sih.fusion.es_ekf import ErrorStateEKF
import torch

ARTIFACT_DIR = r"C:\\\\Users\\\\carpe\\\\SIH\\\\artifacts"
SWEEP_CSV    = os.path.join(ARTIFACT_DIR, "randomized_blackout_sweep_results.csv")
DATA_DIR     = r"C:\\\\Users\\\\carpe\\\\SIH\\\\data\\\\raw\\\\iovnbd_trips"

def load_trip(trip_label):
    name_map = {"S-S1 (Highway)": "S-S1.csv", "S-S2 (Urban)": "S-S2.csv"}
    fname = name_map.get(trip_label, "S-S1.csv")
    return pd.read_csv(os.path.join(DATA_DIR, fname))

def get_ai_velocities(df, model, calibrator):
    try:
        from sih.models.tcn_attention import WINDOW_SIZE, STEP_SIZE
    except ImportError:
        WINDOW_SIZE, STEP_SIZE = 200, 10
    speeds = np.zeros(len(df), dtype=np.float32)
    device = next(model.parameters()).device
    cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    arr = df[cols].values.astype(np.float32)
    for i in range(len(arr)):
        cs = calibrator.calibrate(arr[i, :3], arr[i, 3:], 0.0)
        arr[i, :3] = cs.accel_vehicle.astype(np.float32)
        arr[i, 3:] = cs.gyro_vehicle.astype(np.float32)
    for start in range(0, len(arr) - WINDOW_SIZE, STEP_SIZE):
        window = torch.tensor(arr[start:start + WINDOW_SIZE]).unsqueeze(0).to(device)
        with torch.no_grad():
            v = model(window).squeeze().cpu().item()
        speeds[start + WINDOW_SIZE // 2] = max(0.0, v)
    return speeds

def run_scenario(df, row, speeds, calibrator):
    t_col = "timestamp_ns" if "timestamp_ns" in df.columns else "time_ns"
    times_ns = df[t_col].values.astype(np.float64)
    times_s  = (times_ns - times_ns[0]) * 1e-9

    t_start  = float(row["start_time_s"])
    duration = float(row["duration_s"])
    t_end    = t_start + duration
    pre      = int(np.searchsorted(times_s, t_start, side="left"))
    post     = int(np.searchsorted(times_s, t_end,   side="right"))

    if pre < 5 or post >= len(df) - 1:
        return None

    def make_gnss(idx):
        r = df.iloc[idx]
        return GNSSSample(
            timestamp_ns=int(r[t_col]),
            latitude_deg=float(r["latitude"]),
            longitude_deg=float(r["longitude"]),
            altitude_m=float(r.get("altitude", 0.0)),
            speed_mps=float(r.get("speed", 0.0)),
            bearing_deg=float(r.get("bearing", 0.0)),
            accuracy_h_m=float(r.get("accuracy", 2.0)),
            is_valid=True,
        )

    def make_cal(idx):
        r = df.iloc[idx]
        raw_a = np.array([r["accel_x"], r["accel_y"], r["accel_z"]], dtype=np.float64)
        raw_g = np.array([r["gyro_x"],  r["gyro_y"],  r["gyro_z"]],  dtype=np.float64)
        cs = calibrator.calibrate(raw_a, raw_g, 0.0)
        return CalibratedSample(
            timestamp_ns=int(r[t_col]),
            accel_vehicle=cs.accel_vehicle,
            gyro_vehicle=cs.gyro_vehicle,
            temperature_c=0.0,
        )

    def make_vel(idx):
        v = float(speeds[idx])
        return VelocityEstimate(
            timestamp_ns=int(df[t_col].iloc[idx]),
            forward_speed_mps=v,
            speed_variance=0.3,
            motion_state="MOVING" if v > 0.2 else "STATIONARY",
        )

    ekf = ErrorStateEKF(enable_nhc=True, enable_zupt=True)
    ekf.init_from_gnss(make_gnss(pre - 1))

    for idx in range(max(0, pre - 300), pre):
        ekf.predict(make_cal(idx), make_vel(idx))
        ekf.update_gnss(make_gnss(idx))

    bg_z_before = float(ekf.b_g[2])
    bg_z_trace  = []
    nhc_trace   = []

    for idx in range(pre, post):
        ekf.predict(make_cal(idx), make_vel(idx))
        bg_z_trace.append(float(ekf.b_g[2]))
        nhc_trace.append(float(ekf.last_nhc_dtheta_deg))

    ref      = np.array([df["latitude"].iloc[0], df["longitude"].iloc[0], 0.0])
    gt_enu   = geodetic_to_enu(df["latitude"].iloc[post],  df["longitude"].iloc[post],  0.0, ref[0], ref[1], ref[2])
    s_enu    = geodetic_to_enu(df["latitude"].iloc[pre],   df["longitude"].iloc[pre],   0.0, ref[0], ref[1], ref[2])
    pred_enu = ekf.p

    err_vec   = pred_enu[:2] - gt_enu[:2]
    final_err = float(np.linalg.norm(err_vec))
    distance  = float(row["distance_m"])
    drift_pct = 100.0 * final_err / max(distance, 1.0)

    fwd_dir = gt_enu[:2] - s_enu[:2]
    fwd_len = np.linalg.norm(fwd_dir)
    if fwd_len > 1.0:
        fwd_hat   = fwd_dir / fwd_len
        lat_hat   = np.array([-fwd_hat[1], fwd_hat[0]])
        err_along = float(np.dot(err_vec, fwd_hat))
        err_cross = float(np.dot(err_vec, lat_hat))
    else:
        err_along = float(err_vec[0])
        err_cross = float(err_vec[1])

    gyro_vals = df["gyro_z"].iloc[pre:post].values.astype(np.float64)
    dt_arr    = np.diff(times_s[pre:post + 1])
    dt_arr    = np.clip(dt_arr, 0.0, 1.0)
    total_turn_deg = float(np.sum(np.abs(gyro_vals[:len(dt_arr)]) * dt_arr) * 180.0 / np.pi)

    return dict(
        final_error_m=final_err, drift_pct=drift_pct,
        err_along_m=err_along,   err_cross_m=err_cross,
        total_turn_deg=total_turn_deg,
        bg_z_before=bg_z_before, bg_z_trace=bg_z_trace, nhc_trace=nhc_trace,
    )

def main():
    print("Loading sweep scenarios...")
    sweep = pd.read_csv(SWEEP_CSV)

    print("Loading model...")
    # Try the canonical path first, then fall back to checkpoints subdirectory
    mp = r"C:\\\\Users\\\\carpe\\\\SIH\\\\models\\\\tcn_attention_best.pth"
    if not os.path.exists(mp):
        mp = r"C:\\\\Users\\\\carpe\\\\SIH\\\\models\\\\checkpoints\\\\best_velocity_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TCNAttentionVelocityModel()
    ckpt   = torch.load(mp, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    # If the model was saved with a larger architecture, we adapt
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)

    calibrator = MountCalibrator()

    trip_cache = {}
    print("Precomputing AI velocities...")
    for trip_label in sweep["trip"].unique():
        df     = load_trip(trip_label)
        speeds = get_ai_velocities(df, model, calibrator)
        trip_cache[trip_label] = (df, speeds)
        print(f"  {trip_label}: {len(df)} rows, speed max {speeds.max():.2f} m/s")

    results     = []
    worst_traces = []
    print(f"\\nRunning {len(sweep)} scenarios...")
    for i, row in sweep.iterrows():
        trip_label = row["trip"]
        df, speeds = trip_cache[trip_label]
        res = run_scenario(df, row, speeds, calibrator)
        if res is None:
            print(f"  [{i+1:2d}/{len(sweep)}] SKIP")
            continue
        results.append(dict(
            trip=trip_label, scenario=row["scenario"],
            start_time_s=row["start_time_s"], duration_s=row["duration_s"],
            distance_m=row["distance_m"],
            old_error_m=row["final_error_m"], old_drift_pct=row["drift_pct"],
            new_error_m=res["final_error_m"],  new_drift_pct=res["drift_pct"],
            err_along_m=res["err_along_m"],    err_cross_m=res["err_cross_m"],
            total_turn_deg=res["total_turn_deg"], bg_z_before=res["bg_z_before"],
        ))
        if len(worst_traces) < 6 and abs(res["err_cross_m"]) > 30.0:
            worst_traces.append((row["scenario"], res["bg_z_trace"], trip_label, row["start_time_s"]))

        delta = res["final_error_m"] - row["final_error_m"]
        sign  = "+" if delta > 0 else ""
        bg_d  = np.degrees(res["bg_z_before"])
        sc    = row["scenario"][:35]
        print(f"  [{i+1:2d}/{len(sweep)}] {sc:35s}  old={row['final_error_m']:7.2f}m  new={res['final_error_m']:7.2f}m ({sign}{delta:.2f}m)  bg_z={bg_d:.4f}d/s")

    df_res  = pd.DataFrame(results)
    out_csv = os.path.join(ARTIFACT_DIR, "fixed_sweep_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\\nSaved: {out_csv}")

    print("\\n" + "=" * 74)
    print("BEFORE vs AFTER SUMMARY TABLE")
    print("=" * 74)
    for trip_label in ["S-S1 (Highway)", "S-S2 (Urban)"]:
        sub = df_res[df_res["trip"] == trip_label]
        if sub.empty: continue
        tag = trip_label.split("(")[0].strip()
        print(f"\\n{tag}  (N={len(sub)})")
        for label, ov, nv in [
            ("Drift % Mean",   sub["old_drift_pct"].mean(),            sub["new_drift_pct"].mean()),
            ("Drift % Median", sub["old_drift_pct"].median(),          sub["new_drift_pct"].median()),
            ("Drift % 90th",   sub["old_drift_pct"].quantile(0.9),     sub["new_drift_pct"].quantile(0.9)),
            ("Drift % Worst",  sub["old_drift_pct"].max(),             sub["new_drift_pct"].max()),
            ("Error m Mean",   sub["old_error_m"].mean(),              sub["new_error_m"].mean()),
            ("Error m Median", sub["old_error_m"].median(),            sub["new_error_m"].median()),
            ("Error m 90th",   sub["old_error_m"].quantile(0.9),       sub["new_error_m"].quantile(0.9)),
            ("Error m Worst",  sub["old_error_m"].max(),               sub["new_error_m"].max()),
        ]:
            d  = nv - ov
            sg = "+" if d > 0 else ""
            print(f"  {label:<20}  {ov:>12.2f}  {nv:>12.2f}  ({sg}{d:.2f})")

    print(f"\\nCOMBINED (N={len(df_res)})")
    for label, ov, nv in [
        ("Drift % Mean",   df_res["old_drift_pct"].mean(),   df_res["new_drift_pct"].mean()),
        ("Drift % Median", df_res["old_drift_pct"].median(), df_res["new_drift_pct"].median()),
        ("Drift % 90th",   df_res["old_drift_pct"].quantile(0.9), df_res["new_drift_pct"].quantile(0.9)),
        ("Drift % Worst",  df_res["old_drift_pct"].max(),    df_res["new_drift_pct"].max()),
    ]:
        d  = nv - ov
        sg = "+" if d > 0 else ""
        print(f"  {label:<20}  {ov:>12.2f}  {nv:>12.2f}  ({sg}{d:.2f})")

    # Plot 1: bg_z traces
    n   = max(1, min(6, len(worst_traces)))
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), squeeze=False)
    for ax, (sc_name, trace, trip, t_s) in zip(axes[:, 0], worst_traces):
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.plot(np.degrees(trace), lw=1.5)
        ax.set_ylabel("bg_z (deg/s)")
        ax.set_xlabel("Steps into blackout")
        ax.set_title(f"{sc_name[:40]}  ({trip})  start={t_s:.0f}s", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Gyro Bias bg_z During Blackout — Fixed EKF", fontsize=11, fontweight="bold")
    plt.tight_layout()
    bias_png = os.path.join(ARTIFACT_DIR, "fixed_gyro_bias_convergence.png")
    fig.savefig(bias_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {bias_png}")

    # Plot 2: turn correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = {"S-S1 (Highway)": "#3A7BD5", "S-S2 (Urban)": "#D55A3A"}
    for trip_label, grp in df_res.groupby("trip"):
        c  = cmap.get(trip_label, "gray")
        ls = trip_label.split("(")[0].strip()
        ax1.scatter(grp["total_turn_deg"], grp["old_error_m"].abs(), c=c, alpha=0.65, label=ls, s=55)
        ax2.scatter(grp["total_turn_deg"], grp["new_error_m"].abs(), c=c, alpha=0.65, label=ls, s=55)
    for ax, col, title in [(ax1, "old_error_m", "BEFORE (Baseline)"), (ax2, "new_error_m", "AFTER (Fixed EKF)")]:
        x = df_res["total_turn_deg"].values
        y = df_res[col].abs().values
        r, p = pearsonr(x, y)
        m, b = np.polyfit(x, y, 1)
        xs   = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1.5, label=f"r={r:.3f}  p={p:.4f}")
        ax.set_xlabel("Total Turn Angle During Blackout (deg)")
        ax.set_ylabel("Final Position Error (m)")
        ax.set_title(f"{title}  r={r:.3f}  p={p:.4f}", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        print(f"  {title}: r={r:.4f}, slope={m:.3f} m/deg")
    fig.suptitle("Turn Severity vs Position Error  Before vs After", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_png = os.path.join(ARTIFACT_DIR, "fixed_heading_error_vs_turn_severity.png")
    fig.savefig(corr_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {corr_png}")
    print("\\nDone.")

main()
"""

# Write the actual script (double-backslash was for the inner script string above)
actual_code = CODE.replace("\\\\\\\\", "\\\\").replace("\\\\n", "\\n")

with open(SWEEP_SCRIPT, "w", encoding="utf-8") as f:
    f.write(actual_code)

print(f"Written: {SWEEP_SCRIPT}")
