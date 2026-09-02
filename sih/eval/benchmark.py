"""
Evaluation & Blackout Benchmark Harness.

Simulates GNSS outages (blackouts) across configurable durations/distances on real trip data,
evaluates Dead Reckoning drift against the SIH benchmark (< 10% drift), and generates
comparative trajectory plots and error logs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sih.core.contracts import IMUSample, GNSSSample, FusedPosition
from sih.core.pipeline import IDRPipeline, assemble_pipeline
from sih.core.config import PipelineConfig
from sih.data.loader import TripSequence
from sih.data.geo import geodetic_to_enu, haversine_distance_m, compute_cumulative_distance


@dataclass
class BlackoutConfig:
    """
    Configuration for an injected GNSS outage.
    """
    start_time_s: float = 60.0          # Seconds after trip start to cut GNSS
    duration_s: Optional[float] = 60.0  # Outage duration in seconds
    target_distance_m: Optional[float] = None  # Outage by distance (e.g. 50m, 500m, 1000m)
    name: str = "gnss_blackout"


@dataclass
class BenchmarkResult:
    """
    Quantitative performance metrics for a single blackout evaluation run.
    """
    trip_id: str
    scenario_name: str
    blackout_start_s: float
    blackout_end_s: float
    blackout_duration_s: float
    blackout_distance_m: float
    final_position_error_m: float
    drift_percentage: float             # (final_error / blackout_distance) * 100%
    max_error_m: float
    rmse_position_m: float
    benchmark_passed: bool              # True if drift_percentage < 10.0%
    history_df: pd.DataFrame = field(repr=False)


class BenchmarkRunner:
    """
    Harness to run full pipeline simulations on real trip sequences with injected blackouts.
    """
    def __init__(self, pipeline: Optional[IDRPipeline] = None, config: Optional[PipelineConfig] = None) -> None:
        if pipeline is not None:
            self.pipeline = pipeline
        elif config is not None:
            self.pipeline = assemble_pipeline(config)
        else:
            self.pipeline = assemble_pipeline(PipelineConfig())

    def run_trip(self, trip: TripSequence, blackout: BlackoutConfig, full_trip: bool = False) -> BenchmarkResult:
        """
        Execute evaluation on a trip with an injected GNSS blackout.
        """
        if len(trip.imu_samples) == 0:
            raise ValueError(f"Trip {trip.trip_id} has no IMU samples.")
        if len(trip.gnss_samples) == 0:
            raise ValueError(f"Trip {trip.trip_id} has no GNSS samples.")

        t0_ns = trip.imu_samples[0].timestamp_ns
        blackout_start_ns = t0_ns + int(blackout.start_time_s * 1e9)

        # Precompute ground truth GNSS trajectory in ENU and timestamps
        gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
        gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples], dtype=np.float64)
        gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples], dtype=np.float64)
        gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples], dtype=np.float64)

        ref_lat = trip.reference_lat_deg
        ref_lon = trip.reference_lon_deg
        ref_alt = trip.reference_alt_m

        gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, ref_lat, ref_lon, ref_alt)

        # Determine blackout end timestamp
        if blackout.target_distance_m is not None:
            # Find timestamp where distance from blackout start reaches target
            start_idx = np.searchsorted(gnss_ts, blackout_start_ns)
            start_idx = min(start_idx, len(gnss_ts) - 1)
            cum_d = 0.0
            end_idx = start_idx
            for i in range(start_idx, len(gnss_lats) - 1):
                d = haversine_distance_m(gnss_lats[i], gnss_lons[i], gnss_lats[i + 1], gnss_lons[i + 1])
                cum_d += d
                if cum_d >= blackout.target_distance_m:
                    end_idx = i + 1
                    break
            blackout_end_ns = int(gnss_ts[end_idx])
        else:
            dur_s = blackout.duration_s if blackout.duration_s is not None else 60.0
            blackout_end_ns = blackout_start_ns + int(dur_s * 1e9)

        # Reset pipeline with initial GNSS fix
        self.pipeline.reset(initial_gnss=trip.gnss_samples[0])

        # State tracking
        records: List[Dict[str, Any]] = []
        gnss_idx = 0
        n_gnss = len(trip.gnss_samples)

        from tqdm import tqdm
        pbar = tqdm(trip.imu_samples, desc=f"Evaluating {trip.trip_id} [{blackout.name}]", leave=False, dynamic_ncols=True)

        for imu in pbar:
            t_curr = imu.timestamp_ns
            if not full_trip and t_curr > blackout_end_ns + int(3e9):
                break

            # Check if there are GNSS updates at or before this IMU tick
            while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                g_sample = trip.gnss_samples[gnss_idx]
                # Is GNSS available or inside blackout?
                in_blackout = (blackout_start_ns <= g_sample.timestamp_ns <= blackout_end_ns)
                if not in_blackout:
                    self.pipeline.process_gnss(g_sample)
                gnss_idx += 1

            # Process IMU tick
            calib, vel, fused, matched = self.pipeline.process_imu(imu)

            # Interpolate ground truth GNSS position at current timestamp
            gt_e = np.interp(t_curr, gnss_ts, gnss_enu[:, 0])
            gt_n = np.interp(t_curr, gnss_ts, gnss_enu[:, 1])
            gt_u = np.interp(t_curr, gnss_ts, gnss_enu[:, 2])

            est_e = fused.position_enu_m[0]
            est_n = fused.position_enu_m[1]
            est_u = fused.position_enu_m[2]

            error_m = float(np.sqrt((est_e - gt_e) ** 2 + (est_n - gt_n) ** 2))
            is_blackout_active = (blackout_start_ns <= t_curr <= blackout_end_ns)

            records.append({
                "time_s": (t_curr - t0_ns) * 1e-9,
                "timestamp_ns": t_curr,
                "est_lat": fused.latitude_deg,
                "est_lon": fused.longitude_deg,
                "est_e": est_e,
                "est_n": est_n,
                "est_u": est_u,
                "gt_e": gt_e,
                "gt_n": gt_n,
                "gt_u": gt_u,
                "error_m": error_m,
                "heading_deg": np.degrees(fused.heading_rad),
                "speed_mps": np.linalg.norm(fused.velocity_enu_mps[:2]),
                "in_blackout": is_blackout_active,
                "mode": fused.mode,
            })

        df_hist = pd.DataFrame(records)

        # Slice blackout window for metrics
        df_bo = df_hist[df_hist["in_blackout"]].copy()
        if len(df_bo) == 0:
            # Outage outside recorded timeframe
            df_bo = df_hist.iloc[-100:].copy()

        # Compute ground truth distance during blackout
        gt_e_bo = df_bo["gt_e"].values
        gt_n_bo = df_bo["gt_n"].values
        diffs = np.sqrt(np.diff(gt_e_bo) ** 2 + np.diff(gt_n_bo) ** 2)
        blackout_dist_m = float(np.sum(diffs))
        if blackout_dist_m < 1.0:
            blackout_dist_m = 1.0  # Guard against division by zero

        final_err = float(df_bo["error_m"].iloc[-1])
        max_err = float(df_bo["error_m"].max())
        rmse_err = float(np.sqrt(np.mean(df_bo["error_m"] ** 2)))
        drift_pct = (final_err / blackout_dist_m) * 100.0
        passed = (drift_pct < 10.0)

        actual_dur_s = float(df_bo["time_s"].iloc[-1] - df_bo["time_s"].iloc[0])

        return BenchmarkResult(
            trip_id=trip.trip_id,
            scenario_name=blackout.name,
            blackout_start_s=float(df_bo["time_s"].iloc[0]),
            blackout_end_s=float(df_bo["time_s"].iloc[-1]),
            blackout_duration_s=actual_dur_s,
            blackout_distance_m=blackout_dist_m,
            final_position_error_m=final_err,
            drift_percentage=drift_pct,
            max_error_m=max_err,
            rmse_position_m=rmse_err,
            benchmark_passed=passed,
            history_df=df_hist,
        )


def plot_benchmark_result(result: BenchmarkResult, output_path: str = "artifacts/benchmark_plot.png") -> str:
    """
    Generate clean, publication-quality diagnostic plots for a benchmark blackout run.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = result.history_df

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 2D Trajectory Plot (East vs North)
    ax1 = axes[0, 0]
    ax1.plot(df["gt_e"], df["gt_n"], "k-", label="Ground Truth GNSS", linewidth=2.0, alpha=0.8)

    # Pre-blackout
    df_pre = df[df["time_s"] < result.blackout_start_s]
    if len(df_pre) > 0:
        ax1.plot(df_pre["est_e"], df_pre["est_n"], "g--", label="Estimated (GNSS-aided)", alpha=0.7)

    # Blackout window
    df_bo = df[df["in_blackout"]]
    if len(df_bo) > 0:
        ax1.plot(df_bo["est_e"], df_bo["est_n"], "r-", label="Estimated (Blackout INS)", linewidth=2.5)
        # Highlight start and end
        ax1.scatter([df_bo["gt_e"].iloc[0]], [df_bo["gt_n"].iloc[0]], color="black", s=80, zorder=5, label="Blackout Start")
        ax1.scatter([df_bo["gt_e"].iloc[-1]], [df_bo["gt_n"].iloc[-1]], color="blue", s=80, zorder=5, label="Ground Truth End")
        ax1.scatter([df_bo["est_e"].iloc[-1]], [df_bo["est_n"].iloc[-1]], color="red", s=80, zorder=5, label="Drifted Estimate End")

    ax1.set_title(f"Trajectory (ENU Frame): {result.trip_id} - {result.scenario_name}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("East (meters)")
    ax1.set_ylabel("North (meters)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Position Error over Time
    ax2 = axes[0, 1]
    ax2.plot(df["time_s"], df["error_m"], "r-", linewidth=1.8, label="Horizontal Error (m)")
    ax2.axvspan(result.blackout_start_s, result.blackout_end_s, color="yellow", alpha=0.25, label="Blackout Window")
    ax2.axhline(result.final_position_error_m, color="red", linestyle=":", label=f"Final Error: {result.final_position_error_m:.1f} m")
    ax2.set_title(f"Position Error vs Time (Drift: {result.drift_percentage:.1f}%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Error (meters)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Speed Profile
    ax3 = axes[1, 0]
    ax3.plot(df["time_s"], df["speed_mps"], "b-", linewidth=1.5, label="Estimated Speed (m/s)")
    ax3.axvspan(result.blackout_start_s, result.blackout_end_s, color="yellow", alpha=0.25)
    ax3.set_title("Estimated Vehicle Speed Profile", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Elapsed Time (s)")
    ax3.set_ylabel("Speed (m/s)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Heading Profile
    ax4 = axes[1, 1]
    ax4.plot(df["time_s"], df["heading_deg"], "purple", linewidth=1.5, label="Heading (° from North)")
    ax4.axvspan(result.blackout_start_s, result.blackout_end_s, color="yellow", alpha=0.25)
    ax4.set_title("Estimated Heading / Azimuth", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Elapsed Time (s)")
    ax4.set_ylabel("Heading (degrees)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
