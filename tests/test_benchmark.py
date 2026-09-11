"""
Unit tests for the Benchmark Harness and Naive Baseline Filter.
"""

import unittest
import numpy as np

from sih.core.contracts import IMUSample, GNSSSample
from sih.data.loader import TripSequence
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner, BenchmarkResult
from sih.fusion.naive import NaiveDeadReckoningFilter


class TestBenchmarkHarness(unittest.TestCase):

    def test_benchmark_runner_synthetic(self):
        # Create synthetic straight-line constant velocity trip: 10 m/s East for 100 seconds (1000m)
        dt = 0.1  # 10Hz
        n_samples = 1000  # 100 seconds
        timestamps_ns = [int(i * dt * 1e9) for i in range(n_samples)]

        imu_samples = []
        for ts in timestamps_ns:
            # 0 net acceleration in vehicle body, gravity = 9.80665 Down
            imu_samples.append(IMUSample(
                timestamp_ns=ts,
                accel=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            ))

        # GNSS every 1s (10 ticks): 10 m/s moving East
        ref_lat, ref_lon, ref_alt = 12.9716, 77.5946, 920.0
        gnss_samples = []
        for i in range(0, n_samples, 10):
            # ~10 m East per second -> ~0.000092 deg Lon per second
            lon_offset = (i * dt * 10.0) / (111320.0 * np.cos(np.radians(ref_lat)))
            gnss_samples.append(GNSSSample(
                timestamp_ns=timestamps_ns[i],
                latitude_deg=ref_lat,
                longitude_deg=ref_lon + lon_offset,
                altitude_m=ref_alt,
                speed_mps=10.0,
                bearing_deg=90.0,  # East
                accuracy_h_m=3.0,
            ))

        trip = TripSequence(
            trip_id="synthetic_straight",
            imu_samples=imu_samples,
            gnss_samples=gnss_samples,
            reference_lat_deg=ref_lat,
            reference_lon_deg=ref_lon,
            reference_alt_m=ref_alt,
            total_gnss_distance_m=1000.0,
            duration_s=100.0,
        )

        runner = BenchmarkRunner()
        blackout = BlackoutConfig(
            start_time_s=20.0,
            duration_s=30.0,
            name="synthetic_30s_blackout"
        )

        res = runner.run_trip(trip, blackout)
        self.assertIsInstance(res, BenchmarkResult)
        self.assertEqual(res.trip_id, "synthetic_straight")
        self.assertAlmostEqual(res.blackout_duration_s, 30.0, delta=1.0)
        self.assertAlmostEqual(res.blackout_distance_m, 300.0, delta=10.0)

    def test_metrics_calculations(self):
        from sih.eval.metrics import (
            compute_drift_percentage,
            compute_rmse,
            compute_mae,
            decompose_along_cross_track,
            evaluate_blackout_metrics,
        )

        # Drift percentage: 10m error over 100m distance = 10%
        drift = compute_drift_percentage(10.0, 100.0)
        self.assertAlmostEqual(drift, 10.0, places=4)

        # Zero distance guard
        self.assertEqual(compute_drift_percentage(10.0, 0.0), 0.0)

        # RMSE and MAE
        pred = np.array([1.0, 2.0, 3.0])
        gt = np.array([1.0, 3.0, 5.0])
        self.assertAlmostEqual(compute_mae(pred, gt), 1.0, places=4)
        self.assertAlmostEqual(compute_rmse(pred, gt), np.sqrt(5.0 / 3.0), places=4)

        # Along-track and cross-track decomposition
        # Vehicle drives East from (0, 0) to (100, 0)
        gt_track = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        # Estimate leads by 5m (East) and deviates 2m North (Left)
        est_track = np.column_stack([np.linspace(5, 105, 50), np.full(50, 2.0)])

        along, cross = decompose_along_cross_track(est_track, gt_track)
        self.assertEqual(len(along), 50)
        self.assertEqual(len(cross), 50)
        self.assertAlmostEqual(along[25], 5.0, delta=0.1)
        self.assertAlmostEqual(cross[25], 2.0, delta=0.1)

        # Full blackout metrics evaluation
        metrics = evaluate_blackout_metrics(est_track, gt_track, distance_m=100.0)
        self.assertAlmostEqual(metrics["drift_pct"], np.hypot(5.0, 2.0) / 100.0 * 100.0, delta=0.1)
        self.assertAlmostEqual(metrics["along_track_rmse_m"], 5.0, delta=0.1)
        self.assertAlmostEqual(metrics["cross_track_rmse_m"], 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
