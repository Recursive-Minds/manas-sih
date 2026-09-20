"""
tests/test_app_step2.py
-----------------------
Unit tests for Step 2 Server Components:
1. EngineAdapterStageA:
   - Causal anti-aliasing decimation (50 Hz down to 10 Hz)
   - Mount reuse guard (< 5.0 deg tilt threshold for reuse vs discard)
   - Strict no-future-leak firewall in BLACKOUT state
2. LiveEvaluator:
   - Ground-truth comparison
   - Along-track and cross-track decomposition
   - Real-time drift % computation
3. Replay & Router Integration:
   - Direct in-memory trip replay through NavigationRouter
"""

import os
import sys
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.core.contracts import IMUSample, GNSSSample
from sih.calibration.mount import MountAlignment
from sih.data.loader import GenericDataLoader
from server.engine_adapter import EngineAdapterStageA, CausalAntiAliasFilter
from server.evaluator import LiveEvaluator
from server.router import NavigationRouter
from server.replay import replay_direct_in_memory, build_sensor_batches


class TestEngineAdapterStageA(unittest.TestCase):
    def test_anti_alias_decimation(self):
        """Simulate a 50 Hz IMU stream (dt = 20 ms) and verify 10 Hz decimation."""
        adapter = EngineAdapterStageA(reference_lat_deg=12.97, reference_lon_deg=77.59)

        t0_ns = 1_000_000_000_000
        dt_ns = 20_000_000  # 50 Hz (20 ms)

        emitted_fused = []
        for i in range(100):  # 100 samples @ 50 Hz = 2.0 seconds
            t_ns = t0_ns + i * dt_ns
            imu = IMUSample(
                timestamp_ns=t_ns,
                accel=np.array([0.1 * math.sin(i * 0.5), 0.0, 9.81], dtype=np.float64),
                gyro=np.array([0.0, 0.0, 0.01 * math.cos(i * 0.5)], dtype=np.float64),
            )
            fused = adapter.on_imu(imu)
            if fused is not None:
                emitted_fused.append(fused)

        # In 2.0 seconds, exactly ~19-20 samples should be emitted at 10.0 Hz
        self.assertGreaterEqual(len(emitted_fused), 19)
        self.assertLessEqual(len(emitted_fused), 21)

    def test_mount_reuse_guard_match(self):
        """When current gravity matches saved gravity within 5 deg, reuse alignment."""
        saved_rot = R.from_euler("xyz", [10.0, 5.0, 0.0], degrees=True)
        g_saved_unit = saved_rot.as_matrix()[2, :]  # Vertical axis

        saved_alignment = MountAlignment(
            is_calibrated=True,
            R_phone_to_vehicle=saved_rot,
            forward_axis_phone=saved_rot.as_matrix()[0, :],
            lateral_axis_phone=saved_rot.as_matrix()[1, :],
            vertical_axis_phone=g_saved_unit,
            yaw_axis_index=2,
            yaw_axis_sign=1.0,
            mount_yaw_offset_rad=0.0,
            pitch_deg=10.0,
            roll_deg=5.0,
        )

        adapter = EngineAdapterStageA(saved_alignment=saved_alignment)
        self.assertEqual(adapter.get_mount_status(), "Mount: calibrating 0/8")

        # Feed 35 samples aligned with saved gravity (approx 2 deg deviation)
        t_ns = 1_000_000_000
        # Rotate saved gravity by 2.0 degrees
        rot_small = R.from_euler("z", 2.0, degrees=True)
        current_g = rot_small.apply(g_saved_unit) * 9.81

        for i in range(35):
            imu = IMUSample(
                timestamp_ns=t_ns + i * 100_000_000,
                accel=current_g.copy(),
                gyro=np.array([0.0, 0.0, 0.0]),
            )
            adapter.on_imu(imu)

        self.assertTrue(adapter.mount_reused)
        self.assertFalse(adapter.mount_changed)
        self.assertEqual(adapter.get_mount_status(), "Mount: reused")
        self.assertTrue(adapter.stream.calibrator._yaw_locked)

    def test_mount_reuse_guard_mismatch(self):
        """When current gravity differs by >= 5 deg, discard saved alignment."""
        saved_rot = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)
        g_saved_unit = np.array([0.0, 0.0, 1.0])

        saved_alignment = MountAlignment(
            is_calibrated=True,
            R_phone_to_vehicle=saved_rot,
            forward_axis_phone=np.array([1.0, 0.0, 0.0]),
            lateral_axis_phone=np.array([0.0, 1.0, 0.0]),
            vertical_axis_phone=g_saved_unit,
            yaw_axis_index=2,
            yaw_axis_sign=1.0,
            mount_yaw_offset_rad=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )

        adapter = EngineAdapterStageA(saved_alignment=saved_alignment)

        # Feed 35 samples with phone tilted by 30 degrees (exceeds 5.0 deg guard)
        rot_30 = R.from_euler("x", 30.0, degrees=True)
        tilted_g = rot_30.apply(g_saved_unit) * 9.81

        t_ns = 1_000_000_000
        for i in range(35):
            imu = IMUSample(
                timestamp_ns=t_ns + i * 100_000_000,
                accel=tilted_g.copy(),
                gyro=np.array([0.0, 0.0, 0.0]),
            )
            adapter.on_imu(imu)

        self.assertFalse(adapter.mount_reused)
        self.assertTrue(adapter.mount_changed)
        self.assertEqual(adapter.get_mount_status(), "mount changed - drive turns")
        self.assertIsNone(adapter.saved_alignment)

    def test_blackout_firewall(self):
        """GNSS is completely dropped in BLACKOUT state."""
        adapter = EngineAdapterStageA(reference_lat_deg=12.97, reference_lon_deg=77.59)
        adapter.set_blackout(True)

        gnss = GNSSSample(
            timestamp_ns=2_000_000_000,
            latitude_deg=12.971,
            longitude_deg=77.591,
            altitude_m=0.0,
            speed_mps=15.0,
            bearing_deg=45.0,
            is_valid=True,
        )
        adapter.on_gnss(gnss)

        # Calibrator buffer and stream last GNSS sample must remain empty / None
        self.assertIsNone(adapter.stream.last_gnss_sample)
        self.assertEqual(len(adapter.stream.calibrator._gnss_buf), 0)

    def test_gnss_decimation_to_iovnbd_rate(self):
        """When decimate_gnss is True (default), 1 Hz GNSS is decimated to ~9.0s interval for engine."""
        adapter = EngineAdapterStageA(reference_lat_deg=12.97, reference_lon_deg=77.59, decimate_gnss=True, gnss_decimate_interval_s=9.0)

        # Feed 10 fixes at 1 Hz (1 fix every 1.0s, timestamps 0s to 9s)
        t0 = 1_000_000_000
        for i in range(10):
            gnss = GNSSSample(
                timestamp_ns=t0 + i * int(1e9),
                latitude_deg=12.971 + i * 0.0001,
                longitude_deg=77.591 + i * 0.0001,
                altitude_m=0.0,
                speed_mps=15.0,
                bearing_deg=45.0,
                is_valid=True,
            )
            adapter.on_gnss(gnss)

        # In 9.0 seconds, exactly 2 fixes (fix 0 at t=0s, and fix 9 at t=9s) should be ingested by stream
        self.assertEqual(len(adapter.stream.calibrator._gnss_buf), 2)


class TestLiveEvaluator(unittest.TestCase):
    def test_metric_computation(self):
        evaluator = LiveEvaluator(ref_lat=12.9716, ref_lon=77.5946, ref_alt=920.0)

        # Initial GNSS fix
        t0 = 1_000_000_000
        evaluator.on_gnss(GNSSSample(
            timestamp_ns=t0,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            speed_mps=10.0,
            bearing_deg=90.0,  # Heading East
            is_valid=True,
        ))

        # Start blackout
        evaluator.start_blackout(timestamp_ns=t0)

        # Second GNSS fix: traveled 100m East in 10s
        t1 = t0 + int(10.0 * 1e9)
        # 100m East ~ 100 / (111139 * cos(12.97 deg)) = 0.0009228 deg lon
        d_lon = 100.0 / (111139.0 * math.cos(math.radians(12.9716)))
        evaluator.on_gnss(GNSSSample(
            timestamp_ns=t1,
            latitude_deg=12.9716,
            longitude_deg=77.5946 + d_lon,
            altitude_m=920.0,
            speed_mps=10.0,
            bearing_deg=90.0,
            is_valid=True,
        ))

        # Dead reckoning estimated position: traveled 95m East and 5m North
        from sih.data.geo import enu_to_geodetic
        dr_lat, dr_lon, _ = enu_to_geodetic(95.0, 5.0, 0.0, 12.9716, 77.5946, 920.0)
        from sih.core.contracts import FusedPosition
        dr_fused = FusedPosition(
            timestamp_ns=t1,
            latitude_deg=dr_lat,
            longitude_deg=dr_lon,
            altitude_m=920.0,
            position_enu_m=np.array([95.0, 5.0, 0.0]),
            velocity_enu_mps=np.array([9.5, 0.5, 0.0]),
            heading_rad=math.radians(90.0),
            covariance=np.eye(15),
            mode="INS_ONLY_BLACKOUT",
        )

        metrics = evaluator.on_dr(dr_fused)
        self.assertAlmostEqual(metrics.elapsed_s, 10.0, places=1)
        self.assertAlmostEqual(metrics.gnss_dist_m, 100.0, delta=1.5)
        self.assertAlmostEqual(metrics.horizontal_error_m, math.sqrt(5**2 + 5**2), delta=1.5)
        # Drift % should be ~ 7.07% (< 10%)
        self.assertEqual(metrics.drift_label, f"Drift: {metrics.drift_pct:.2f}% (target <10%)")
        self.assertTrue(metrics.target_met)
        self.assertEqual(metrics.speed_regime, "City (20-50 km/h)")


class TestReplayAndRouter(unittest.TestCase):
    def test_in_memory_replay(self):
        loader = GenericDataLoader()
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-M.csv")
        trip = loader.load_file(trip_path)

        # Slice 60 seconds (600 IMU samples)
        t_start = trip.imu_samples[100].timestamp_ns
        t_end = t_start + int(60.0 * 1e9)

        trip_slice = trip.__class__(
            trip_id="test_slice",
            imu_samples=[im for im in trip.imu_samples if t_start <= im.timestamp_ns <= t_end],
            gnss_samples=[g for g in trip.gnss_samples if t_start <= g.timestamp_ns <= t_end],
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=trip.reference_alt_m,
            total_gnss_distance_m=100.0,
            duration_s=60.0,
        )

        router = NavigationRouter(
            ref_lat=trip.reference_lat_deg,
            ref_lon=trip.reference_lon_deg,
            ref_alt=trip.reference_alt_m,
        )

        # Replay with 30s warm-up and 30s blackout
        summary = replay_direct_in_memory(
            trip=trip_slice,
            router=router,
            blackout_start_s=30.0,
            blackout_duration_s=30.0,
        )

        self.assertIn("drift_pct", summary)
        self.assertIn("horizontal_error_m", summary)
        self.assertGreater(len(router.evaluator.dr_fused), 50)

    def test_replay_phone_format_csv(self):
        """Verifies that server.replay accepts phone-logged CSV directly."""
        import tempfile
        from server.replay import load_any_trip

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("type,timestamp_ns,val1,val2,val3,val4,val5,val6\n")
            t0 = 1_700_000_000_000_000_000
            for i in range(50):
                t_i = t0 + i * int(1e8)
                f.write(f"IMU,{t_i},0.1,0.2,9.81,0.01,0.02,0.03\n")
                if i % 10 == 0:
                    f.write(f"GNSS,{t_i},12.9716,77.5946,920.0,12.5,90.0,3.5\n")
            temp_path = f.name

        try:
            trip = load_any_trip(temp_path)
            self.assertEqual(len(trip.imu_samples), 50)
            self.assertEqual(len(trip.gnss_samples), 5)
            self.assertAlmostEqual(trip.reference_lat_deg, 12.9716)
            self.assertAlmostEqual(trip.reference_lon_deg, 77.5946)

            router = NavigationRouter(
                ref_lat=trip.reference_lat_deg,
                ref_lon=trip.reference_lon_deg,
                ref_alt=trip.reference_alt_m,
            )
            summary = replay_direct_in_memory(
                trip=trip,
                router=router,
                blackout_start_s=2.0,
                blackout_duration_s=2.0,
            )
            self.assertIn("drift_pct", summary)
            self.assertIn("speed_regime", summary)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
