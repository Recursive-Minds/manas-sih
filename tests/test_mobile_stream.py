"""
Unit Tests for Mobile Streaming Engine and DeadReckoningEngine.
"""

import unittest
import numpy as np
from sih.mobile.causal_stream import MobileDeadReckoningStream
from sih.engine.dead_reckoning_engine import DeadReckoningEngine
from sih.core.contracts import IMUSample, GNSSSample, FusedPosition


class TestMobileStream(unittest.TestCase):
    def setUp(self):
        self.stream = MobileDeadReckoningStream(
            reference_lat_deg=18.5204,
            reference_lon_deg=73.8567,
            reference_alt_m=560.0,
            enable_map_matching=False,
        )

    def test_gnss_ingestion(self):
        t0_ns = 1_000_000_000
        self.stream.on_gnss_sample(
            lat_deg=18.5204,
            lon_deg=73.8567,
            alt_m=560.0,
            speed_mps=15.0,
            bearing_deg=90.0,
            accuracy_h_m=3.0,
            timestamp_ns=t0_ns,
        )
        self.assertTrue(self.stream.is_gnss_healthy)
        self.assertEqual(self.stream.consecutive_outage_samples, 0)

    def test_imu_streaming_and_outage(self):
        t0_ns = 1_000_000_000
        # Warmup with GNSS
        self.stream.on_gnss_sample(
            lat_deg=18.5204,
            lon_deg=73.8567,
            alt_m=560.0,
            speed_mps=10.0,
            bearing_deg=90.0,
            accuracy_h_m=3.0,
            timestamp_ns=t0_ns,
        )

        # Stream IMU samples (10 Hz = 100ms dt)
        fused = None
        for i in range(25):
            t_ns = t0_ns + (i + 1) * 100_000_000
            # Straight cruising: forward accel 0, gravity on Z (9.81), zero gyro
            fused = self.stream.on_imu_sample(
                ax=0.0,
                ay=0.0,
                az=9.81,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                timestamp_ns=t_ns,
            )

        self.assertIsNotNone(fused)
        self.assertIsInstance(fused, FusedPosition)
        self.assertFalse(np.isnan(fused.position_enu_m).any())
        self.assertFalse(np.isnan(fused.velocity_enu_mps).any())

        # After 25 samples without GNSS update (> 15 samples threshold), dead reckoning flag should be True
        self.assertTrue(fused.is_dead_reckoning)

        telemetry = self.stream.get_current_state()
        self.assertIn("east_m", telemetry)
        self.assertIn("north_m", telemetry)
        self.assertIn("speed_mps", telemetry)
        self.assertTrue(telemetry["is_dead_reckoning"])

    def test_torchscript_auto_load_and_inference(self):
        # Verify model auto-loaded from models/exported/
        self.assertIsNotNone(self.stream.torch_model)
        # Push 65 IMU samples to fill ring buffer (maxlen=60) and trigger neural inference
        t0_ns = 2_000_000_000
        for i in range(65):
            t_ns = t0_ns + (i + 1) * 100_000_000
            fused = self.stream.on_imu_sample(
                ax=0.2,
                ay=0.0,
                az=9.81,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                timestamp_ns=t_ns,
            )
            self.assertIsNotNone(fused)
        self.assertGreaterEqual(len(self.stream.feature_buffer), 60)
        state = self.stream.get_current_state()
        self.assertIsInstance(state["speed_mps"], float)
        self.assertFalse(np.isnan(state["speed_mps"]))


class TestDeadReckoningEngine(unittest.TestCase):
    def test_engine_init(self):
        engine = DeadReckoningEngine(
            turn_threshold_rad_s=np.radians(1.5),
            cooldown_duration_s=0.5,
            smoothing_factor=0.35,
        )
        self.assertEqual(engine.cooldown_duration_s, 0.5)
        self.assertEqual(engine.smoothing_factor, 0.35)


if __name__ == "__main__":
    unittest.main()
