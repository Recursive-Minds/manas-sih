"""
Unit tests for the 15-State Error-State Kalman Filter (ES-EKF) with NHC.
"""

import unittest
import numpy as np

from sih.core.contracts import IMUSample, GNSSSample, CalibratedSample
from sih.core.config import PipelineConfig, FusionFilterConfig
from sih.core.pipeline import assemble_pipeline
from sih.fusion.es_ekf import ErrorStateEKF


class TestErrorStateEKF(unittest.TestCase):

    def setUp(self):
        self.ekf = ErrorStateEKF(
            accel_noise_std=0.2,
            gyro_noise_std=0.015,
            nhc_lateral_std=0.1,
            nhc_vertical_std=0.1,
            enable_nhc=True,
        )

    def test_initialization_and_gnss_update(self):
        gnss = GNSSSample(
            timestamp_ns=1_000_000_000,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            speed_mps=15.0,
            bearing_deg=90.0,  # Heading East
            accuracy_h_m=2.5,
        )
        state = self.ekf.update_gnss(gnss)

        self.assertEqual(state.mode, "GNSS_AIDED")
        self.assertAlmostEqual(state.latitude_deg, 12.9716, places=5)
        self.assertAlmostEqual(state.longitude_deg, 77.5946, places=5)
        self.assertAlmostEqual(state.velocity_enu_mps[0], 15.0, delta=0.5)  # East velocity
        self.assertAlmostEqual(state.velocity_enu_mps[1], 0.0, delta=0.5)   # North velocity

    def test_covariance_positive_definiteness_during_propagation(self):
        # Initialize
        gnss = GNSSSample(
            timestamp_ns=0,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            speed_mps=10.0,
            bearing_deg=0.0,  # North
        )
        self.ekf.update_gnss(gnss)

        # Propagate 50 IMU ticks (5 seconds at 10Hz)
        for i in range(1, 51):
            ts = int(i * 0.1 * 1e9)
            calib = CalibratedSample(
                timestamp_ns=ts,
                accel_vehicle=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
                gyro_vehicle=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                rotation_body_to_vehicle=np.eye(3),
                gravity_vehicle=np.array([0.0, 0.0, 9.80665]),
                is_calibrated=True,
            )
            state = self.ekf.predict(calib)

            # Verify covariance P is symmetric and positive semi-definite
            P = self.ekf.P
            np.testing.assert_allclose(P, P.T, atol=1e-10)
            eigvals = np.linalg.eigvalsh(P)
            self.assertTrue(np.all(eigvals >= -1e-12), f"Negative eigenvalue detected: {eigvals.min()}")

    def test_pipeline_assembly_with_es_ekf_nhc(self):
        config = PipelineConfig(
            fusion=FusionFilterConfig(algorithm="es_ekf_nhc")
        )
        pipeline = assemble_pipeline(config)

        imu = IMUSample(
            timestamp_ns=1_000_000_000,
            accel=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            gyro=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        calib, vel, fused, matched = pipeline.process_imu(imu)
        self.assertIsNotNone(fused)
        self.assertEqual(fused.position_enu_m.shape, (3,))


if __name__ == "__main__":
    unittest.main()
