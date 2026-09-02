"""
Unit tests for core contracts, interfaces, and pipeline assembly using unittest.
"""

import unittest
import numpy as np

from sih.core.contracts import (
    IMUSample,
    GNSSSample,
    CalibratedSample,
    VelocityEstimate,
    FusedPosition,
    MatchedPosition,
)
from sih.core.interfaces import IFusionFilter
from sih.core.config import PipelineConfig, FusionFilterConfig
from sih.core.pipeline import (
    assemble_pipeline,
    register_fusion_filter,
)


class MockFusionFilter(IFusionFilter):
    """Simple mock fusion filter for testing assembly and pipeline propagation."""
    def __init__(self, **params) -> None:
        self.state = FusedPosition(
            timestamp_ns=0,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            position_enu_m=np.zeros(3, dtype=np.float64),
            velocity_enu_mps=np.zeros(3, dtype=np.float64),
            heading_rad=0.0,
            covariance=np.eye(6, dtype=np.float64),
            mode="INITIALIZING",
            gnss_outage_duration_s=0.0,
        )

    def reset(self, initial_gnss=None) -> None:
        pass

    def predict(self, sample: CalibratedSample, vel=None) -> FusedPosition:
        self.state = FusedPosition(
            timestamp_ns=sample.timestamp_ns,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            position_enu_m=np.array([1.0, 2.0, 0.0], dtype=np.float64),
            velocity_enu_mps=np.array([0.5, 1.0, 0.0], dtype=np.float64),
            heading_rad=np.pi / 4,
            covariance=np.eye(6, dtype=np.float64),
            mode="INS_ONLY_BLACKOUT",
            gnss_outage_duration_s=1.5,
        )
        return self.state

    def update_gnss(self, gnss: GNSSSample) -> FusedPosition:
        self.state = FusedPosition(
            timestamp_ns=gnss.timestamp_ns,
            latitude_deg=gnss.latitude_deg,
            longitude_deg=gnss.longitude_deg,
            altitude_m=gnss.altitude_m,
            position_enu_m=np.zeros(3, dtype=np.float64),
            velocity_enu_mps=np.zeros(3, dtype=np.float64),
            heading_rad=0.0,
            covariance=np.eye(6, dtype=np.float64) * 0.1,
            mode="GNSS_AIDED",
            gnss_outage_duration_s=0.0,
        )
        return self.state

    def get_state(self) -> FusedPosition:
        return self.state


class TestContractsAndPipeline(unittest.TestCase):

    def test_imu_sample_creation_and_immutability(self):
        accel = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        gyro = np.array([0.01, -0.02, 0.03], dtype=np.float64)
        sample = IMUSample(timestamp_ns=1_000_000_000, accel=accel, gyro=gyro)

        self.assertEqual(sample.timestamp_ns, 1_000_000_000)
        np.testing.assert_array_equal(sample.accel, accel)
        np.testing.assert_array_equal(sample.gyro, gyro)

        # Immutability check
        with self.assertRaises(Exception):
            sample.timestamp_ns = 2_000_000_000  # type: ignore

    def test_imu_sample_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "accel must be a 3D vector"):
            IMUSample(
                timestamp_ns=0,
                accel=np.array([1.0, 2.0]),  # 2D vector should raise ValueError
                gyro=np.array([0.0, 0.0, 0.0]),
            )

        with self.assertRaisesRegex(ValueError, "gyro must be a 3D vector"):
            IMUSample(
                timestamp_ns=0,
                accel=np.array([1.0, 2.0, 3.0]),
                gyro=np.array([0.0, 0.0]),  # 2D vector should raise ValueError
            )

    def test_pipeline_assembly_and_flow(self):
        register_fusion_filter("mock_fusion", lambda **params: MockFusionFilter(**params))

        config = PipelineConfig(
            fusion=FusionFilterConfig(algorithm="mock_fusion")
        )
        pipeline = assemble_pipeline(config)

        # Process IMU tick
        imu = IMUSample(
            timestamp_ns=1_000_000_000,
            accel=np.array([0.1, 0.0, 9.8], dtype=np.float64),
            gyro=np.array([0.0, 0.0, 0.05], dtype=np.float64),
        )

        calib, vel, fused, matched = pipeline.process_imu(imu)

        self.assertIsInstance(calib, CalibratedSample)
        self.assertIsInstance(vel, VelocityEstimate)
        self.assertIsInstance(fused, FusedPosition)
        self.assertIsInstance(matched, MatchedPosition)

        self.assertEqual(fused.timestamp_ns, 1_000_000_000)
        self.assertAlmostEqual(matched.bearing_deg, 45.0, places=2)

        # Process GNSS fix
        gnss = GNSSSample(
            timestamp_ns=1_000_000_000,
            latitude_deg=13.0827,
            longitude_deg=80.2707,
            altitude_m=10.0,
            accuracy_h_m=3.0,
        )
        updated_state = pipeline.process_gnss(gnss)
        self.assertIsNotNone(updated_state)
        self.assertEqual(updated_state.mode, "GNSS_AIDED")
        self.assertEqual(updated_state.latitude_deg, 13.0827)


if __name__ == "__main__":
    unittest.main()
