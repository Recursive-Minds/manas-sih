"""
Unit tests for the TCN-Attention Velocity Model and Dataset.
"""

import unittest
import torch
import numpy as np

from sih.models.tcn_attention import TCNAttentionVelocityModel, gaussian_nll_loss
from sih.velocity.ai_estimator import AIVelocityEstimator
from sih.core.contracts import CalibratedSample


class TestVelocityModel(unittest.TestCase):

    def test_model_forward_shapes(self):
        batch_size = 4
        in_channels = 6
        window_size = 100

        model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
        x = torch.randn(batch_size, in_channels, window_size)

        speed, log_var = model(x)

        self.assertEqual(speed.shape, (batch_size, 1))
        self.assertEqual(log_var.shape, (batch_size, 1))
        # Speed must be non-negative (Softplus output)
        self.assertTrue((speed >= 0).all().item())

    def test_gaussian_nll_loss_computation(self):
        y_true = torch.tensor([[10.0], [15.0]])
        y_pred = torch.tensor([[9.5], [15.2]])
        log_var = torch.tensor([[0.5], [0.2]])

        loss = gaussian_nll_loss(y_true, y_pred, log_var)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_online_ai_velocity_estimator_buffer(self):
        estimator = AIVelocityEstimator(window_size=10, checkpoint_path="non_existent.pt")

        # Ingest 15 samples
        for i in range(15):
            sample = CalibratedSample(
                timestamp_ns=int(i * 0.1 * 1e9),
                accel_vehicle=np.array([0.1, 0.0, 9.8], dtype=np.float64),
                gyro_vehicle=np.array([0.01, 0.0, 0.0], dtype=np.float64),
                rotation_body_to_vehicle=np.eye(3),
                gravity_vehicle=np.array([0.0, 0.0, 9.8]),
                is_calibrated=True,
            )
            vel = estimator.estimate(sample)
            if i < 9:
                self.assertEqual(vel.motion_state, "INITIALIZING")
            else:
                self.assertIn(vel.motion_state, ["STATIONARY", "DRIVING_UNTRAINED", "DRIVING"])


if __name__ == "__main__":
    unittest.main()
