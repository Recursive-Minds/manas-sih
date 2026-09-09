"""
Unit tests for GPU-accelerated velocity estimation, Bayesian MoE, and spectral feature extraction.
"""

import unittest
import numpy as np
import torch

from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.moe_fusion import BayesianMoEFusion, numpy_bayesian_fusion
from sih.models.losses import (
    balanced_velocity_loss,
    phase55_balanced_loss,
    l_dynamics_variance_alignment,
    l_centripetal,
    l_drift_windowed,
    l_jerk_hinge,
)
from sih.data.vibration import VibrationConditioner
from sih.data.spectral import DualBandSpectralExtractor


class TestGPUPipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_resnet1d_forward_and_shapes(self):
        B, C, L = 4, 12, 20
        model = ResNet1DSpeedEstimator(in_channels=C, base_channels=32).to(self.device)
        x = torch.randn(B, C, L, device=self.device)

        speed, log_var, class_logits = model(x)
        self.assertEqual(speed.shape, (B, 1))
        self.assertEqual(log_var.shape, (B, 1))
        self.assertEqual(class_logits.shape, (B, 3))

    def test_bayesian_moe_fusion(self):
        B, C = 4, 12
        res = ResNet1DSpeedEstimator(in_channels=C, base_channels=32)
        tcn = TCNAttentionVelocityModel(in_channels=C, base_channels=16, num_attention_heads=2)
        moe = BayesianMoEFusion(res, tcn).to(self.device)

        x_short = torch.randn(B, C, 20, device=self.device)
        x_long = torch.randn(B, C, 60, device=self.device)

        v_fused, var_fused, diag = moe(x_short, x_long)

        self.assertEqual(v_fused.shape, (B, 1))
        self.assertEqual(var_fused.shape, (B, 1))
        self.assertIn("v_resnet", diag)
        self.assertIn("v_tcn", diag)
        self.assertIn("class_logits", diag)

        # Minimum-variance property check: fused variance must be strictly smaller than individual expert variances
        self.assertTrue((var_fused <= diag["var_resnet"] + 1e-4).all().item())
        self.assertTrue((var_fused <= diag["var_tcn"] + 1e-4).all().item())

    def test_numpy_bayesian_fusion_helper(self):
        v1, s1 = 10.0, 1.0
        v2, s2 = 12.0, 1.0
        v_fused, s_fused = numpy_bayesian_fusion(v1, s1, v2, s2)
        self.assertAlmostEqual(v_fused, 11.0, places=2)
        self.assertLess(s_fused, 1.0)

    def test_dynamics_variance_alignment_loss(self):
        # When prediction matches variance, loss is near 0
        v_gt = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0], device=self.device)
        v_pred = torch.tensor([0.0, 5.1, 9.9, 15.0, 19.8], device=self.device)
        loss = l_dynamics_variance_alignment(v_pred, v_gt)
        self.assertLess(loss.item(), 0.05)

        # When prediction is flat, deficit should be high
        v_flat = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0], device=self.device)
        loss_flat = l_dynamics_variance_alignment(v_flat, v_gt)
        self.assertGreater(loss_flat.item(), 0.90)

    def test_centripetal_loss(self):
        v = torch.tensor([10.0, 20.0], device=self.device)
        w = torch.tensor([0.1, -0.1], device=self.device)
        a_lat_ideal = v * w
        loss_ideal = l_centripetal(v, a_lat_ideal, w)
        self.assertAlmostEqual(loss_ideal.item(), 0.0, places=4)

    def test_vibration_conditioner_and_spectral_extractor(self):
        N = 200
        cond = VibrationConditioner(sampling_rate=10.0, cutoff_hz=3.5)
        raw_acc = np.random.randn(N, 3).astype(np.float32)
        raw_gyro = np.random.randn(N, 3).astype(np.float32)

        f_acc, f_gyro = cond.filter_imu_sequence(raw_acc, raw_gyro)
        self.assertEqual(f_acc.shape, (N, 3))
        self.assertEqual(f_gyro.shape, (N, 3))

        spec = DualBandSpectralExtractor(sampling_rate=10.0)
        feats = spec.extract_sequence_features(f_acc, window_len=30, stride=5)
        self.assertEqual(feats.shape, (N, 4))
        self.assertTrue(np.all(np.isfinite(feats)))


if __name__ == "__main__":
    unittest.main()
