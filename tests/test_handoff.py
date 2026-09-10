"""
tests/test_handoff.py
---------------------
Unit and integration tests for Phase 6: Seamless GNSS <-> INS Handoff State Machine.
"""

import unittest
import numpy as np

from sih.core.contracts import (
    IMUSample,
    GNSSSample,
    CalibratedSample,
    VelocityEstimate,
    FusedPosition,
)
from sih.core.config import PipelineConfig, HandoffStageConfig
from sih.core.pipeline import assemble_pipeline, SeamlessGNSSHandoffManager
from sih.handoff.integrity import (
    compute_position_nis,
    check_kinematic_feasibility,
    evaluate_signal_quality,
)
from sih.handoff.reconciliation import HermiteReconciler
from sih.handoff.manager import HandoffState, HandoffConfig


class TestHandoffIntegrity(unittest.TestCase):
    """Tests statistical innovation and kinematic feasibility functions."""

    def test_compute_position_nis(self):
        # 1-sigma error on unit covariance should give NIS = 1.0
        y_p = np.array([1.0, 0.0, 0.0])
        S_p = np.eye(3)
        nis = compute_position_nis(y_p, S_p, dims=2)
        self.assertAlmostEqual(nis, 1.0, places=5)

        # 2D diagonal covariance
        y_p = np.array([2.0, 3.0, 0.0])
        S_p = np.diag([4.0, 9.0, 1.0])
        # (2^2 / 4) + (3^2 / 9) = 1.0 + 1.0 = 2.0
        nis = compute_position_nis(y_p, S_p, dims=2)
        self.assertAlmostEqual(nis, 2.0, places=5)

    def test_check_kinematic_feasibility(self):
        p1 = np.array([0.0, 0.0, 0.0])
        # In 1.0 second, driving at 20 m/s moves 20 meters (plausible)
        p2_plausible = np.array([20.0, 0.0, 0.0])
        self.assertTrue(check_kinematic_feasibility(p2_plausible, p1, dt_s=1.0, v_max_mps=35.0))

        # In 0.1 second, moving 50 meters implies 500 m/s (impossible multipath spike)
        p3_impossible = np.array([50.0, 0.0, 0.0])
        self.assertFalse(check_kinematic_feasibility(p3_impossible, p1, dt_s=0.1, v_max_mps=35.0))

    def test_evaluate_signal_quality(self):
        good_fix = GNSSSample(
            timestamp_ns=1_000_000_000,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            accuracy_h_m=4.5,
            is_valid=True,
        )
        self.assertTrue(evaluate_signal_quality(good_fix, max_accuracy_h_m=25.0))

        bad_acc_fix = GNSSSample(
            timestamp_ns=1_000_000_000,
            latitude_deg=12.9716,
            longitude_deg=77.5946,
            altitude_m=920.0,
            accuracy_h_m=50.0,
            is_valid=True,
        )
        self.assertFalse(evaluate_signal_quality(bad_acc_fix, max_accuracy_h_m=25.0))

        nan_fix = GNSSSample(
            timestamp_ns=1_000_000_000,
            latitude_deg=float("nan"),
            longitude_deg=77.5946,
            altitude_m=920.0,
            accuracy_h_m=5.0,
            is_valid=True,
        )
        self.assertFalse(evaluate_signal_quality(nan_fix))


class TestHermiteReconciler(unittest.TestCase):
    """Tests C^2 cubic Hermite smoothstep zero-jump reconciliation properties."""

    def test_boundary_conditions(self):
        reconciler = HermiteReconciler(blend_duration_s=1.0)
        t_start = 10_000_000_000 # 10s in ns
        p_dr = np.array([100.0, 50.0, 0.0])
        p_fused = np.array([120.0, 55.0, 0.0]) # 20.6m discrepancy

        reconciler.initiate_blend(t_start, p_dr, p_fused)
        self.assertTrue(reconciler.is_blending)

        # 1. At tau = 0 (t = 10s): alpha = 0.0 -> display must equal p_dr exactly!
        p_disp_0, alpha_0 = reconciler.get_blended_position(t_start, p_fused)
        self.assertAlmostEqual(alpha_0, 0.0, places=5)
        np.testing.assert_allclose(p_disp_0, p_dr, atol=1e-5)

        # 2. Midway at tau = 0.5 (t = 10.5s): alpha = 3(0.25) - 2(0.125) = 0.5
        t_mid = t_start + 500_000_000
        p_disp_mid, alpha_mid = reconciler.get_blended_position(t_mid, p_fused)
        self.assertAlmostEqual(alpha_mid, 0.5, places=5)
        expected_mid = 0.5 * p_dr + 0.5 * p_fused
        np.testing.assert_allclose(p_disp_mid, expected_mid, atol=1e-5)

        # 3. At tau = 1.0 (t = 11.0s): alpha = 1.0 -> display must equal p_fused exactly!
        t_end = t_start + 1_000_000_000
        p_disp_end, alpha_end = reconciler.get_blended_position(t_end, p_fused)
        self.assertAlmostEqual(alpha_end, 1.0, places=5)
        np.testing.assert_allclose(p_disp_end, p_fused, atol=1e-5)
        # Blending should shut off
        self.assertFalse(reconciler.is_blending)


class TestSeamlessGNSSHandoffManager(unittest.TestCase):
    """Tests the 6-state finite state machine transitions and reacquisition logic."""

    def setUp(self):
        self.config = HandoffConfig(
            healthy_accuracy_max_m=15.0,
            degraded_accuracy_max_m=35.0,
            blackout_timeout_s=1.5,
            reacquisition_fixes_required=3,
            blend_duration_s=1.0,
            v_max_kinematic_mps=35.0,
        )
        self.manager = SeamlessGNSSHandoffManager(config=self.config)
        self.manager.set_reference_origin(12.9716, 77.5946, 920.0)

    def test_state_transitions(self):
        # 1. First valid fix transitions INITIALIZING -> GNSS_HEALTHY
        t0 = 1_000_000_000
        g0 = GNSSSample(timestamp_ns=t0, latitude_deg=12.9716, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=5.0, is_valid=True)
        accepted = self.manager.evaluate_gnss(g0)
        self.assertTrue(accepted)
        self.assertEqual(self.manager.state, HandoffState.GNSS_HEALTHY)
        self.assertFalse(self.manager.should_freeze_adaptation())

        # 2. Degraded accuracy fix transitions GNSS_HEALTHY -> GNSS_DEGRADED
        t1 = t0 + 1_000_000_000
        g1 = GNSSSample(timestamp_ns=t1, latitude_deg=12.9717, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=20.0, is_valid=True)
        accepted = self.manager.evaluate_gnss(g1)
        self.assertTrue(accepted)
        self.assertEqual(self.manager.state, HandoffState.GNSS_DEGRADED)
        # Critical rule: parameter adaptation MUST freeze during degradation
        self.assertTrue(self.manager.should_freeze_adaptation())

        # 3. Time passes with zero fixes (tunnel entry) -> timeout triggers INS_DEAD_RECKONING
        t2 = t1 + 2_000_000_000 # 2.0s elapsed > 1.5s timeout
        p_dr = np.array([50.0, 20.0, 0.0])
        self.manager.notify_imu_step(timestamp_ns=t2, p_current_enu=p_dr, dt_s=0.1)
        self.assertEqual(self.manager.state, HandoffState.INS_DEAD_RECKONING)

        # 4. First reacquisition fix arrives upon exiting tunnel
        # Must transition to REACQUISITION_VERIFY and return False (do not jump immediately!)
        t3 = t2 + 1_000_000_000
        g_reacq1 = GNSSSample(timestamp_ns=t3, latitude_deg=12.9718, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=6.0, is_valid=True)
        acc1 = self.manager.evaluate_gnss(g_reacq1)
        self.assertFalse(acc1)
        self.assertEqual(self.manager.state, HandoffState.REACQUISITION_VERIFY)

        # 5. Second reacquisition fix arrives (still verifying)
        t4 = t3 + 1_000_000_000
        g_reacq2 = GNSSSample(timestamp_ns=t4, latitude_deg=12.9719, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=5.0, is_valid=True)
        acc2 = self.manager.evaluate_gnss(g_reacq2)
        self.assertFalse(acc2)
        self.assertEqual(self.manager.state, HandoffState.REACQUISITION_VERIFY)

        # 6. Third consistent fix arrives -> N=3 requirement met!
        # Must verify, initiate blend, and transition to REACQUISITION_BLENDING
        t5 = t4 + 1_000_000_000
        g_reacq3 = GNSSSample(timestamp_ns=t5, latitude_deg=12.9720, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=4.0, is_valid=True)
        acc3 = self.manager.evaluate_gnss(g_reacq3)
        self.assertTrue(acc3)
        self.assertEqual(self.manager.state, HandoffState.REACQUISITION_BLENDING)
        self.assertTrue(self.manager.reconciler.is_blending)

    def test_multipath_quarantine_at_tunnel_exit(self):
        # Establish DR state
        t0 = 1_000_000_000
        g0 = GNSSSample(timestamp_ns=t0, latitude_deg=12.9716, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=5.0, is_valid=True)
        self.manager.evaluate_gnss(g0)

        # Simulate tunnel outage
        t_outage = t0 + 5_000_000_000
        self.manager.notify_imu_step(timestamp_ns=t_outage, p_current_enu=np.array([50.0, 0.0, 0.0]), dt_s=0.1)
        self.assertEqual(self.manager.state, HandoffState.INS_DEAD_RECKONING)

        # 1st fix at exit: buffered
        t1 = t_outage + 1_000_000_000
        g1 = GNSSSample(timestamp_ns=t1, latitude_deg=12.9720, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=6.0, is_valid=True)
        self.manager.evaluate_gnss(g1)
        self.assertEqual(self.manager.state, HandoffState.REACQUISITION_VERIFY)

        # 2nd fix is an impossible multipath spike (+200m away in 0.5s)
        t2 = t1 + 500_000_000
        g_spike = GNSSSample(timestamp_ns=t2, latitude_deg=12.9740, longitude_deg=77.5946, altitude_m=920.0, accuracy_h_m=6.0, is_valid=True)
        acc_spike = self.manager.evaluate_gnss(g_spike)
        self.assertFalse(acc_spike)
        # Verification failed! Must reset buffer and stay in dead reckoning
        self.assertEqual(self.manager.state, HandoffState.INS_DEAD_RECKONING)


class TestPipelineHandoffIntegration(unittest.TestCase):
    """Tests integration of seamless handoff into the central IDRPipeline."""

    def test_pipeline_assembly(self):
        cfg = PipelineConfig()
        cfg.handoff = HandoffStageConfig(algorithm="seamless_handoff", params={"blend_duration_s": 1.0})

        pipeline = assemble_pipeline(cfg)
        self.assertIsInstance(pipeline.handoff_policy, SeamlessGNSSHandoffManager)

        # Run sample step
        imu = IMUSample(
            timestamp_ns=1_000_000_000,
            accel=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.0]),
        )
        cal, vel, fused, matched = pipeline.process_imu(imu)
        self.assertIsNotNone(fused)
        self.assertIsNotNone(matched)


if __name__ == "__main__":
    unittest.main()
