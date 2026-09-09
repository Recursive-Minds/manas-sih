"""Unit tests for Phase 4.5 modules: GNSSHandoffManager, RLSAffineCalibrator, and 5-Fold Splitting."""

import unittest
import numpy as np

from sih.fusion.handoff import GNSSHandoffManager, HandoffOutput
from sih.calibration.online_calibrator import RLSAffineCalibrator
from train_5fold_cross_validation import build_purged_5fold_splits


class TestPhase45Modules(unittest.TestCase):

    def test_handoff_manager_blackout_and_recovery(self):
        hm = GNSSHandoffManager(gnss_timeout_s=1.0, blend_duration_s=1.5)

        # 1. Full GNSS driving
        out1 = hm.update(
            timestamp=10.0,
            ins_pos=(12.9716, 77.5946),
            ins_heading_deg=90.0,
            gnss_data={"lat": 12.9716, "lon": 77.5946, "accuracy": 2.0, "bearing_deg": 90.0},
        )
        self.assertEqual(out1.mode, "FULL_GNSS")
        self.assertAlmostEqual(out1.blend_weight, 1.0)

        # 2. Blackout entry (tunnel)
        out2 = hm.update(
            timestamp=12.0,  # 2 seconds later, no GNSS
            ins_pos=(12.9720, 77.5950),
            ins_heading_deg=92.0,
            gnss_data=None,
        )
        self.assertEqual(out2.mode, "DEAD_RECKONING")
        self.assertAlmostEqual(out2.blend_weight, 0.0)
        self.assertAlmostEqual(out2.lat, 12.9720)

        # 3. GNSS recovery with position jump (emerging from tunnel)
        # GNSS fix recovers at slightly different coordinates
        out3 = hm.update(
            timestamp=20.0,
            ins_pos=(12.9730, 77.5960),
            ins_heading_deg=95.0,
            gnss_data={"lat": 12.9732, "lon": 77.5962, "accuracy": 3.0, "bearing_deg": 96.0},
        )
        self.assertEqual(out3.mode, "RECOVERY_BLENDING")
        self.assertGreaterEqual(out3.teleport_mitigated_m, 0.0)

        # 4. Completion of blending after blend_duration_s
        out4 = hm.update(
            timestamp=22.0,  # 2.0s later > 1.5s blend duration
            ins_pos=(12.9740, 77.5970),
            ins_heading_deg=96.0,
            gnss_data={"lat": 12.9741, "lon": 77.5971, "accuracy": 2.5, "bearing_deg": 96.0},
        )
        self.assertEqual(out4.mode, "FULL_GNSS")
        self.assertAlmostEqual(out4.blend_weight, 1.0)

    def test_rls_affine_calibrator_convergence(self):
        calibrator = RLSAffineCalibrator(forgetting_factor=0.98, min_updates_for_calibration=5)

        # True relation: v_gnss = 1.10 * v_ai + 0.5
        true_alpha = 1.10
        true_beta = 0.50

        np.random.seed(42)
        for _ in range(50):
            v_ai = np.random.uniform(3.0, 20.0)
            v_gnss = true_alpha * v_ai + true_beta + np.random.normal(0, 0.05)
            calibrator.update(v_gnss, v_ai)

        calibrator.freeze()
        # Calibrated output must closely match true relation
        test_v = 10.0
        expected = true_alpha * test_v + true_beta
        actual = calibrator.get_calibrated(test_v)
        self.assertAlmostEqual(actual, expected, delta=0.5)

    def test_5fold_purged_embargo_splits_integrity(self):
        # Create synthetic trip features: 1000 samples each
        N = 1000
        feats = np.random.randn(N, 12).astype(np.float32)
        speeds = np.linspace(0.0, 25.0, N).astype(np.float32)
        f_acc = np.random.randn(N, 3).astype(np.float32)
        f_gyr = np.random.randn(N, 3).astype(np.float32)
        trip_data = [(feats, speeds, f_acc, f_gyr)]

        embargo = 50
        folds = build_purged_5fold_splits(
            trip_data,
            short_len=10,
            long_len=20,
            stride=5,
            embargo_samples=embargo,
        )

        self.assertEqual(len(folds), 5)
        for f in folds:
            self.assertGreater(len(f["train_ds"]), 0)
            self.assertGreater(len(f["test_ds"]), 0)
            self.assertEqual(f["mean"].shape, (1, 12))
            self.assertEqual(f["std"].shape, (1, 12))
            self.assertTrue(np.all(f["std"] > 0))


if __name__ == "__main__":
    unittest.main()
