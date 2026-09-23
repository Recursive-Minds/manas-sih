"""
Round-1 unit tests (synthetic data only, no dataset needed).
    python -m pytest tests/test_round1.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from round1_synth import make_drive, gnss_at  # noqa: E402

from sih.round1.config import Round1Config, set_active_config  # noqa: E402
from sih.round1.engine_hooks import Round1EngineHooks  # noqa: E402
from sih.round1.stop_detector import StopDetector  # noqa: E402
from sih.round1.gyro_scale import estimate_gyro_scale  # noqa: E402
from sih.round1.online_speed_calib import BandSpeedCalibrator  # noqa: E402
from sih.round1.history import build_pre_blackout_history  # noqa: E402
from sih.round1.junction_anchor import find_corners  # noqa: E402
from sih.engine.dead_reckoning_engine import run_dead_reckoning_scenario, SteppableDeadReckoningEngine  # noqa: E402


def run(drive, cfg: dict):
    set_active_config(Round1Config.from_dict(cfg))
    try:
        return run_dead_reckoning_scenario(drive.trip, drive.calib, drive.v_ai, drive.road,
                                           gnss_at(drive, 185.0), 75.0, domain="Urban")
    finally:
        set_active_config(None)


class TestRound1Flags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drive = make_drive(ai_gain=1.12, ai_creep_at_stop=1.0, gyro_scale_err=1.05)
        cls.base = run(cls.drive, {})

    def test_all_off_means_no_hooks(self):
        self.assertTrue(Round1Config().is_all_off())
        self.assertIsNone(Round1EngineHooks.create("Urban", Round1Config()))
        set_active_config(None)
        eng = SteppableDeadReckoningEngine(12.97, 77.59, road_network=None, domain="Urban")
        self.assertIsNone(eng.r1)
        self.assertFalse(any(k.startswith("r1_") for k in self.base))

    def test_diagnostics_is_bit_identical(self):
        diag = run(self.drive, {"diagnostics": True})
        np.testing.assert_array_equal(diag["map_pts"], self.base["map_pts"])
        np.testing.assert_array_equal(diag["pure_pts"], self.base["pure_pts"])
        self.assertIn("r1_scale_effective", diag)

    def test_t9_removes_double_scale(self):
        diag = run(self.drive, {"diagnostics": True})
        self.assertAlmostEqual(diag["r1_scale_effective"], diag["r1_scale_engine"] * diag["r1_scale_ekf"], places=3)
        fixed = run(self.drive, {"scale_fix": {"source": "engine"}})
        self.assertLess(fixed["map_err_m"], self.base["map_err_m"])

    def test_t3_stop_reduces_error_with_t9(self):
        a = run(self.drive, {"scale_fix": {"source": "engine"}})
        b = run(self.drive, {"scale_fix": {"source": "engine"}, "stop": {"enabled": True}})
        self.assertGreaterEqual(b["r1_stops"], 1)
        self.assertLess(b["map_err_m"], a["map_err_m"])

    def test_t8_anchor_applies(self):
        a = run(self.drive, {"scale_fix": {"source": "engine"}})
        b = run(self.drive, {"scale_fix": {"source": "engine"}, "junction": {"enabled": True}})
        self.assertEqual(b["r1_anchors"], 1)
        self.assertLess(b["map_err_m"], a["map_err_m"])

    def test_every_flag_runs(self):
        for cfg in ({"gyro_scale": {"enabled": True}}, {"online_calib": {"enabled": True}},
                    {"speed_mode": {"mode": "hold_entry"}}, {"speed_mode": {"mode": "entry_offset_decay"}},
                    {"scale_fix": {"source": "ekf"}}):
            r = run(self.drive, cfg)
            self.assertTrue(np.isfinite(r["map_err_m"]), cfg)


class TestLearners(unittest.TestCase):
    def test_gyro_scale_recovers_error(self):
        d = make_drive(gyro_scale_err=1.05)
        h = build_pre_blackout_history(d.trip, d.calib, d.v_ai, d.trip.gnss_samples[185].timestamp_ns, 180.0)
        p = Round1Config().gyro_scale
        p.prior_windows = 0.0
        out = estimate_gyro_scale(h, p)
        self.assertGreaterEqual(out["n_windows"], 3)
        self.assertAlmostEqual(out["scale"], 1 / 1.05, delta=0.012)

    def test_history_is_causal(self):
        d = make_drive()
        t_bo = d.trip.gnss_samples[185].timestamp_ns
        h = build_pre_blackout_history(d.trip, d.calib, d.v_ai, t_bo, 180.0)
        self.assertTrue(np.all(h.imu_ts_ns < t_bo) and np.all(h.gnss_ts_ns < t_bo))

    def test_band_calibrator_learns_shape(self):
        prof = [(60, 3, 3, 0), (60, 14, 14, 0), (30, 3, 3, 0), (60, 14, 14, 0)]
        d = make_drive(profile=prof)
        v_ai = d.v_ai.copy()
        v_ai[d.v_true < 8] *= 1.25                         # AI over-reads at low speed only
        h = build_pre_blackout_history(d.trip, d.calib, v_ai, d.trip.gnss_samples[-1].timestamp_ns, 400.0)
        p = Round1Config().online_calib
        p.min_window_dist_m = 10.0
        cal = BandSpeedCalibrator(p)
        cal.fit(h)
        self.assertTrue(cal.fitted)
        self.assertLess(cal.factor(3.5), cal.factor(14.0))

    def test_stop_detector_enters_and_launches(self):
        p = Round1Config().stop
        p.enabled = True
        sd = StopDetector(p)
        sd.reset(v_entry=8.0)
        rng = np.random.default_rng(0)
        g = np.array([0.0, 0.0, 9.81])
        for _ in range(30):                                 # idle, AI creeps at 0.9 m/s
            sd.update(0.9, g + rng.normal(0, 0.1, 3), rng.normal(0, 0.003, 3), 0.1)
        self.assertTrue(sd.stopped)
        for _ in range(10):                                 # launch: +1.2 m/s^2 forward
            sd.update(1.0, g + np.array([1.2, 0, 0]) + rng.normal(0, 0.1, 3), rng.normal(0, 0.003, 3), 0.1)
        self.assertFalse(sd.stopped)

    def test_corner_finder_and_ambiguity(self):
        class S:
            def __init__(self, a, b):
                self.start_enu_m, self.end_enu_m = np.array(a, float), np.array(b, float)
                d = self.end_enu_m - self.start_enu_m
                self.bearing_deg = float(np.degrees(np.arctan2(d[0], d[1])) % 360)
        segs = [S((0, -100), (0, 0)), S((0, 0), (100, 0))]           # north then east, corner at origin
        c = find_corners(segs, 0.0, 90.0, np.array([5.0, -5.0]), 60.0, 25.0)
        self.assertEqual(len(c), 1)
        np.testing.assert_allclose(c[0][1], [0, 0], atol=1e-6)
        segs += [S((40, -100), (40, 0)), S((40, 0), (140, 0.5))]      # a second, parallel junction 40 m away
        c2 = find_corners(segs, 0.0, 90.0, np.array([20.0, -5.0]), 60.0, 25.0)
        self.assertEqual(len(c2), 2)


class TestIntervalLoss(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_alpha_cancels_constant_bias(self):
        import torch
        from sih.models.interval_loss import emulate_alpha, interval_distance_loss
        v = 8 + 2 * torch.sin(torch.arange(950) / 40.0).unsqueeze(0).repeat(3, 1)
        v_hat = 0.9 * v
        a = emulate_alpha(v_hat[:, :200], v[:, :200])
        self.assertTrue(torch.allclose(a, torch.full((3,), 1 / 0.9), atol=1e-4))
        self.assertLess(float(interval_distance_loss(v_hat[:, 200:], v[:, 200:], a)), 1e-4)
        self.assertGreater(float(interval_distance_loss(v_hat[:, 200:], v[:, 200:], torch.ones(3))), 0.05)

    def test_sampler_refuses_unseen_trips(self):
        from sih.models.interval_dataset import IntervalSequenceSampler
        arr = {"feats": np.zeros((100, 12), np.float32), "v": np.zeros(100, np.float32),
               "a_lat": np.zeros(100, np.float32), "w_yaw": np.zeros(100, np.float32)}
        for bad in ("S-S3a", "S-S4"):
            with self.assertRaises(ValueError):
                IntervalSequenceSampler({bad: arr}, np.zeros(12), np.ones(12))


if __name__ == "__main__":
    unittest.main()
