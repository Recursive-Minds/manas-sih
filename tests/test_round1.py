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
        set_active_config(Round1Config())
        try:
            eng = SteppableDeadReckoningEngine(12.97, 77.59, road_network=None, domain="Urban")
        finally:
            set_active_config(None)
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

    def test_t10_default_bounds_are_bit_identical(self):
        same = run(self.drive, {"scale_level": {"enabled": True}})
        np.testing.assert_array_equal(same["map_pts"], self.base["map_pts"])
        self.assertTrue(np.isfinite(same["r1_scale_raw"]))

    def test_t10_wider_clip_fixes_saturated_scale(self):
        # needs scale ~0.74, baseline clips at 0.85. scale_fix=engine mimics real data, where the
        # autopsy showed the EKF's own speed scale stays at ~1.0 (the synthetic EKF learns it).
        d = make_drive(ai_gain=1.35)
        base = run(d, {"scale_fix": {"source": "engine"}})
        wide = run(d, {"scale_fix": {"source": "engine"}, "scale_level": {"enabled": True, "lo": 0.6, "hi": 1.6}})
        self.assertAlmostEqual(base["r1_scale_engine"], 0.85, places=3)
        self.assertLess(wide["map_err_m"], base["map_err_m"])
        hist = run(d, {"scale_level": {"enabled": True, "lo": 0.6, "hi": 1.6, "source": "history"}})
        self.assertAlmostEqual(hist["r1_scale_history"], 1 / 1.35, delta=0.03)

    def test_r2_level_window(self):
        from sih.round1.history import slice_history_tail
        d = make_drive(ai_gain=1.35)
        h = build_pre_blackout_history(d.trip, d.calib, d.v_ai, d.trip.gnss_samples[185].timestamp_ns, 180.0)
        t = slice_history_tail(h, 60.0)
        self.assertLessEqual((t.imu_ts_ns[-1] - t.imu_ts_ns[0]) * 1e-9, 60.0 + 1e-6)
        self.assertEqual(t.imu_ts_ns[-1], h.imu_ts_ns[-1])
        r = run(d, {"scale_fix": {"source": "engine"},
                    "scale_level": {"enabled": True, "source": "history", "window_s": 60.0, "lo": 0.6, "hi": 1.6}})
        self.assertAlmostEqual(r["r1_scale_history"], 1 / 1.35, delta=0.04)

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


class TestPromotionSupport(unittest.TestCase):
    def test_profile_precedence(self):
        import json
        import tempfile
        import sih.round1.config as C
        old_prof, old_env = C.PRODUCTION_PROFILE, os.environ.pop("SIH_ROUND1_CONFIG", None)
        try:
            with tempfile.TemporaryDirectory() as d:
                prof = os.path.join(d, "production.json")
                C.PRODUCTION_PROFILE = prof
                set_active_config(None)
                self.assertTrue(C.get_active_config().is_all_off())            # no profile -> old behaviour
                json.dump({"name": "production", "junction": {"enabled": True}}, open(prof, "w"))
                self.assertEqual(C.get_active_config().name, "production")    # profile picked up
                os.environ["SIH_ROUND1_CONFIG"] = "off"
                self.assertTrue(C.get_active_config().is_all_off())            # env kill-switch wins
                set_active_config(Round1Config(name="explicit"))
                self.assertEqual(C.get_active_config().name, "explicit")      # explicit wins over all
        finally:
            set_active_config(None)
            C.PRODUCTION_PROFILE = old_prof
            os.environ.pop("SIH_ROUND1_CONFIG", None)
            if old_env is not None:
                os.environ["SIH_ROUND1_CONFIG"] = old_env

    def test_missing_configured_checkpoint_raises(self):
        from sih.round1.model_select import resolve_velocity_checkpoint
        set_active_config(Round1Config(velocity_checkpoint="models/checkpoints/does_not_exist.pt"))
        try:
            with self.assertRaises(FileNotFoundError):
                resolve_velocity_checkpoint(os.getcwd())
        finally:
            set_active_config(None)

    def test_live_history_matches_benchmark_history(self):
        from sih.round1.history import build_history_from_buffers
        d = make_drive()
        t_bo = d.trip.gnss_samples[185].timestamp_ns
        a = build_pre_blackout_history(d.trip, d.calib, d.v_ai, t_bo, 180.0)
        b = build_history_from_buffers(list(d.calib), list(d.v_ai), [g for g in d.trip.gnss_samples if g.is_valid],
                                       d.trip.reference_lat_deg, d.trip.reference_lon_deg, t_bo, 180.0)
        for f in ("imu_ts_ns", "gyro_z", "v_ai_raw", "gnss_ts_ns", "gnss_en", "gnss_speed"):
            np.testing.assert_allclose(getattr(a, f), getattr(b, f), err_msg=f)

    def test_mean_ensemble(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from sih.models.resnet1d import ResNet1DSpeedEstimator
        from sih.models.tcn_attention import TCNAttentionVelocityModel
        from sih.models.moe_fusion import BayesianMoEFusion
        from sih.round1.model_select import MeanMoEEnsemble
        torch.manual_seed(0)
        ms = [BayesianMoEFusion(ResNet1DSpeedEstimator(12, 64), TCNAttentionVelocityModel(12, 32, 4)).eval() for _ in range(2)]
        ens = MeanMoEEnsemble(ms).eval()
        xs, xl = torch.randn(3, 12, 20), torch.randn(3, 12, 60)
        with torch.no_grad():
            v, var, diag = ens(xs, xl)
            v0, v1 = ms[0](xs, xl)[0], ms[1](xs, xl)[0]
        self.assertTrue(torch.allclose(v, (v0 + v1) / 2, atol=1e-6))
        self.assertEqual(tuple(diag["v_members"].shape[:1]), (2,))


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
