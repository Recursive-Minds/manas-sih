"""
tests/test_ondevice_drift_parity.py
-----------------------------------
Verifies that the on-device real-time evaluator (LiveEvaluator in server/evaluator.py)
computes drift percentage from recorded GNSS ground truth EXACTLY identical to the
laptop benchmark engine (sih/engine/dead_reckoning_engine.py) within 0.01 percentage points.
"""

import os
import sys
import unittest
import numpy as np
import pickle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.geo import geodetic_to_enu
from sih.core.contracts import FusedPosition, GNSSSample
from server.evaluator import LiveEvaluator


class TestOnDeviceDriftParity(unittest.TestCase):
    def test_ondevice_evaluator_drift_matches_laptop_evaluator_within_0_01pp(self):
        bundle_path = os.path.join(ROOT_DIR, "data", "exported", "s_s3a_parity_bundle.pkl")
        if not os.path.exists(bundle_path):
            self.skipTest(f"Bundle {bundle_path} not found")

        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        trip = bundle["trip"]
        scenarios = bundle["scenarios"]
        ref_lat = trip.reference_lat_deg
        ref_lon = trip.reference_lon_deg
        ref_alt = 0.0

        for sc in scenarios:
            tgt = sc["target"]
            sc_id = tgt["id"]
            dur_s = tgt["dur"]
            entry_g = sc["entry_gnss"]
            bo_start_ns = entry_g.timestamp_ns
            bo_end_ns = bo_start_ns + int(dur_s * 1e9)

            batch_pts = sc["batch_pts"]
            batch_ts = sc["batch_ts"]
            bo_gnss = [x for x in trip.gnss_samples if bo_start_ns <= x.timestamp_ns <= bo_end_ns and x.is_valid]

            # 1. Laptop evaluator formula (as in dead_reckoning_engine.py lines 640-654)
            gt_pts = np.array([
                geodetic_to_enu(x.latitude_deg, x.longitude_deg, 0.0, ref_lat, ref_lon, 0.0)[:2]
                for x in bo_gnss
            ])
            gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1)))
            gt_end_enu = gt_pts[-1]
            eval_t_ns = float(bo_gnss[-1].timestamp_ns)

            b_east = float(np.interp(eval_t_ns, batch_ts, batch_pts[:, 0]))
            b_north = float(np.interp(eval_t_ns, batch_ts, batch_pts[:, 1]))
            eval_pt = np.array([b_east, b_north])
            final_err_map = float(np.linalg.norm(eval_pt - gt_end_enu))
            laptop_drift_pct = (final_err_map / gt_dist) * 100.0

            # 2. On-device LiveEvaluator execution
            live_eval = LiveEvaluator(ref_lat=ref_lat, ref_lon=ref_lon, ref_alt=ref_alt)

            # Pre-feed warmup GNSS to establish baseline
            for g in trip.gnss_samples:
                if g.timestamp_ns < bo_start_ns and g.is_valid:
                    live_eval.on_gnss(g)

            # Start blackout
            live_eval.start_blackout(timestamp_ns=bo_start_ns)

            # Feed fused positions and blackout GNSS
            g_idx = 0
            n_g = len(bo_gnss)
            for pt_idx in range(len(batch_pts)):
                t_curr = int(batch_ts[pt_idx])
                pos_enu = np.array([batch_pts[pt_idx, 0], batch_pts[pt_idx, 1], 0.0])

                while g_idx < n_g and bo_gnss[g_idx].timestamp_ns <= t_curr:
                    live_eval.on_gnss(bo_gnss[g_idx])
                    g_idx += 1

                fused = FusedPosition(
                    timestamp_ns=t_curr,
                    latitude_deg=0.0,
                    longitude_deg=0.0,
                    altitude_m=0.0,
                    position_enu_m=pos_enu,
                    velocity_enu_mps=np.zeros(3),
                    heading_rad=0.0,
                    covariance=np.eye(3),
                    mode="INS_ONLY_BLACKOUT",
                )
                live_eval.on_dr(fused)

            # Feed any remaining blackout GNSS fixes
            while g_idx < n_g:
                live_eval.on_gnss(bo_gnss[g_idx])
                g_idx += 1

            live_eval.stop_blackout(timestamp_ns=bo_end_ns)
            summary = live_eval.last_completed_summary
            self.assertIsNotNone(summary, f"Scenario #{sc_id} produced no summary!")
            ondevice_drift_pct = summary.get("drift_pct_raw", summary["drift_pct"])
            ondevice_final_err = summary.get("final_error_m_raw", summary["final_error_m"])
            ondevice_gt_dist = summary.get("gnss_dist_m_raw", summary["gnss_dist_m"])

            diff_pp = abs(ondevice_drift_pct - laptop_drift_pct)
            print(f"Scenario #{sc_id}: Laptop: err={final_err_map:.4f}m, dist={gt_dist:.2f}m, drift={laptop_drift_pct:.4f}% | On-Device: err={ondevice_final_err:.4f}m, dist={ondevice_gt_dist:.2f}m, drift={ondevice_drift_pct:.4f}%, diff={diff_pp:.4f}pp")
            self.assertLessEqual(diff_pp, 0.01, f"Scenario #{sc_id} drift discrepancy {diff_pp:.4f} pp exceeds 0.01 pp target!")
            self.assertAlmostEqual(ondevice_gt_dist, gt_dist, delta=0.01)
            self.assertAlmostEqual(ondevice_final_err, final_err_map, delta=0.01)


if __name__ == "__main__":
    unittest.main()
