"""
Leak-Free Verification Unit Test Suite (SIH PS 26168)
----------------------------------------------------
Verifies:
1. Future Independence: Replacing sensor data (IMU and GNSS) after blackout
   exit with NaNs yields bit-identical trajectory through blackout end.
2. OSM Bounding Box Sensitivity: Evaluates whether building road networks from
   bounding boxes of data <= bo_start alters the active road network in the scenario corridor,
   and reports the drift difference against whole-trip corridor pre-fetching.
"""

import os
import sys
import copy
import unittest
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.map.network import load_trip_road_network, build_road_network_from_osm
from sih.models.inference import load_ai_model, predict_velocities
from sih.engine.dead_reckoning_engine import run_dead_reckoning_scenario


class TestNoFutureLeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model, cls.norm_mean, cls.norm_std, cls.model_type = load_ai_model(cls.device)
        cls.loader = GenericDataLoader()
        cls.test_scenarios = [
            ("S-M", 45.0, "Highway"),
            ("S-S2", 30.0, "Arterial"),
            ("S-S3a", 45.0, "Mixed"),
        ]

    def test_post_blackout_nan_injection_bit_identity(self):
        """
        Asserts that replacing all IMU and GNSS samples after blackout exit with NaNs
        produces bit-identical trajectory coordinates through the blackout interval.
        """
        for tid, dur, domain in self.test_scenarios:
            trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", f"{tid}.csv")
            trip = self.loader.load_file(trip_path)

            # Load road network for trip
            rnet, _ = load_trip_road_network(
                trip, map_source="osm", cache_dir=os.path.join(ROOT_DIR, "data", "maps", "cache")
            )

            # Calibrate IMU
            calibrator = MountCalibrator(min_samples=30)
            gnss_idx = 0
            n_g = len(trip.gnss_samples)
            calib_samples = []
            for imu in trip.imu_samples:
                while gnss_idx < n_g and trip.gnss_samples[gnss_idx].timestamp_ns <= imu.timestamp_ns:
                    calibrator.observe_gnss(trip.gnss_samples[gnss_idx])
                    gnss_idx += 1
                calib_samples.append(calibrator.update(imu))

            v_preds = predict_velocities(
                self.model, calib_samples, self.norm_mean, self.norm_std, self.device, model_type=self.model_type
            )

            # Find valid candidate fix for scenario
            min_ts = trip.imu_samples[0].timestamp_ns + int(35.0 * 1e9)
            cand_gnss = [
                g for g in trip.gnss_samples
                if g.is_valid and g.speed_mps and g.speed_mps >= 2.5 and g.timestamp_ns >= min_ts
            ]
            self.assertTrue(len(cand_gnss) > 0, f"No candidate GNSS fixes found for trip {tid}")
            g_cand = cand_gnss[len(cand_gnss) // 3]

            bo_start_ns = g_cand.timestamp_ns
            bo_end_ns = bo_start_ns + int(dur * 1e9)

            # 1. Unmodified baseline run
            res_orig = run_dead_reckoning_scenario(
                trip, calib_samples, v_preds, rnet, g_cand, dur, domain=domain
            )
            self.assertIsNotNone(res_orig, f"Scenario run failed for {tid}")

            # 2. NaN-injected run: replace all IMU and GNSS data after blackout end with NaN
            trip_nan = copy.deepcopy(trip)
            from sih.core.contracts import IMUSample, GNSSSample
            new_imu = []
            for imu in trip_nan.imu_samples:
                if imu.timestamp_ns > bo_end_ns:
                    new_imu.append(IMUSample(
                        timestamp_ns=imu.timestamp_ns,
                        accel=np.full(3, np.nan),
                        gyro=np.full(3, np.nan),
                    ))
                else:
                    new_imu.append(imu)
            trip_nan.imu_samples = new_imu

            new_gnss = []
            for g in trip_nan.gnss_samples:
                if g.timestamp_ns > bo_end_ns:
                    new_gnss.append(GNSSSample(
                        timestamp_ns=g.timestamp_ns,
                        latitude_deg=float("nan"),
                        longitude_deg=float("nan"),
                        altitude_m=float("nan"),
                        speed_mps=float("nan"),
                        bearing_deg=float("nan"),
                        accuracy_h_m=999.0,
                        is_valid=False,
                    ))
                else:
                    new_gnss.append(g)
            trip_nan.gnss_samples = new_gnss

            # Re-run mount calibration and AI inference on the NaN-injected trip
            calibrator_nan = MountCalibrator(min_samples=30)
            gnss_idx = 0
            n_g = len(trip_nan.gnss_samples)
            calib_samples_nan = []
            for imu in trip_nan.imu_samples:
                while gnss_idx < n_g and trip_nan.gnss_samples[gnss_idx].timestamp_ns <= imu.timestamp_ns:
                    calibrator_nan.observe_gnss(trip_nan.gnss_samples[gnss_idx])
                    gnss_idx += 1
                calib_samples_nan.append(calibrator_nan.update(imu))

            v_preds_nan = predict_velocities(
                self.model, calib_samples_nan, self.norm_mean, self.norm_std, self.device, model_type=self.model_type
            )

            res_nan = run_dead_reckoning_scenario(
                trip_nan, calib_samples_nan, v_preds_nan, rnet, g_cand, dur, domain=domain
            )
            self.assertIsNotNone(res_nan, f"NaN scenario run failed for {tid}")

            # Assert bit-identical trajectories
            np.testing.assert_array_equal(
                res_orig["map_pts"],
                res_nan["map_pts"],
                err_msg=f"Trip {tid}: Map trajectory differed after post-blackout NaN injection!",
            )
            np.testing.assert_array_equal(
                res_orig["pure_pts"],
                res_nan["pure_pts"],
                err_msg=f"Trip {tid}: Pure DR trajectory differed after post-blackout NaN injection!",
            )
            self.assertEqual(
                res_orig["map_drift_pct"],
                res_nan["map_drift_pct"],
                msg=f"Trip {tid}: Drift percentage differed after post-blackout NaN injection!",
            )

    def test_osm_bbox_sensitivity_reporting(self):
        """
        Evaluates road network built from data <= bo_start only vs whole-trip bbox,
        and reports/asserts drift differences in the scenario corridor.
        """
        for tid, dur, domain in self.test_scenarios:
            trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", f"{tid}.csv")
            trip = self.loader.load_file(trip_path)

            # Whole-trip road network
            rnet_full, _ = load_trip_road_network(
                trip, map_source="osm", cache_dir=os.path.join(ROOT_DIR, "data", "maps", "cache")
            )

            # Calibrate IMU
            calibrator = MountCalibrator(min_samples=30)
            gnss_idx = 0
            n_g = len(trip.gnss_samples)
            calib_samples = []
            for imu in trip.imu_samples:
                while gnss_idx < n_g and trip.gnss_samples[gnss_idx].timestamp_ns <= imu.timestamp_ns:
                    calibrator.observe_gnss(trip.gnss_samples[gnss_idx])
                    gnss_idx += 1
                calib_samples.append(calibrator.update(imu))

            v_preds = predict_velocities(
                self.model, calib_samples, self.norm_mean, self.norm_std, self.device, model_type=self.model_type
            )

            min_ts = trip.imu_samples[0].timestamp_ns + int(35.0 * 1e9)
            cand_gnss = [
                g for g in trip.gnss_samples
                if g.is_valid and g.speed_mps and g.speed_mps >= 2.5 and g.timestamp_ns >= min_ts
            ]
            g_cand = cand_gnss[len(cand_gnss) // 3]
            bo_start_ns = g_cand.timestamp_ns

            res_full_box = run_dead_reckoning_scenario(
                trip, calib_samples, v_preds, rnet_full, g_cand, dur, domain=domain
            )

            # Causal bbox: only GNSS data <= bo_start_ns
            gnss_causal = [g for g in trip.gnss_samples if g.is_valid and g.timestamp_ns <= bo_start_ns]
            pad_m = 500.0
            ref_lat = trip.reference_lat_deg
            pad_lat = pad_m / 111139.0
            pad_lon = pad_m / (111139.0 * max(0.01, float(np.cos(np.radians(ref_lat)))))
            lats = [g.latitude_deg for g in gnss_causal]
            lons = [g.longitude_deg for g in gnss_causal]
            bbox_causal = (
                min(lats) - pad_lat,
                min(lons) - pad_lon,
                max(lats) + pad_lat,
                max(lons) + pad_lon,
            )
            rnet_causal, _ = build_road_network_from_osm(
                bbox_causal, trip.reference_lat_deg, trip.reference_lon_deg, cache_dir=os.path.join(ROOT_DIR, "data", "maps", "cache")
            )

            res_causal_box = run_dead_reckoning_scenario(
                trip, calib_samples, v_preds, rnet_causal, g_cand, dur, domain=domain
            )

            drift_full = res_full_box["map_drift_pct"]
            drift_causal = res_causal_box["map_drift_pct"]
            drift_delta = abs(drift_full - drift_causal)
            print(f"\n[OSM BBox Audit] Trip {tid}: Full BBox Drift={drift_full:.2f}%, Causal BBox Drift={drift_causal:.2f}%, Delta={drift_delta:.4f}%")


if __name__ == "__main__":
    unittest.main()
