"""
tests/test_app_mount_seed.py
----------------------------
Verifies the saved-alignment seeding workaround for MountCalibrator:
1. Seeds MountCalibrator with a pre-saved MountAlignment and _yaw_locked = True.
2. Streams 5 minutes (300 seconds / 3,000 IMU samples + 300 GNSS fixes) of real
   driving data from Trip S-M containing genuine turns.
3. Asserts that the alignment is NEVER overwritten by the initial fallback heuristic
   or subsequent turn events.
"""

import os
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator, MountAlignment


class TestAppMountSeed(unittest.TestCase):
    def test_seeded_alignment_immutability_over_real_stream(self):
        loader = GenericDataLoader()
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-M.csv")
        trip = loader.load_file(trip_path)

        # Construct a distinct synthetic saved alignment (simulating a phone docked tilted in cradle)
        saved_rot = R.from_euler("xyz", [12.5, -8.0, 45.0], degrees=True)
        saved_alignment = MountAlignment(
            is_calibrated=True,
            R_phone_to_vehicle=saved_rot,
            forward_axis_phone=saved_rot.as_matrix()[0, :],
            lateral_axis_phone=saved_rot.as_matrix()[1, :],
            vertical_axis_phone=saved_rot.as_matrix()[2, :],
            yaw_axis_index=1,
            yaw_axis_sign=-1.0,
            mount_yaw_offset_rad=0.0,
            pitch_deg=12.5,
            roll_deg=-8.0,
        )

        calibrator = MountCalibrator(min_samples=30)
        # Apply the seeding workaround
        calibrator._alignment = saved_alignment
        calibrator._yaw_locked = True

        self.assertTrue(calibrator.is_calibrated)
        self.assertTrue(calibrator.is_aligned())

        # Select a 5-minute slice (300 seconds @ 10 Hz = 3000 IMU samples)
        t_start = trip.imu_samples[500].timestamp_ns
        t_end = t_start + int(300.0 * 1e9)

        imu_slice = [im for im in trip.imu_samples if t_start <= im.timestamp_ns <= t_end]
        gnss_slice = [g for g in trip.gnss_samples if t_start <= g.timestamp_ns <= t_end]

        self.assertGreaterEqual(len(imu_slice), 2900)
        self.assertGreaterEqual(len(gnss_slice), 20)

        # Run an unseeded control calibrator on the exact same stream to demonstrate contrast
        control_calibrator = MountCalibrator(min_samples=30)

        # Stream chronologically through both calibrators
        g_idx = 0
        n_g = len(gnss_slice)
        for imu in imu_slice:
            while g_idx < n_g and gnss_slice[g_idx].timestamp_ns <= imu.timestamp_ns:
                calibrator.observe_gnss(gnss_slice[g_idx])
                control_calibrator.observe_gnss(gnss_slice[g_idx])
                g_idx += 1
            cal_sample = calibrator.update(imu)
            control_sample = control_calibrator.update(imu)
            self.assertTrue(cal_sample.is_calibrated)
            np.testing.assert_allclose(cal_sample.rotation_body_to_vehicle, saved_rot.as_matrix())

        # Assert alignment was completely preserved
        curr_alignment = calibrator.alignment
        self.assertIsNotNone(curr_alignment)
        self.assertTrue(curr_alignment.is_calibrated)
        self.assertEqual(curr_alignment.yaw_axis_index, 1)
        self.assertEqual(curr_alignment.yaw_axis_sign, -1.0)
        self.assertAlmostEqual(curr_alignment.pitch_deg, 12.5, places=5)
        self.assertAlmostEqual(curr_alignment.roll_deg, -8.0, places=5)
        np.testing.assert_allclose(curr_alignment.R_phone_to_vehicle.as_matrix(), saved_rot.as_matrix())
        self.assertTrue(calibrator._yaw_locked)
        self.assertEqual(len(calibrator._turn_events), 0, "Turn events should be bypassed when _yaw_locked is True")


if __name__ == "__main__":
    unittest.main()
