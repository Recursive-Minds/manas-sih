"""
Unit tests and real data verification for the Schema-flexible Data Loader and Geodetic Math.
"""

import unittest
import os
import numpy as np
import pandas as pd

from sih.data.schema import ColumnMapping
from sih.data.geo import (
    geodetic_to_enu,
    enu_to_geodetic,
    haversine_distance_m,
    compute_cumulative_distance,
)
from sih.data.loader import (
    GenericDataLoader,
    split_dataset_by_trips,
    TripSequence,
)


class TestGeoAndDataLoader(unittest.TestCase):

    def test_geodetic_to_enu_roundtrip(self):
        ref_lat, ref_lon, ref_alt = 12.9716, 77.5946, 920.0
        # Target point ~1km East and ~500m North
        target_lat = ref_lat + 0.0045
        target_lon = ref_lon + 0.0092
        target_alt = ref_alt + 15.0

        enu = geodetic_to_enu(target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt)
        self.assertEqual(enu.shape, (3,))

        lat_back, lon_back, alt_back = enu_to_geodetic(enu[0], enu[1], enu[2], ref_lat, ref_lon, ref_alt)
        self.assertAlmostEqual(lat_back, target_lat, places=6)
        self.assertAlmostEqual(lon_back, target_lon, places=6)
        self.assertAlmostEqual(alt_back, target_alt, places=2)

    def test_synthetic_trip_loading(self):
        # Create synthetic CSV with unconventional column headers
        df = pd.DataFrame({
            "t_ms": [0, 100, 200, 300, 400],
            "AX_Body (m/s2)": [0.1, 0.2, 0.3, 0.4, 0.5],
            "AY_Body (m/s2)": [0.0, 0.0, 0.1, 0.0, -0.1],
            "AZ_Body (m/s2)": [9.8, 9.81, 9.79, 9.82, 9.80],
            "Roll_rate (rad/s)": [0.01, -0.01, 0.0, 0.02, 0.0],
            "Pitch_rate (rad/s)": [0.0, 0.01, -0.01, 0.0, 0.0],
            "Yaw_rate (rad/s)": [0.05, 0.05, 0.05, 0.04, 0.06],
            "GPS Lat": [12.9716, 12.9716, 12.97165, 12.97165, 12.97170],
            "GPS Long": [77.5946, 77.5946, 77.59465, 77.59465, 77.59470],
            "GPS Altitude (m)": [920.0, 920.0, 920.1, 920.1, 920.2],
            "GPS Speed (km/h)": [36.0, 36.0, 37.0, 37.0, 38.0],
        })

        loader = GenericDataLoader()
        trip = loader.load_dataframe(df, trip_id="synthetic_01")

        self.assertEqual(trip.trip_id, "synthetic_01")
        self.assertEqual(len(trip.imu_samples), 5)
        # GNSS updates filtered down to unique fixes
        self.assertEqual(len(trip.gnss_samples), 3)
        self.assertAlmostEqual(trip.gnss_samples[0].speed_mps, 10.0, places=2)  # 36 km/h = 10 m/s
        self.assertEqual(trip.imu_samples[0].timestamp_ns, 0)
        self.assertEqual(trip.imu_samples[1].timestamp_ns, 100_000_000)

    def test_strict_trip_level_splitting(self):
        trips = [
            TripSequence(f"trip_{i}", [], [], 0, 0, 0, 100.0, 50.0, {})
            for i in range(10)
        ]
        train, val, test = split_dataset_by_trips(trips, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        self.assertEqual(len(train), 7)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 2)

        # Ensure no overlap between splits (zero leakage)
        train_ids = {t.trip_id for t in train}
        val_ids = {t.trip_id for t in val}
        test_ids = {t.trip_id for t in test}

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_real_iovnbd_trip_loading(self):
        real_csv = os.path.join("data", "raw", "iovnbd_trips", "S-S1.csv")
        if not os.path.exists(real_csv):
            self.skipTest(f"Real data file not found: {real_csv}")

        loader = GenericDataLoader()
        trip = loader.load_file(real_csv)

        self.assertEqual(trip.trip_id, "S-S1")
        self.assertGreater(len(trip.imu_samples), 50000)
        self.assertGreater(len(trip.gnss_samples), 500)
        self.assertGreater(trip.duration_s, 5000.0)
        self.assertGreater(trip.total_gnss_distance_m, 20000.0)  # > 20 km drive

        # Verify rate ~10Hz
        approx_rate = trip.metadata["approx_imu_rate_hz"]
        self.assertAlmostEqual(approx_rate, 10.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
