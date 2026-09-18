"""
tests/test_map_ingestion.py
---------------------------
Unit tests for Live Indian Road Vector Ingestion & Predictive Corridor Caching Engine.
"""

import unittest
import os
import shutil
import tempfile
import time
import numpy as np

from sih.map.network import RoadSegment, RoadNetwork
from sih.map.cache import SpatialDiskCache
from sih.map.local_gis import LocalGISProvider
from sih.map.hybrid_provider import HybridIndiaMapProvider
from sih.map.corridor_manager import PredictiveCorridorManager, compute_lookahead_radius
from sih.map.provider import IRoadNetworkProvider


class MockProvider(IRoadNetworkProvider):
    """Mock provider generating synthetic test segments."""
    def __init__(self):
        self.call_count = 0

    def is_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        return False

    def get_corridor_network(self, lat: float, lon: float, radius_m: float, bearing_deg=None) -> RoadNetwork:
        self.call_count += 1
        net = RoadNetwork(cell_size_m=100.0)
        # Create a simple 200m road segment
        seg = RoadSegment(
            segment_id=f"mock_seg_{self.call_count}",
            start_enu_m=np.array([0.0, 0.0]),
            end_enu_m=np.array([100.0, 100.0]),
            start_lat_lon=(lat, lon),
            end_lat_lon=(lat + 0.001, lon + 0.001),
            bearing_deg=45.0,
            length_m=141.42,
            road_type="primary",
        )
        net.add_segment(seg)
        return net


class TestMapIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_lookahead_radius(self):
        # 0 km/h: clamped to min 800m
        r0 = compute_lookahead_radius(speed_mps=0.0)
        self.assertEqual(r0, 800.0)

        # 10 m/s (36 km/h) * 180s = 1800m
        r10 = compute_lookahead_radius(speed_mps=10.0, lookahead_time_s=180.0)
        self.assertEqual(r10, 1800.0)

        # 40 m/s (144 km/h) * 180s = 7200m -> clamped to max 6000m
        r40 = compute_lookahead_radius(speed_mps=40.0, lookahead_time_s=180.0)
        self.assertEqual(r40, 6000.0)

    def test_road_network_geojson_roundtrip(self):
        net = RoadNetwork(cell_size_m=100.0)
        seg = RoadSegment(
            segment_id="test_road_01",
            start_enu_m=np.array([10.0, 20.0]),
            end_enu_m=np.array([30.0, 40.0]),
            start_lat_lon=(18.5204, 73.8567),
            end_lat_lon=(18.5214, 73.8577),
            bearing_deg=45.0,
            length_m=28.28,
            road_type="secondary",
            speed_limit_mps=15.0,
            is_oneway=True,
        )
        net.add_segment(seg)

        geojson = net.to_geojson_dict()
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 1)

        # Restore from GeoJSON
        restored = RoadNetwork.from_geojson_dict(
            geojson,
            ref_lat=18.5204,
            ref_lon=73.8567,
        )
        self.assertEqual(len(restored.segments), 1)
        r_seg = restored.segments[0]
        self.assertEqual(r_seg.segment_id, "test_road_01")
        self.assertAlmostEqual(r_seg.start_enu_m[0], 0.0, places=1)
        self.assertAlmostEqual(r_seg.start_enu_m[1], 0.0, places=1)
        self.assertEqual(r_seg.road_type, "secondary")
        self.assertTrue(r_seg.is_oneway)

    def test_spatial_disk_cache(self):
        cache = SpatialDiskCache(cache_dir=self.temp_dir, tile_size_deg=0.05)
        lat, lon = 18.5204, 73.8567
        tile_key = cache.compute_tile_key(lat, lon)

        # Initially empty
        self.assertFalse(cache.is_tile_cached_on_disk(tile_key))

        net = RoadNetwork(cell_size_m=100.0)
        seg = RoadSegment(
            segment_id="pune_road_01",
            start_enu_m=np.array([0.0, 0.0]),
            end_enu_m=np.array([50.0, 50.0]),
            start_lat_lon=(lat, lon),
            end_lat_lon=(lat + 0.0005, lon + 0.0005),
            bearing_deg=45.0,
            length_m=70.71,
            is_oneway=True,
        )
        net.add_segment(seg)

        # Save to disk
        saved_path = cache.save_tile(tile_key, net)
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(cache.is_tile_cached_on_disk(tile_key))

        # Clear RAM and load from disk
        cache.clear_memory()
        loaded = cache.load_tile(tile_key, ref_lat=lat, ref_lon=lon)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.segments), 1)

    def test_road_network_merge(self):
        net1 = RoadNetwork(cell_size_m=100.0)
        net1.add_segment(RoadSegment("seg_A", np.array([0.0, 0.0]), np.array([10.0, 10.0]), (0, 0), (0, 0), 45.0, 14.1))

        net2 = RoadNetwork(cell_size_m=100.0)
        # One duplicate, one new
        net2.add_segment(RoadSegment("seg_A", np.array([0.0, 0.0]), np.array([10.0, 10.0]), (0, 0), (0, 0), 45.0, 14.1))
        net2.add_segment(RoadSegment("seg_B", np.array([10.0, 10.0]), np.array([20.0, 20.0]), (0, 0), (0, 0), 45.0, 14.1))

        added = net1.merge(net2)
        self.assertEqual(added, 1)
        self.assertEqual(len(net1.segments), 2)
        self.assertIn("seg_B", net1.segment_map)

    def test_local_gis_provider(self):
        gis = LocalGISProvider(gis_data_dir=self.temp_dir)
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[73.8567, 18.5204], [73.8577, 18.5214]],
                    },
                    "properties": {
                        "name": "PMGSY Rural Track #102",
                        "road_type": "rural",
                        "speed_limit_mps": 11.1,
                    },
                }
            ],
        }
        gis.add_dataset(mock_geojson)
        self.assertTrue(gis.is_cached(18.5204, 73.8567, 500.0))

        net = gis.get_corridor_network(18.5204, 73.8567, radius_m=500.0)
        self.assertGreaterEqual(len(net.segments), 1)
        self.assertEqual(net.segments[0].road_type, "rural")

    def test_predictive_corridor_manager(self):
        mock_prov = MockProvider()
        manager = PredictiveCorridorManager(
            provider=mock_prov,
            lookahead_time_s=180.0,
            min_radius_m=800.0,
            boundary_prefetch_ratio=0.40,
        )

        # First position: should trigger prefetch
        triggered = manager.notify_position(lat=18.5204, lon=73.8567, speed_mps=15.0)
        self.assertTrue(triggered)

        # Allow background thread to finish
        time.sleep(0.1)

        active = manager.get_active_network()
        self.assertGreaterEqual(len(active.segments), 1)


if __name__ == "__main__":
    unittest.main()
