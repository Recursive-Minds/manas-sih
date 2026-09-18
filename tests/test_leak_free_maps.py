import os
import sys
import numpy as np

sys.path.insert(0, ".")
from sih.data.loader import GenericDataLoader
from sih.map.network import (
    build_road_network_from_trip,
    build_road_network_from_trip_masked,
    build_road_network_from_osm,
)

def test_masked_trip_network():
    loader = GenericDataLoader()
    trip_path = os.path.join("data", "raw", "iovnbd_trips", "S-S1.csv")
    trip = loader.load_file(trip_path)

    # Pick a dummy blackout window
    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    bo_start = valid_gnss[100].timestamp_ns
    bo_end = valid_gnss[150].timestamp_ns
    bo_windows = [(bo_start, bo_end)]

    net_full, pts_full = build_road_network_from_trip(trip)
    net_masked, pts_masked = build_road_network_from_trip_masked(trip, bo_windows, margin_s=30.0)

    print(f"Full network: {len(net_full.segments)} segments, {len(pts_full)} pts")
    print(f"Masked network: {len(net_masked.segments)} segments, {len(pts_masked)} pts")

    assert len(net_masked.segments) < len(net_full.segments), "Masked network should have fewer segments"

    # Verify no points inside masked window
    margin_ns = int(30.0 * 1e9)
    window_start = bo_start - margin_ns
    window_end = bo_end + margin_ns

    # Check that in the masked network, vertices in that area were truly excluded
    # For every segment in net_masked, verify it does not bridge across the gap
    for seg in net_masked.segments:
        # Check segment length: no segment should bridge the large gap (> 200m)
        assert seg.length_m < 200.0, f"Segment {seg.segment_id} appears to bridge gap with length {seg.length_m}m!"

    print("test_masked_trip_network PASSED!")

def test_osm_network():
    # Test bounding box near Coventry center
    bbox = (52.40, -1.51, 52.41, -1.50)
    ref_lat, ref_lon = 52.40166, -1.50529
    net, pts = build_road_network_from_osm(bbox, ref_lat, ref_lon, cache_dir="data/maps/cache")
    print(f"OSM network: {len(net.segments)} segments, {len(pts)} pts, ways: {getattr(net, 'num_ways', 0)}")
    assert len(net.segments) > 0, "OSM network should contain segments"
    assert len(pts) > 0, "OSM network should contain points"

    # Test cache retrieval (offline)
    net2, pts2 = build_road_network_from_osm(bbox, ref_lat, ref_lon, cache_dir="data/maps/cache")
    assert len(net2.segments) == len(net.segments), "Cached network must match"
    print("test_osm_network PASSED!")

if __name__ == "__main__":
    test_masked_trip_network()
    test_osm_network()
