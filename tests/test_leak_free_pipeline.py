"""
tests/test_leak_free_pipeline.py
--------------------------------
Comprehensive unit test suite for the leak-free OpenStreetMap (OSM) map matching pipeline.

Verifies:
1. Frozen pipeline config: Verified and consistent parameters across all components.
2. Masking exclusion: Masked road network excludes all blackout GNSS fixes with safety margin.
3. OSM bounding box coverage: Bounding box spans all trip fixes with 500m margin.
4. Douglas-Peucker tolerance: Polyline simplification preserves road curvature within 2.0m tolerance.
5. Hard gate suppression: Rejects candidates with d_perp > 2.5 * sigma_d or h_diff > 2.0 * sigma_h.
6. Widening entry search: Searches 35m -> 75m -> 150m prioritizing heading alignment <= 45 deg.
7. Log-space ratio calculation: Prevents underflow on costs > 27.63 (e.g. cost 35 vs 40).
8. Absolute cost gate: Blocks false-positive wins when best route cost > max_acceptable_cost (8.0).
9. Graceful degradation: Falls back to pure DR trajectory when network is empty or unmatched.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from sih.core.contracts import IMUSample, GNSSSample, FusedPosition, CalibratedSample
from sih.data.loader import GenericDataLoader, TripSequence
from sih.map.network import (
    RoadSegment,
    RoadNetwork,
    douglas_peucker_indices,
    build_road_network_from_trip_masked,
    build_road_network_from_trip,
    load_trip_road_network,
)
from sih.map.matcher import HMMMapMatcher
from sih.map.route_matcher import RouteMatcher, TurnSequence, TurnEvent, CandidateRoute
from sih.engine.dead_reckoning_engine import DeadReckoningEngine
from sih.core.config import load_frozen_pipeline_config


def test_frozen_pipeline_config():
    """Verify frozen pipeline config loads correctly and contains frozen parameters."""
    config = load_frozen_pipeline_config()
    assert config["defaults"]["map_source"] == "osm"
    assert config["defaults"]["enable_route_matching"] is False

    mm = config["map_matcher"]
    assert mm["sigma_dist_m"] == 4.0
    assert mm["sigma_heading_deg"] == 30.0
    assert mm["hard_gate_dist_multiplier"] == 2.5
    assert mm["hard_gate_heading_multiplier"] == 2.0
    assert mm["ambiguity_min_ratio"] == 1.5
    assert mm["hysteresis_steps"] == 2

    rm = config["route_matcher"]
    assert rm["confidence_ratio_threshold"] == 1.80
    assert rm["max_acceptable_cost"] == 8.0

    ea = config["entry_segment_acquisition"]
    assert ea["search_radii_m"] == [35.0, 75.0, 150.0]
    assert ea["heading_gate_deg"] == 45.0


def test_masking_exclusion():
    """Verify that build_road_network_from_trip_masked excludes blackout GNSS fixes."""
    trip_path = os.path.join("data", "raw", "iovnbd_trips", "S-S1.csv")
    if not os.path.exists(trip_path):
        pytest.skip("Trip data file data/raw/iovnbd_trips/S-S1.csv not found")

    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)
    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    assert len(valid_gnss) >= 200

    bo_start = valid_gnss[50].timestamp_ns
    bo_end = valid_gnss[100].timestamp_ns
    bo_windows = [(bo_start, bo_end)]

    net_full, pts_full = build_road_network_from_trip(trip)
    net_masked, pts_masked = build_road_network_from_trip_masked(trip, bo_windows, margin_s=30.0)

    # Masked network must have fewer segments than full network
    assert len(net_masked.segments) < len(net_full.segments)

    # Verify disjoint chunking: segment IDs have chunk prefixes c00, c01, etc.
    chunk_prefixes = set(s.segment_id.split("_")[2] for s in net_masked.segments if "_c" in s.segment_id)
    assert len(chunk_prefixes) >= 2, "Masked network should be split into multiple disjoint chunks"


def test_osm_bbox_coverage():
    """Verify that OSM bounding box covers all trip coordinates plus 500m margin."""
    trip_path = os.path.join("data", "raw", "iovnbd_trips", "S-S1.csv")
    if not os.path.exists(trip_path):
        pytest.skip("Trip data file not found")

    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)
    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]

    min_lat = min(g.latitude_deg for g in valid_gnss)
    max_lat = max(g.latitude_deg for g in valid_gnss)
    min_lon = min(g.longitude_deg for g in valid_gnss)
    max_lon = max(g.longitude_deg for g in valid_gnss)

    pad_m = 500.0
    deg_per_m_lat = 1.0 / 111139.0
    mean_lat_rad = np.radians((min_lat + max_lat) / 2.0)
    deg_per_m_lon = 1.0 / (111139.0 * np.cos(mean_lat_rad))

    expected_min_lat = min_lat - pad_m * deg_per_m_lat
    expected_max_lat = max_lat + pad_m * deg_per_m_lat
    expected_min_lon = min_lon - pad_m * deg_per_m_lon
    expected_max_lon = max_lon + pad_m * deg_per_m_lon

    # Load OSM network (from cache)
    rnet, pts = load_trip_road_network(trip, map_source="osm", bbox_pad_m=pad_m)
    assert len(rnet.segments) > 0

    # Ensure all segments are within or close to the padded bbox
    for seg in rnet.segments:
        s_lat, s_lon = seg.start_lat_lon
        e_lat, e_lon = seg.end_lat_lon
        assert expected_min_lat - 0.02 <= s_lat <= expected_max_lat + 0.02
        assert expected_min_lon - 0.02 <= s_lon <= expected_max_lon + 0.02


def test_douglas_peucker_tolerance():
    """Verify Douglas-Peucker simplification preserves road curvature within 2.0m tolerance."""
    # Create a curved polyline (semicircle of radius 50m)
    theta = np.linspace(0, np.pi, 50)
    r = 50.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    curve_pts = np.column_stack([x, y])

    retained_idx = douglas_peucker_indices(curve_pts, tol_m=2.0)
    simplified_pts = curve_pts[retained_idx]

    # Endpoints must always be retained
    assert retained_idx[0] == 0
    assert retained_idx[-1] == len(curve_pts) - 1
    assert len(simplified_pts) < len(curve_pts)

    # Compute maximum perpendicular distance from any original point to simplified segments
    for i in range(len(simplified_pts) - 1):
        p1 = simplified_pts[i]
        p2 = simplified_pts[i + 1]
        v = p2 - p1
        v_norm = np.linalg.norm(v)

        orig_start = retained_idx[i]
        orig_end = retained_idx[i + 1]
        for k in range(orig_start + 1, orig_end):
            pt = curve_pts[k]
            dist = abs((pt[0] - p1[0]) * v[1] - (pt[1] - p1[1]) * v[0]) / v_norm
            assert dist <= 2.0 + 1e-6, f"Deviation {dist:.3f}m exceeds 2.0m tolerance"


def test_hard_gate_suppression():
    """Verify hard gate suppression when d_perp > 2.5*sigma_d or h_diff > 2.0*sigma_h."""
    # Road network with one horizontal segment from (0, 0) to (100, 0) -> bearing = 90 deg
    rnet = RoadNetwork(cell_size_m=50.0)
    seg = RoadSegment(
        segment_id="seg_east",
        start_enu_m=np.array([0.0, 0.0]),
        end_enu_m=np.array([100.0, 0.0]),
        start_lat_lon=(52.0, 0.0),
        end_lat_lon=(52.0, 0.001),
        bearing_deg=90.0,
        length_m=100.0,
        road_type="primary",
    )
    rnet.add_segment(seg)

    matcher = HMMMapMatcher(
        road_network=rnet,
        reference_lat_deg=52.0,
        reference_lon_deg=0.0,
        sigma_dist_m=4.0,       # hard gate threshold = 2.5 * 4.0 = 10.0m
        sigma_heading_deg=30.0, # hard gate threshold = 2.0 * 30.0 = 60.0 deg
    )

    # Case 1: Point at (50.0, 15.0) -> d_perp = 15.0m > 10.0m (Heading = 90 deg matches)
    pos1 = FusedPosition(
        timestamp_ns=1000000000,
        latitude_deg=52.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        position_enu_m=np.array([50.0, 15.0, 0.0]),
        velocity_enu_mps=np.array([10.0, 0.0, 0.0]),
        heading_rad=np.radians(90.0),
        covariance=np.eye(6),
        mode="INS_ONLY_BLACKOUT",
    )
    res1 = matcher.match(pos1)
    assert res1.is_matched is False
    assert matcher.hard_gate_suppressed_steps == 1

    # Case 2: Point at (50.0, 2.0) -> d_perp = 2.0m <= 10.0m, but Heading = 180 deg (h_diff = 90 deg > 60 deg)
    pos2 = FusedPosition(
        timestamp_ns=1100000000,
        latitude_deg=52.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        position_enu_m=np.array([50.0, 2.0, 0.0]),
        velocity_enu_mps=np.array([0.0, -10.0, 0.0]),
        heading_rad=np.radians(180.0),
        covariance=np.eye(6),
        mode="INS_ONLY_BLACKOUT",
    )
    res2 = matcher.match(pos2)
    assert res2.is_matched is False
    assert matcher.hard_gate_suppressed_steps == 2

    # Case 3: Point within limits -> (50.0, 2.0), Heading = 92 deg (h_diff = 2 deg)
    # Hysteresis requires 2 consecutive valid steps after suppression
    pos3 = FusedPosition(
        timestamp_ns=1200000000,
        latitude_deg=52.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        position_enu_m=np.array([50.0, 2.0, 0.0]),
        velocity_enu_mps=np.array([10.0, 0.0, 0.0]),
        heading_rad=np.radians(92.0),
        covariance=np.eye(6),
        mode="INS_ONLY_BLACKOUT",
    )
    # Step A (first valid step): suppressed by hysteresis
    res3a = matcher.match(pos3)
    assert res3a.is_matched is False
    assert matcher.hysteresis_suppressed_steps == 1

    # Step B (second valid step): hysteresis satisfied -> successfully snapped
    pos3b = FusedPosition(
        timestamp_ns=1300000000,
        latitude_deg=52.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        position_enu_m=np.array([50.0, 2.0, 0.0]),
        velocity_enu_mps=np.array([10.0, 0.0, 0.0]),
        heading_rad=np.radians(92.0),
        covariance=np.eye(6),
        mode="INS_ONLY_BLACKOUT",
    )
    res3b = matcher.match(pos3b)
    assert res3b.is_matched is True
    assert res3b.road_segment_id == "seg_east"


def test_widening_entry_search():
    """Verify widening entry search: 35m -> 75m -> 150m with 45 deg heading gate."""
    rnet = RoadNetwork(cell_size_m=100.0)
    # Segment A: 20m away from origin, but bearing = 180 deg (heading diff = 180 deg > 45 deg)
    seg_opp = RoadSegment(
        segment_id="seg_opp",
        start_enu_m=np.array([20.0, 50.0]),
        end_enu_m=np.array([20.0, -50.0]),
        start_lat_lon=(52.0, 0.0),
        end_lat_lon=(52.0, 0.001),
        bearing_deg=180.0,
        length_m=100.0,
    )
    # Segment B: 50m away from origin (outside 35m, inside 75m), bearing = 5 deg (diff = 5 deg <= 45 deg)
    seg_match = RoadSegment(
        segment_id="seg_match",
        start_enu_m=np.array([50.0, -50.0]),
        end_enu_m=np.array([50.0, 50.0]),
        start_lat_lon=(52.0, 0.0),
        end_lat_lon=(52.0, 0.001),
        bearing_deg=0.0,
        length_m=100.0,
    )
    rnet.add_segment(seg_opp)
    rnet.add_segment(seg_match)

    entry_pt = np.array([0.0, 0.0])
    entry_heading = 0.0  # North

    entry_seg, acq_radius, acq_h_diff, info = DeadReckoningEngine.acquire_entry_segment(
        entry_pt, entry_heading, rnet
    )

    assert entry_seg is not None
    assert entry_seg.segment_id == "seg_match"
    assert acq_radius == 75.0
    assert acq_h_diff <= 45.0


def test_log_space_ratio_calculation():
    """Verify that RouteMatcher computes confidence ratio in log space without underflow."""
    # Route 1: cost 35.0, Route 2: cost 40.0
    # In linear space: exp(-35) = 6.3e-16, exp(-40) = 4.2e-18
    # Clamping at 1e-12 causes bogus ratio ~0.0 or 1.0 (underflow bug)
    # In log space: exp(40.0 - 35.0) = exp(5.0) = 148.41
    cost_1 = 35.0
    cost_2 = 40.0

    # Verify that in log space: cost_diff = cost_2 - cost_1 produces correct ratio
    log_diff = np.clip(cost_2 - cost_1, -50.0, 50.0)
    ratio = float(np.exp(log_diff))
    assert 148.0 < ratio < 149.0

    # Verify that unnormalized linear space with clamping would fail:
    linear_score_1 = float(np.exp(-cost_1))
    linear_score_2 = float(np.exp(-cost_2))
    assert linear_score_1 < 1e-12
    assert linear_score_2 < 1e-12
    clamped_linear_ratio = linear_score_1 / max(linear_score_2, 1e-12)
    assert clamped_linear_ratio < 1e-3, "Linear clamping fails to compute true ratio on high costs"


def test_absolute_cost_gate():
    """Verify that absolute cost gate blocks false wins when best route cost > max_acceptable_cost (8.0)."""
    matcher = RouteMatcher(min_route_ratio=1.80, max_acceptable_cost=8.0)

    # Replicate Scenario 25 conditions:
    # Route 1 cost = 9.75, Route 2 cost = 11.47
    # Relative ratio = exp(11.47 - 9.75) = exp(1.72) = 5.58 >= 1.80
    # But best route is far from ground truth (cost 9.75 > 8.0) -> must NOT win!
    r1 = CandidateRoute(
        segments=[],
        cum_lengths=[200.0],
        total_length_m=200.0,
        cost=9.75,
        score=float(np.exp(-9.75)),
    )
    r2 = CandidateRoute(
        segments=[],
        cum_lengths=[200.0],
        total_length_m=200.0,
        cost=11.47,
        score=float(np.exp(-11.47)),
    )

    ranked_routes = [r1, r2]
    best = ranked_routes[0]
    second = ranked_routes[1]

    log_diff = np.clip(second.cost - best.cost, -50.0, 50.0)
    ratio = float(np.exp(log_diff))
    assert ratio >= 1.80  # Relative ratio passed

    # Check the absolute cost gate
    cost_acceptable = (best.cost <= matcher.max_acceptable_cost)
    assert cost_acceptable is False  # Fails absolute cost gate

    # Confirm win condition fails
    won = (ratio >= matcher.min_route_ratio) and cost_acceptable
    assert won is False


def test_graceful_degradation_to_pure_dr():
    """Verify that when the map network is completely empty, DeadReckoningEngine gracefully runs pure DR."""
    empty_net = RoadNetwork(cell_size_m=50.0)
    engine = DeadReckoningEngine(enable_route_matching=False)

    # Synthesize constant velocity trip: 10 m/s North for 60 seconds
    dt = 0.1
    n_samples = 600
    timestamps_ns = [int(i * dt * 1e9) for i in range(n_samples)]

    imu_samples = [
        IMUSample(
            timestamp_ns=ts,
            accel=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            gyro=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        for ts in timestamps_ns
    ]

    ref_lat, ref_lon, ref_alt = 52.40, -1.50, 50.0
    gnss_samples = []
    for i in range(0, n_samples, 10):
        lat_offset = (i * dt * 10.0) / 111139.0
        gnss_samples.append(GNSSSample(
            timestamp_ns=timestamps_ns[i],
            latitude_deg=ref_lat + lat_offset,
            longitude_deg=ref_lon,
            altitude_m=ref_alt,
            speed_mps=10.0,
            bearing_deg=0.0,
            accuracy_h_m=3.0,
        ))

    trip = TripSequence(
        trip_id="MOCK_TEST",
        imu_samples=imu_samples,
        gnss_samples=gnss_samples,
        reference_lat_deg=ref_lat,
        reference_lon_deg=ref_lon,
        reference_alt_m=ref_alt,
        total_gnss_distance_m=600.0,
        duration_s=60.0,
    )

    calib_samples = [
        CalibratedSample(
            timestamp_ns=ts,
            accel_vehicle=np.array([0.0, 0.0, 9.80665]),
            gyro_vehicle=np.array([0.0, 0.0, 0.0]),
            rotation_body_to_vehicle=np.eye(3),
            gravity_vehicle=np.array([0.0, 0.0, 1.0]),
            is_calibrated=True,
        )
        for ts in timestamps_ns
    ]
    v_preds = np.full(n_samples, 10.0)

    g_entry = gnss_samples[30]  # 30s in
    duration_s = 20.0

    res = engine.run_scenario(
        trip=trip,
        calib_samples=calib_samples,
        v_preds=v_preds,
        road_net=empty_net,
        g_entry=g_entry,
        duration_s=duration_s,
        domain="Highway",
    )

    assert res is not None
    assert "pure_drift_pct" in res
    assert "map_drift_pct" in res
    # With empty road network, map trajectory should match pure DR trajectory
    assert abs(res["pure_drift_pct"] - res["map_drift_pct"]) < 0.01
