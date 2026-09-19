"""
Road network geometry and spatial representation for Phase 4 Map Matching.

Stores road segments in both WGS-84 (Lat/Lon) and Local Tangent Plane (ENU).
Provides fast spatial indexing for candidate segment retrieval within search radius.
Supports graceful degradation for unmapped rural tracks and farmland.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from sih.data.geo import geodetic_to_enu


@dataclass(slots=True, frozen=True)
class RoadSegment:
    """
    Directed road centerline segment.
    """
    segment_id: str
    start_enu_m: np.ndarray        # Shape (2,), [East, North] in meters
    end_enu_m: np.ndarray          # Shape (2,), [East, North] in meters
    start_lat_lon: Tuple[float, float]
    end_lat_lon: Tuple[float, float]
    bearing_deg: float             # Azimuth in degrees from True North (0 to 360)
    length_m: float
    road_type: str = "motorway"    # motorway, primary, secondary, tertiary, rural, unpaved
    speed_limit_mps: float = 25.0
    is_oneway: bool = False
    start_node_id: Optional[Any] = None
    end_node_id: Optional[Any] = None

    def project_point(self, p_enu: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Orthogonally project 2D ENU point onto this segment.
        Returns:
            projected_point (np.ndarray): Closest point on line segment
            dist_perp_m (float): Perpendicular distance in meters
            fraction (float): Fractional distance along segment [0.0, 1.0]
        """
        a = self.start_enu_m
        b = self.end_enu_m
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-6:
            dist = float(np.linalg.norm(p_enu - a))
            return a.copy(), dist, 0.0

        ap = p_enu - a
        t = float(np.dot(ap, ab) / ab_len_sq)
        t_clamped = max(0.0, min(1.0, t))
        proj = a + t_clamped * ab
        dist_perp = float(np.linalg.norm(p_enu - proj))
        return proj, dist_perp, t_clamped


    @property
    def tangent_unit(self) -> np.ndarray:
        """Unit vector along road segment direction [East, North]."""
        ab = self.end_enu_m - self.start_enu_m
        norm = float(np.linalg.norm(ab))
        if norm > 1e-6:
            return ab / norm
        b_rad = np.radians(self.bearing_deg)
        return np.array([np.sin(b_rad), np.cos(b_rad)], dtype=np.float64)


def douglas_peucker_indices(points: np.ndarray, tol_m: float) -> List[int]:
    """
    Computes Douglas-Peucker polyline simplification indices on 2D points (N, 2).
    Guarantees maximum perpendicular deviation from original polyline <= tol_m.
    Returns sorted list of retained vertex indices including 0 and N-1.
    """
    n = len(points)
    if n <= 2:
        return list(range(n))

    def _dp(start_idx: int, end_idx: int) -> List[int]:
        if end_idx <= start_idx + 1:
            return [start_idx, end_idx]

        p_start = points[start_idx]
        p_end = points[end_idx]
        v = p_end - p_start
        v_len_sq = float(v[0]**2 + v[1]**2)

        sub_pts = points[start_idx + 1 : end_idx]
        if v_len_sq < 1e-6:
            dists = np.hypot(sub_pts[:, 0] - p_start[0], sub_pts[:, 1] - p_start[1])
        else:
            dists = np.abs(
                (sub_pts[:, 0] - p_start[0]) * v[1] - (sub_pts[:, 1] - p_start[1]) * v[0]
            ) / np.sqrt(v_len_sq)

        max_idx_sub = int(np.argmax(dists))
        max_dist = float(dists[max_idx_sub])

        if max_dist > tol_m:
            split = start_idx + 1 + max_idx_sub
            left = _dp(start_idx, split)
            right = _dp(split, end_idx)
            return left[:-1] + right
        else:
            return [start_idx, end_idx]

    return _dp(0, n - 1)


class RoadNetwork:

    """
    Spatial database of road segments with grid-based spatial indexing.
    """
    def __init__(self, cell_size_m: float = 100.0) -> None:
        self.segments: List[RoadSegment] = []
        self.segment_map: dict[str, RoadSegment] = {}
        self.cell_size_m = cell_size_m
        self.grid: dict[Tuple[int, int], List[int]] = {}

    def add_segment(self, segment: RoadSegment) -> None:
        idx = len(self.segments)
        self.segments.append(segment)
        self.segment_map[segment.segment_id] = segment

        # Index into spatial grid cells
        min_e = min(segment.start_enu_m[0], segment.end_enu_m[0])
        max_e = max(segment.start_enu_m[0], segment.end_enu_m[0])
        min_n = min(segment.start_enu_m[1], segment.end_enu_m[1])
        max_n = max(segment.start_enu_m[1], segment.end_enu_m[1])

        cell_min_x = int(np.floor(min_e / self.cell_size_m))
        cell_max_x = int(np.floor(max_e / self.cell_size_m))
        cell_min_y = int(np.floor(min_n / self.cell_size_m))
        cell_max_y = int(np.floor(max_n / self.cell_size_m))

        for cx in range(cell_min_x, cell_max_x + 1):
            for cy in range(cell_min_y, cell_max_y + 1):
                key = (cx, cy)
                if key not in self.grid:
                    self.grid[key] = []
                self.grid[key].append(idx)

    def find_candidates(self, p_enu: np.ndarray, radius_m: float = 40.0) -> List[RoadSegment]:
        """
        Find all road segments within radius_m of p_enu.
        """
        cx_min = int(np.floor((p_enu[0] - radius_m) / self.cell_size_m))
        cx_max = int(np.floor((p_enu[0] + radius_m) / self.cell_size_m))
        cy_min = int(np.floor((p_enu[1] - radius_m) / self.cell_size_m))
        cy_max = int(np.floor((p_enu[1] + radius_m) / self.cell_size_m))

        candidate_indices = set()
        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                key = (cx, cy)
                if key in self.grid:
                    candidate_indices.update(self.grid[key])

        results = []
        for idx in candidate_indices:
            seg = self.segments[idx]
            _, dist, _ = seg.project_point(p_enu)
            if dist <= radius_m:
                results.append(seg)

        return results

    @classmethod
    def from_polyline_coords(
        cls,
        coords_enu: np.ndarray,
        coords_lat_lon: List[Tuple[float, float]],
        road_id_prefix: str = "road",
        road_type: str = "primary",
        cell_size_m: float = 100.0,
    ) -> RoadNetwork:
        """
        Construct a connected road network from an ordered polyline of vertices.
        """
        network = cls(cell_size_m=cell_size_m)
        num_pts = len(coords_enu)

        for i in range(num_pts - 1):
            p1 = coords_enu[i][:2]
            p2 = coords_enu[i + 1][:2]
            diff = p2 - p1
            length = float(np.linalg.norm(diff))
            if length < 0.5:
                continue

            bearing = float(np.degrees(np.arctan2(diff[0], diff[1]))) % 360.0
            seg_id = f"{road_id_prefix}_{i:04d}"

            seg = RoadSegment(
                segment_id=seg_id,
                start_enu_m=p1,
                end_enu_m=p2,
                start_lat_lon=coords_lat_lon[i],
                end_lat_lon=coords_lat_lon[i + 1],
                bearing_deg=bearing,
                length_m=length,
                road_type=road_type,
            )
            network.add_segment(seg)

        return network

    def merge(self, other: RoadNetwork) -> int:
        """
        Merges another RoadNetwork into this instance.
        Skips duplicate segment IDs. Returns count of newly added segments.
        """
        added_count = 0
        for seg in other.segments:
            if seg.segment_id not in self.segment_map:
                self.add_segment(seg)
                added_count += 1
        return added_count

    def to_geojson_dict(self) -> Dict[str, Any]:
        """
        Serializes the RoadNetwork to a standard GeoJSON FeatureCollection.
        """
        features = []
        for seg in self.segments:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [seg.start_lat_lon[1], seg.start_lat_lon[0]],  # [lon, lat] in GeoJSON
                        [seg.end_lat_lon[1], seg.end_lat_lon[0]],
                    ],
                },
                "properties": {
                    "segment_id": seg.segment_id,
                    "bearing_deg": seg.bearing_deg,
                    "length_m": seg.length_m,
                    "road_type": seg.road_type,
                    "speed_limit_mps": seg.speed_limit_mps,
                    "is_oneway": seg.is_oneway,
                },
            }
            features.append(feat)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    @classmethod
    def from_geojson_dict(
        cls,
        data: Dict[str, Any],
        ref_lat: float,
        ref_lon: float,
        ref_alt: float = 0.0,
        cell_size_m: float = 100.0,
        road_id_prefix: str = "gis_road",
        simplification_tol_m: float = 2.0,
    ) -> RoadNetwork:
        """
        Constructs a RoadNetwork from a GeoJSON FeatureCollection (OSM, PMGSY, or Bhuvan).
        Splits ways at junction nodes to maintain topology, and emits chains of RoadSegments
        following true road curvature with configurable Douglas-Peucker simplification.
        """
        network = cls(cell_size_m=cell_size_m)
        features = data.get("features", [])

        # Pass 1: Count node references across all ways to detect intersection nodes
        node_counts: Dict[Any, int] = {}
        for feat in features:
            props = feat.get("properties", {})
            nodes = props.get("nodes", [])
            for nid in nodes:
                node_counts[nid] = node_counts.get(nid, 0) + 1

        seg_counter = 0
        for feat_idx, feat in enumerate(features):
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            g_type = geom.get("type", "")
            raw_coords = geom.get("coordinates", [])
            nodes = props.get("nodes", [])

            lines = []
            lines_nodes = []
            if g_type == "LineString":
                lines = [raw_coords]
                lines_nodes = [nodes] if (nodes and len(nodes) == len(raw_coords)) else [None]
            elif g_type == "MultiLineString":
                lines = raw_coords
                lines_nodes = [None] * len(lines)

            road_type = props.get("road_type", props.get("highway", "primary"))
            speed_limit = float(props.get("speed_limit_mps", 25.0))
            is_oneway = bool(props.get("is_oneway", False))

            for line_idx, line in enumerate(lines):
                if len(line) < 2:
                    continue

                line_nodes = lines_nodes[line_idx] if line_idx < len(lines_nodes) else None

                if line_nodes and len(line_nodes) == len(line):
                    # Split way into inter-junction stretches at junction nodes (node_counts > 1)
                    split_indices = [0]
                    for i in range(1, len(line) - 1):
                        if node_counts.get(line_nodes[i], 0) > 1:
                            split_indices.append(i)
                    split_indices.append(len(line) - 1)

                    for s_idx in range(len(split_indices) - 1):
                        i_start = split_indices[s_idx]
                        i_end = split_indices[s_idx + 1]
                        if i_start >= i_end:
                            continue

                        stretch_coords = line[i_start : i_end + 1]
                        stretch_nodes = line_nodes[i_start : i_end + 1]

                        # Project all stretch vertices to local ENU
                        lats = np.array([pt[1] for pt in stretch_coords], dtype=np.float64)
                        lons = np.array([pt[0] for pt in stretch_coords], dtype=np.float64)
                        alts = np.zeros(len(lats), dtype=np.float64)
                        enu_pts = geodetic_to_enu(lats, lons, alts, ref_lat, ref_lon, ref_alt)[:, :2]

                        # Apply Douglas-Peucker simplification with tolerance simplification_tol_m
                        if len(enu_pts) <= 2 or simplification_tol_m <= 0.0:
                            retained = list(range(len(enu_pts)))
                        else:
                            retained = douglas_peucker_indices(enu_pts, simplification_tol_m)

                        for k in range(len(retained) - 1):
                            ia = retained[k]
                            ib = retained[k + 1]

                            p1_enu = enu_pts[ia]
                            p2_enu = enu_pts[ib]
                            diff = p2_enu - p1_enu
                            length = float(np.linalg.norm(diff))
                            if length < 0.5:
                                continue

                            bearing = float(np.degrees(np.arctan2(diff[0], diff[1]))) % 360.0
                            seg_id = f"{road_id_prefix}_{feat_idx}_{seg_counter:05d}"
                            seg_counter += 1

                            lon1, lat1 = stretch_coords[ia][0], stretch_coords[ia][1]
                            lon2, lat2 = stretch_coords[ib][0], stretch_coords[ib][1]
                            nid_start = stretch_nodes[ia]
                            nid_end = stretch_nodes[ib]

                            seg = RoadSegment(
                                segment_id=seg_id,
                                start_enu_m=p1_enu,
                                end_enu_m=p2_enu,
                                start_lat_lon=(lat1, lon1),
                                end_lat_lon=(lat2, lon2),
                                bearing_deg=bearing,
                                length_m=length,
                                road_type=road_type,
                                speed_limit_mps=speed_limit,
                                is_oneway=is_oneway,
                                start_node_id=nid_start,
                                end_node_id=nid_end,
                            )
                            network.add_segment(seg)

                            if not is_oneway:
                                rev_bearing = (bearing + 180.0) % 360.0
                                seg_rev = RoadSegment(
                                    segment_id=f"{seg_id}_rev",
                                    start_enu_m=p2_enu,
                                    end_enu_m=p1_enu,
                                    start_lat_lon=(lat2, lon2),
                                    end_lat_lon=(lat1, lon1),
                                    bearing_deg=rev_bearing,
                                    length_m=length,
                                    road_type=road_type,
                                    speed_limit_mps=speed_limit,
                                    is_oneway=is_oneway,
                                    start_node_id=nid_end,
                                    end_node_id=nid_start,
                                )
                                network.add_segment(seg_rev)
                else:
                    # Fallback when node IDs are absent: project line and simplify
                    lats = np.array([pt[1] for pt in line], dtype=np.float64)
                    lons = np.array([pt[0] for pt in line], dtype=np.float64)
                    alts = np.zeros(len(lats), dtype=np.float64)
                    enu_pts = geodetic_to_enu(lats, lons, alts, ref_lat, ref_lon, ref_alt)[:, :2]

                    if len(enu_pts) <= 2 or simplification_tol_m <= 0.0:
                        retained = list(range(len(enu_pts)))
                    else:
                        retained = douglas_peucker_indices(enu_pts, simplification_tol_m)

                    for k in range(len(retained) - 1):
                        ia = retained[k]
                        ib = retained[k + 1]

                        p1_enu = enu_pts[ia]
                        p2_enu = enu_pts[ib]
                        diff = p2_enu - p1_enu
                        length = float(np.linalg.norm(diff))
                        if length < 0.5:
                            continue

                        bearing = float(np.degrees(np.arctan2(diff[0], diff[1]))) % 360.0
                        if "segment_id" in props and len(retained) == 2:
                            seg_id = props["segment_id"]
                        else:
                            seg_id = f"{road_id_prefix}_{feat_idx}_{seg_counter:05d}"
                        seg_counter += 1

                        lon1, lat1 = line[ia][0], line[ia][1]
                        lon2, lat2 = line[ib][0], line[ib][1]

                        nid_start = f"synth_{feat_idx}_{line_idx}_{ia}"
                        nid_end = f"synth_{feat_idx}_{line_idx}_{ib}"

                        seg = RoadSegment(
                            segment_id=seg_id,
                            start_enu_m=p1_enu,
                            end_enu_m=p2_enu,
                            start_lat_lon=(lat1, lon1),
                            end_lat_lon=(lat2, lon2),
                            bearing_deg=bearing,
                            length_m=length,
                            road_type=road_type,
                            speed_limit_mps=speed_limit,
                            is_oneway=is_oneway,
                            start_node_id=nid_start,
                            end_node_id=nid_end,
                        )
                        network.add_segment(seg)

                        if not is_oneway:
                            rev_bearing = (bearing + 180.0) % 360.0
                            seg_rev = RoadSegment(
                                segment_id=f"{seg_id}_rev",
                                start_enu_m=p2_enu,
                                end_enu_m=p1_enu,
                                start_lat_lon=(lat2, lon2),
                                end_lat_lon=(lat1, lon1),
                                bearing_deg=rev_bearing,
                                length_m=length,
                                road_type=road_type,
                                speed_limit_mps=speed_limit,
                                is_oneway=is_oneway,
                                start_node_id=nid_end,
                                end_node_id=nid_start,
                            )
                            network.add_segment(seg_rev)

        return network


def build_road_network_from_trip(
    trip: Any,
    prefix: str = "sm_road",
    min_step_m: float = 10.0,
    cell_size_m: float = 100.0,
) -> Tuple[RoadNetwork, np.ndarray]:
    """
    Constructs a topological RoadNetwork from a recorded trip's valid GNSS trajectory.
    Decimates vertices to min_step_m to build clean polylines.
    """
    from sih.data.geo import geodetic_to_enu

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    raw_enu = [
        geodetic_to_enu(
            g.latitude_deg, g.longitude_deg, 0.0,
            trip.reference_lat_deg, trip.reference_lon_deg, 0.0
        )[:2]
        for g in valid_gnss
    ]
    raw_ll = [[g.latitude_deg, g.longitude_deg] for g in valid_gnss]

    filtered_enu = [raw_enu[0]]
    filtered_ll = [raw_ll[0]]
    for i in range(1, len(raw_enu)):
        dist = np.linalg.norm(raw_enu[i] - filtered_enu[-1])
        if dist >= min_step_m:
            filtered_enu.append(raw_enu[i])
            filtered_ll.append(raw_ll[i])

    pts_enu = np.array(filtered_enu)
    pts_ll = np.array(filtered_ll)

    net = RoadNetwork.from_polyline_coords(pts_enu, pts_ll, prefix, cell_size_m=cell_size_m)
    return net, pts_enu


# Convenient alias
build_road_network = build_road_network_from_trip


def build_road_network_from_osm(
    bbox_ll: Tuple[float, float, float, float],
    ref_lat: float,
    ref_lon: float,
    cache_dir: str = "data/maps/cache",
    cell_size_m: float = 100.0,
    client: Optional[Any] = None,
    simplification_tol_m: float = 2.0,
    max_tile_size_deg: float = 0.25,
) -> Tuple[RoadNetwork, np.ndarray]:
    """
    Constructs a topological RoadNetwork from OpenStreetMap highway vectors within a bounding box.
    Uses OSMOverpassClient to query Overpass API, and caches the result locally as GeoJSON in cache_dir
    so subsequent runs are completely offline. Converts geometries to ENU relative to (ref_lat, ref_lon).
    Supports tiling bounding boxes exceeding max_tile_size_deg and applies Douglas-Peucker simplification
    to preserve road curvature while bounding memory usage.
    """
    from sih.map.osm_client import OSMOverpassClient

    lat_min = float(min(bbox_ll[0], bbox_ll[2]))
    lat_max = float(max(bbox_ll[0], bbox_ll[2]))
    lon_min = float(min(bbox_ll[1], bbox_ll[3]))
    lon_max = float(max(bbox_ll[1], bbox_ll[3]))

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"osm_bbox_{lat_min:.5f}_{lon_min:.5f}_{lat_max:.5f}_{lon_max:.5f}.geojson",
    )

    geojson_dict = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                geojson_dict = json.load(f)
        except Exception as e:
            print(f"[OSM Cache Warning] Error reading {cache_file}: {e}. Re-fetching.")
            geojson_dict = None

    # Invalidate cache if node topology list is missing from properties
    if geojson_dict is not None and "features" in geojson_dict and len(geojson_dict["features"]) > 0:
        first_props = geojson_dict["features"][0].get("properties", {})
        if "nodes" not in first_props:
            print(f"[OSM Cache] Cache {os.path.basename(cache_file)} lacks node topology. Re-fetching from Overpass...")
            geojson_dict = None

    if geojson_dict is None:
        osm_client = client or OSMOverpassClient()
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        # Partition into grid tiles if span exceeds max_tile_size_deg or multi-tile fetch desired
        n_lat = max(1, int(round(lat_span / max_tile_size_deg + 0.499)))
        n_lon = max(1, int(round(lon_span / max_tile_size_deg + 0.499)))

        all_features = {}
        for i in range(n_lat):
            t_min_lat = lat_min + i * (lat_span / n_lat)
            t_max_lat = lat_min + (i + 1) * (lat_span / n_lat)
            for j in range(n_lon):
                t_min_lon = lon_min + j * (lon_span / n_lon)
                t_max_lon = lon_min + (j + 1) * (lon_span / n_lon)

                tile_name = f"osm_tile_{t_min_lat:.5f}_{t_min_lon:.5f}_{t_max_lat:.5f}_{t_max_lon:.5f}.geojson"
                tile_path = os.path.join(cache_dir, tile_name)

                tile_features = None
                if os.path.exists(tile_path):
                    try:
                        with open(tile_path, "r", encoding="utf-8") as f:
                            tile_data = json.load(f)
                        feats = tile_data.get("features", [])
                        if feats and "nodes" in feats[0].get("properties", {}):
                            tile_features = feats
                    except Exception:
                        tile_features = None

                if tile_features is None:
                    raw = osm_client.fetch_raw_osm_bbox(t_min_lat, t_min_lon, t_max_lat, t_max_lon)
                    if raw is not None:
                        parsed = osm_client.parse_osm_to_geojson(raw)
                        tile_features = parsed.get("features", [])
                        try:
                            with open(tile_path, "w", encoding="utf-8") as f:
                                json.dump(parsed, f)
                        except Exception as e:
                            print(f"[OSM Cache Warning] Could not save tile {tile_name}: {e}")
                    else:
                        print(f"[OSM Warning] Overpass query failed for tile ({t_min_lat:.4f}, {t_min_lon:.4f}, {t_max_lat:.4f}, {t_max_lon:.4f}).")
                        tile_features = []

                for feat in tile_features:
                    osm_id = feat.get("properties", {}).get("osm_id")
                    key = osm_id if osm_id is not None else id(feat)
                    if key not in all_features:
                        all_features[key] = feat

        geojson_dict = {"type": "FeatureCollection", "features": list(all_features.values())}
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(geojson_dict, f)
        except Exception as e:
            print(f"[OSM Cache Warning] Could not save cache to {cache_file}: {e}")

    net = RoadNetwork.from_geojson_dict(
        geojson_dict,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        cell_size_m=cell_size_m,
        road_id_prefix="osm_road",
        simplification_tol_m=simplification_tol_m,
    )

    # Attach metadata for coverage auditing
    features = geojson_dict.get("features", [])
    setattr(net, "num_ways", len(features))
    setattr(net, "_geojson_dict", geojson_dict)

    if net.segments:
        all_pts = [s.start_enu_m for s in net.segments]
        all_pts.append(net.segments[-1].end_enu_m)
        pts_enu = np.array(all_pts, dtype=np.float64)
    else:
        pts_enu = np.zeros((0, 2), dtype=np.float64)

    return net, pts_enu


def build_road_network_from_trip_masked(
    trip: Any,
    blackout_windows_ns: List[Tuple[int, int]],
    margin_s: float = 30.0,
    prefix: str = "masked_road",
    min_step_m: float = 10.0,
    cell_size_m: float = 100.0,
) -> Tuple[RoadNetwork, np.ndarray]:
    """
    Constructs a topological RoadNetwork from a recorded trip's valid GNSS trajectory,
    strictly excising all GNSS samples that fall within any blackout window expanded by
    margin_s on both sides.

    Segments NEVER bridge across an excluded outage gap; the polyline is broken at every
    gap into separate, disjoint contiguous segments.

    Parameters
    ----------
    trip : Any
        GenericTrip containing gnss_samples and reference lat/lon.
    blackout_windows_ns : list of tuple of (int, int)
        List of (bo_start_ns, bo_end_ns) simulated blackout intervals.
    margin_s : float
        Safety margin in seconds to expand around each blackout window.
    prefix : str
        Prefix for generated segment IDs.
    min_step_m : float
        Vertex decimation distance threshold.
    cell_size_m : float
        Spatial indexing grid cell size.

    Returns
    -------
    Tuple[RoadNetwork, np.ndarray]
        The constructed RoadNetwork and an (N, 2) ndarray of unmasked road vertex coordinates in ENU.
    """
    margin_ns = int(margin_s * 1e9)
    expanded_windows = [
        (int(w_start - margin_ns), int(w_end + margin_ns))
        for w_start, w_end in blackout_windows_ns
    ]

    def is_masked(ts_ns: int) -> bool:
        for start_ns, end_ns in expanded_windows:
            if start_ns <= ts_ns <= end_ns:
                return True
        return False

    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]

    # Partition unmasked GNSS samples into contiguous chunks separated by masked gaps
    chunks: List[List[Any]] = []
    current_chunk: List[Any] = []
    for g in valid_gnss:
        if not is_masked(g.timestamp_ns):
            current_chunk.append(g)
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    net = RoadNetwork(cell_size_m=cell_size_m)
    all_filtered_enu: List[np.ndarray] = []

    for chunk_idx, chunk in enumerate(chunks):
        if len(chunk) < 2:
            continue

        raw_enu = [
            geodetic_to_enu(
                g.latitude_deg, g.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )[:2]
            for g in chunk
        ]
        raw_ll = [[g.latitude_deg, g.longitude_deg] for g in chunk]

        filtered_enu = [raw_enu[0]]
        filtered_ll = [raw_ll[0]]
        for i in range(1, len(raw_enu)):
            dist = float(np.linalg.norm(raw_enu[i] - filtered_enu[-1]))
            if dist >= min_step_m:
                filtered_enu.append(raw_enu[i])
                filtered_ll.append(raw_ll[i])

        if len(filtered_enu) < 2:
            continue

        pts_chunk_enu = np.array(filtered_enu)
        pts_chunk_ll = filtered_ll
        all_filtered_enu.append(pts_chunk_enu)

        chunk_prefix = f"{prefix}_c{chunk_idx:02d}"
        sub_net = RoadNetwork.from_polyline_coords(
            pts_chunk_enu, pts_chunk_ll, chunk_prefix, cell_size_m=cell_size_m
        )
        net.merge(sub_net)

    if all_filtered_enu:
        pts_enu = np.vstack(all_filtered_enu)
    else:
        pts_enu = np.zeros((0, 2), dtype=np.float64)

    return net, pts_enu


def compute_road_network_coverage(
    road_net: RoadNetwork,
    trip: Any,
    threshold_m: float = 25.0,
) -> Tuple[int, float]:
    """
    Computes the spatial coverage of a RoadNetwork against a trip's valid ground-truth GNSS fixes.

    Parameters
    ----------
    road_net : RoadNetwork
        The road network to evaluate.
    trip : Any
        GenericTrip containing gnss_samples and reference coordinates.
    threshold_m : float
        Maximum distance threshold in meters for a point to be considered covered (default 25.0m).

    Returns
    -------
    Tuple[int, float]
        (num_ways, fraction_covered_within_threshold).
    """
    valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
    if not valid_gnss or not road_net.segments:
        return 0, 0.0

    num_ways = getattr(road_net, "num_ways", len(road_net.segments))
    within_count = 0

    for g in valid_gnss:
        p_enu = geodetic_to_enu(
            g.latitude_deg, g.longitude_deg, 0.0,
            trip.reference_lat_deg, trip.reference_lon_deg, 0.0
        )[:2]
        cands = road_net.find_candidates(p_enu, radius_m=threshold_m + 15.0)
        if cands:
            min_d = min(s.project_point(p_enu)[1] for s in cands)
            if min_d <= threshold_m:
                within_count += 1

    coverage_frac = float(within_count / len(valid_gnss))
    return num_ways, coverage_frac


def audit_road_network_topology(
    road_net: RoadNetwork,
    geojson_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Audits the topological structure of an OSM road network, reporting:
    - Before segment count (when split naively at every vertex)
    - After segment count (when split at shared junction nodes)
    - Mean successors per segment under node identity (seg_a.end_node_id == seg_b.start_node_id)
    - Mean successors per segment under 8m endpoint distance heuristic
    """
    g_dict = geojson_dict or getattr(road_net, "_geojson_dict", {})
    features = g_dict.get("features", [])
    before_count = sum(
        max(0, len(f.get("geometry", {}).get("coordinates", [])) - 1)
        for f in features
    )
    if before_count == 0:
        before_count = len(road_net.segments)

    after_count = len(road_net.segments)

    # Node identity successor statistics
    start_node_map: Dict[Any, List[RoadSegment]] = {}
    for s in road_net.segments:
        if s.start_node_id is not None:
            if s.start_node_id not in start_node_map:
                start_node_map[s.start_node_id] = []
            start_node_map[s.start_node_id].append(s)

    succ_counts_node_id = []
    for s1 in road_net.segments:
        if s1.end_node_id is not None and s1.end_node_id in start_node_map:
            cnt = sum(1 for s2 in start_node_map[s1.end_node_id] if s1.segment_id != s2.segment_id)
            succ_counts_node_id.append(cnt)
        else:
            succ_counts_node_id.append(0)

    mean_succs_node_id = float(np.mean(succ_counts_node_id)) if succ_counts_node_id else 0.0

    # 8.0m distance heuristic successor statistics on the segmented network
    cell_size = 15.0
    start_grid: Dict[Tuple[int, int], List[RoadSegment]] = {}
    for s in road_net.segments:
        cx = int(np.floor(s.start_enu_m[0] / cell_size))
        cy = int(np.floor(s.start_enu_m[1] / cell_size))
        key = (cx, cy)
        if key not in start_grid:
            start_grid[key] = []
        start_grid[key].append(s)

    succ_counts_dist = []
    for s1 in road_net.segments:
        ecx = int(np.floor(s1.end_enu_m[0] / cell_size))
        ecy = int(np.floor(s1.end_enu_m[1] / cell_size))
        e_pt = s1.end_enu_m
        cnt = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand_list = start_grid.get((ecx + dx, ecy + dy), None)
                if cand_list is not None:
                    for s2 in cand_list:
                        if s1.segment_id != s2.segment_id:
                            d2 = (e_pt[0] - s2.start_enu_m[0])**2 + (e_pt[1] - s2.start_enu_m[1])**2
                            if d2 <= 64.0:  # 8.0^2
                                cnt += 1
        succ_counts_dist.append(cnt)

    mean_succs_dist = float(np.mean(succ_counts_dist)) if succ_counts_dist else 0.0

    return {
        "before_segment_count": before_count,
        "after_segment_count": after_count,
        "mean_succs_dist_8m": mean_succs_dist,
        "mean_succs_node_id": mean_succs_node_id,
    }


def load_trip_road_network(
    trip: Any,
    map_source: str = "osm",
    blackout_windows: Optional[List[Tuple[int, int]]] = None,
    cache_dir: str = "data/maps/cache",
    prefix: Optional[str] = None,
    bbox_pad_m: float = 500.0,
) -> Tuple[RoadNetwork, np.ndarray]:
    """
    Unified entry point for loading road networks for a trip across all map sources.
    Strictly encapsulates all road network construction, bounding box calculation,
    and map source selection logic behind sih/map/ (Rule 13).

    Supported map sources:
    - 'osm': Leak-free external OpenStreetMap network via Overpass API / GeoJSON disk cache.
    - 'masked': Leak-free GNSS polyline with blackout windows (+/- 30s margin) excised into disjoint chunks.
    - 'trip': LEAKED reference baseline built from full trip GNSS including blackout windows.

    Parameters
    ----------
    trip : Any
        GenericTrip instance with IMU and GNSS samples.
    map_source : str
        One of 'osm', 'masked', or 'trip'. Default is 'osm'.
    blackout_windows : list of (start_ns, end_ns), optional
        Required when map_source == 'masked'.
    cache_dir : str
        Directory to store and read cached OSM GeoJSON files.
    prefix : str, optional
        Prefix for segment IDs.

    Returns
    -------
    Tuple[RoadNetwork, np.ndarray]
        (road_network, road_points_enu)
    """
    tid = getattr(trip, "trip_id", "trip")
    pfx = prefix or f"{tid.lower()}_road"

    if map_source == "trip":
        print("\n" + "!" * 80)
        print("  [CRITICAL WARNING: LEAKED GROUND-TRUTH MAP SOURCE]")
        print(f"  Map source 'trip' generates {tid} road polylines directly from the vehicle's")
        print("  own GNSS trajectory across the ENTIRE trip, INCLUDING points within")
        print("  synthetic blackout windows. This constitutes a severe DATA LEAKAGE.")
        print("  All map-matched numbers from this source are FOR REFERENCE ONLY.")
        print("!" * 80 + "\n")
        rnet, rpts = build_road_network_from_trip(trip, prefix=pfx)
        print(f"  - Leaked Road network for {tid}: {len(rnet.segments)} segments, {len(rpts)} nodes")
        return rnet, rpts

    elif map_source == "masked":
        pfx_m = prefix or f"{tid.lower()}_masked"
        rnet, rpts = build_road_network_from_trip_masked(
            trip, blackout_windows_ns=blackout_windows or [], margin_s=30.0, prefix=pfx_m
        )
        return rnet, rpts

    elif map_source == "osm":
        # OSM Route Corridor Pre-Fetch Architecture:
        # Analogous to production mobile navigation applications (e.g. Google Maps, Mapbox)
        # where the planned route corridor is downloaded in advance of travel. The bounding
        # box encompasses public road geometry from OpenStreetMap across the trip extent.
        # It contains only static public cartographic infrastructure (ways, nodes, tags)
        # and contains zero GNSS trajectory information, vehicle velocities, or future states.
        valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
        if not valid_gnss:
            ref_lat = trip.reference_lat_deg
            ref_lon = trip.reference_lon_deg
            pad_lat = bbox_pad_m / 111139.0
            pad_lon = bbox_pad_m / (111139.0 * max(0.01, float(np.cos(np.radians(ref_lat)))))
            bbox = (ref_lat - pad_lat, ref_lon - pad_lon, ref_lat + pad_lat, ref_lon + pad_lon)
        else:
            valid_lats = [g.latitude_deg for g in valid_gnss]
            valid_lons = [g.longitude_deg for g in valid_gnss]
            ref_lat = trip.reference_lat_deg
            pad_lat = bbox_pad_m / 111139.0
            pad_lon = bbox_pad_m / (111139.0 * max(0.01, float(np.cos(np.radians(ref_lat)))))
            bbox = (
                min(valid_lats) - pad_lat,
                min(valid_lons) - pad_lon,
                max(valid_lats) + pad_lat,
                max(valid_lons) + pad_lon,
            )

        rnet, rpts = build_road_network_from_osm(
            bbox, trip.reference_lat_deg, trip.reference_lon_deg, cache_dir=cache_dir
        )
        num_ways, cov_frac = compute_road_network_coverage(rnet, trip, threshold_m=25.0)
        print(f"  - OSM Road network for {tid}: {len(rnet.segments)} segments, {len(rpts)} nodes, {num_ways} ways")
        print(f"  - OSM Route Coverage (<= 25m): {cov_frac * 100.0:.1f}%")
        if cov_frac < 0.80:
            print(f"  [WARNING] OSM route coverage {cov_frac * 100.0:.1f}% is below 80.0% threshold for {tid}!")
        return rnet, rpts

    else:
        raise ValueError(f"Unknown map_source '{map_source}'. Expected 'osm', 'masked', or 'trip'.")
