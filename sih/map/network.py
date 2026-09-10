"""
Road network geometry and spatial representation for Phase 4 Map Matching.

Stores road segments in both WGS-84 (Lat/Lon) and Local Tangent Plane (ENU).
Provides fast spatial indexing for candidate segment retrieval within search radius.
Supports graceful degradation for unmapped rural tracks and farmland.
"""

from __future__ import annotations
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
    ) -> RoadNetwork:
        """
        Constructs a RoadNetwork from a GeoJSON FeatureCollection (OSM, PMGSY, or Bhuvan).
        """
        network = cls(cell_size_m=cell_size_m)
        features = data.get("features", [])

        seg_counter = 0
        for feat_idx, feat in enumerate(features):
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            g_type = geom.get("type", "")
            raw_coords = geom.get("coordinates", [])

            lines = []
            if g_type == "LineString":
                lines = [raw_coords]
            elif g_type == "MultiLineString":
                lines = raw_coords

            road_type = props.get("road_type", props.get("highway", "primary"))
            speed_limit = float(props.get("speed_limit_mps", 25.0))
            is_oneway = bool(props.get("is_oneway", False))

            for line in lines:
                if len(line) < 2:
                    continue

                for i in range(len(line) - 1):
                    lon1, lat1 = line[i][0], line[i][1]
                    lon2, lat2 = line[i + 1][0], line[i + 1][1]

                    p1_enu = geodetic_to_enu(lat1, lon1, 0.0, ref_lat, ref_lon, ref_alt)[:2]
                    p2_enu = geodetic_to_enu(lat2, lon2, 0.0, ref_lat, ref_lon, ref_alt)[:2]

                    diff = p2_enu - p1_enu
                    length = float(np.linalg.norm(diff))
                    if length < 0.5:
                        continue

                    bearing = float(np.degrees(np.arctan2(diff[0], diff[1]))) % 360.0
                    custom_id = props.get("segment_id")
                    if custom_id and len(line) == 2:
                        seg_id = custom_id
                    else:
                        seg_id = f"{road_id_prefix}_{feat_idx}_{seg_counter:04d}"
                    seg_counter += 1

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
                    )
                    network.add_segment(seg)

        return network
