"""
sih.map.osm_client
------------------
OpenStreetMap Overpass API client for live road network vector extraction.
Fetches road centerlines, surface classifications, speed limits, and one-way constraints.
"""

from __future__ import annotations
import urllib.request
import urllib.parse
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from sih.map.network import RoadNetwork, RoadSegment
from sih.map.provider import IRoadNetworkProvider


class OSMOverpassClient(IRoadNetworkProvider):
    """
    Live OpenStreetMap vector road network client querying the Overpass API.
    Converts OSM routable ways directly into an indexed RoadNetwork.
    """

    OVERPASS_ENDPOINTS = [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    ]

    HIGHWAY_SPEED_LIMITS_MPS = {
        "motorway": 33.3,        # 120 km/h
        "motorway_link": 22.2,   # 80 km/h
        "trunk": 27.8,           # 100 km/h
        "trunk_link": 19.4,      # 70 km/h
        "primary": 22.2,         # 80 km/h
        "primary_link": 16.7,    # 60 km/h
        "secondary": 16.7,       # 60 km/h
        "secondary_link": 13.9,  # 50 km/h
        "tertiary": 13.9,        # 50 km/h
        "residential": 11.1,     # 40 km/h
        "unclassified": 11.1,    # 40 km/h
        "service": 8.3,          # 30 km/h
    }

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.endpoint = endpoint_url or self.OVERPASS_ENDPOINTS[0]
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def is_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        """Live client does not maintain an internal cache; always live query."""
        return False

    def build_query(self, lat: float, lon: float, radius_m: float) -> str:
        """Constructs a compact Overpass QL query string for circular radius."""
        query = f"""[out:json][timeout:{int(self.timeout_s)}];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|link"](around:{int(radius_m)},{lat},{lon});
);
out body geom;"""
        return query

    def build_bbox_query(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> str:
        """Constructs a compact Overpass QL query string for a bounding box."""
        query = f"""[out:json][timeout:{int(self.timeout_s)}];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|link"]({min_lat:.6f},{min_lon:.6f},{max_lat:.6f},{max_lon:.6f});
);
out body geom;"""
        return query

    def _execute_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Queries Overpass API with retry fallback across endpoints."""
        encoded_data = urllib.parse.urlencode({"data": query}).encode("utf-8")
        endpoints_to_try = [self.endpoint] + [ep for ep in self.OVERPASS_ENDPOINTS if ep != self.endpoint]

        for attempt in range(self.max_retries + 1):
            for ep in endpoints_to_try:
                req = urllib.request.Request(
                    ep,
                    data=encoded_data,
                    headers={"User-Agent": "SIH-Smart-Dead-Reckoning/1.0 (India-Transit-Research)"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                        if response.status == 200:
                            raw_text = response.read().decode("utf-8")
                            return json.loads(raw_text)
                except urllib.error.HTTPError as e:
                    if e.code in (429, 504):
                        backoff_s = 2.0 * (attempt + 1)
                        print(f"[OSM Overpass] HTTP {e.code} on {ep}, backing off {backoff_s:.1f}s...")
                        time.sleep(backoff_s)
                    continue
                except Exception:
                    # Fallback to next mirror endpoint
                    continue

        return None

    def fetch_raw_osm(self, lat: float, lon: float, radius_m: float) -> Optional[Dict[str, Any]]:
        """
        Queries Overpass API for radius with retry fallback across endpoints.
        """
        query = self.build_query(lat, lon, radius_m)
        return self._execute_query(query)

    def fetch_raw_osm_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Queries Overpass API for bounding box with retry fallback across endpoints.
        """
        query = self.build_bbox_query(min_lat, min_lon, max_lat, max_lon)
        return self._execute_query(query)

    def parse_osm_to_geojson(self, osm_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts Overpass JSON format with geometry into standard GeoJSON FeatureCollection.
        """
        elements = osm_json.get("elements", [])
        features = []

        for el in elements:
            if el.get("type") != "way":
                continue

            geom = el.get("geometry", [])
            if len(geom) < 2:
                continue

            coords = [[pt["lon"], pt["lat"]] for pt in geom]
            tags = el.get("tags", {})
            hw_type = tags.get("highway", "primary")

            speed_limit = self.HIGHWAY_SPEED_LIMITS_MPS.get(hw_type, 16.7)
            if "maxspeed" in tags:
                try:
                    speed_limit = float(tags["maxspeed"].split()[0]) / 3.6
                except (ValueError, IndexError):
                    pass

            is_oneway = tags.get("oneway") in ("yes", "1", "true")

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "osm_id": el.get("id"),
                    "nodes": el.get("nodes", []),
                    "name": tags.get("name", "Unnamed Road"),
                    "road_type": hw_type,
                    "speed_limit_mps": speed_limit,
                    "is_oneway": is_oneway,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def get_corridor_network(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        bearing_deg: Optional[float] = None,
        cache_dir: str = "data/maps/cache",
        simplification_tol_m: float = 2.0,
    ) -> RoadNetwork:
        """
        Queries OSM Overpass and returns a populated RoadNetwork for the vehicle corridor.
        Uses tile-based caching and Douglas-Peucker simplification (simplification_tol_m=2.0).
        Gracefully degrades to empty RoadNetwork on complete network / cache failure.
        """
        from sih.map.network import build_road_network_from_osm
        d_lat = radius_m / 111139.0
        d_lon = radius_m / (111139.0 * max(0.01, float(np.cos(np.radians(lat)))))
        bbox = (lat - d_lat, lon - d_lon, lat + d_lat, lon + d_lon)
        rnet, _ = build_road_network_from_osm(
            bbox_ll=bbox,
            ref_lat=lat,
            ref_lon=lon,
            cache_dir=cache_dir,
            client=self,
            simplification_tol_m=simplification_tol_m,
        )
        return rnet
