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
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
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
        timeout_s: float = 12.0,
        max_retries: int = 2,
    ) -> None:
        self.endpoint = endpoint_url or self.OVERPASS_ENDPOINTS[0]
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def is_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        """Live client does not maintain an internal cache; always live query."""
        return False

    def build_query(self, lat: float, lon: float, radius_m: float) -> str:
        """Constructs a compact Overpass QL query string."""
        # Query highways within radius
        query = f"""[out:json][timeout:{int(self.timeout_s)}];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|link"](around:{int(radius_m)},{lat},{lon});
);
out body geom;"""
        return query

    def fetch_raw_osm(self, lat: float, lon: float, radius_m: float) -> Optional[Dict[str, Any]]:
        """
        Queries Overpass API with retry fallback across endpoints.
        """
        query = self.build_query(lat, lon, radius_m)
        encoded_data = urllib.parse.urlencode({"data": query}).encode("utf-8")

        endpoints_to_try = [self.endpoint] + [ep for ep in self.OVERPASS_ENDPOINTS if ep != self.endpoint]

        for ep in endpoints_to_try[: self.max_retries + 1]:
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
            except Exception as e:
                # Fallback to next mirror endpoint
                continue

        return None

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
    ) -> RoadNetwork:
        """
        Queries OSM Overpass and returns a populated RoadNetwork.
        """
        osm_json = self.fetch_raw_osm(lat, lon, radius_m)
        if osm_json is None:
            # Graceful fallback: return empty network
            return RoadNetwork(cell_size_m=100.0)

        geojson_dict = self.parse_osm_to_geojson(osm_json)
        return RoadNetwork.from_geojson_dict(
            geojson_dict,
            ref_lat=lat,
            ref_lon=lon,
            road_id_prefix="osm_way",
        )
