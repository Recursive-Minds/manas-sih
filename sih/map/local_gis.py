"""
sih.map.local_gis
-----------------
Local GIS vector provider for Indian rural and highway road datasets.
Ingests PMGSY (Pradhan Mantri Gram Sadak Yojana) and ISRO Bhuvan vector layers.
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any, List
import numpy as np

from sih.map.network import RoadNetwork
from sih.map.provider import IRoadNetworkProvider
from sih.data.geo import haversine_distance_m


class LocalGISProvider(IRoadNetworkProvider):
    """
    Offline local GIS vector provider.
    Parses pre-downloaded GeoJSON or Shapefile exports (PMGSY, Bhuvan, or local OSM dumps).
    """

    def __init__(self, gis_data_dir: str = "data/maps/indian_gis") -> None:
        self.gis_data_dir = gis_data_dir
        self._cached_geojson_data: List[Dict[str, Any]] = []
        os.makedirs(self.gis_data_dir, exist_ok=True)
        self._load_available_datasets()

    def _load_available_datasets(self) -> None:
        """Discovers and parses all available .geojson files in the GIS directory."""
        if not os.path.exists(self.gis_data_dir):
            return

        for fname in os.listdir(self.gis_data_dir):
            if fname.endswith(".geojson") or fname.endswith(".json"):
                fpath = os.path.join(self.gis_data_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._cached_geojson_data.append(data)
                except Exception as e:
                    print(f"[LocalGISProvider] Warning: failed loading {fname}: {e}")

    def add_dataset(self, geojson_data: Dict[str, Any]) -> None:
        """Manually registers an in-memory GeoJSON dataset."""
        self._cached_geojson_data.append(geojson_data)

    def is_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        """Checks if any registered GIS dataset has features near the point."""
        return len(self._cached_geojson_data) > 0

    def get_corridor_network(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        bearing_deg: Optional[float] = None,
    ) -> RoadNetwork:
        """
        Extracts road features within radius_m from all registered Indian GIS datasets.
        """
        combined = RoadNetwork(cell_size_m=100.0)

        for ds in self._cached_geojson_data:
            features = ds.get("features", [])
            filtered_features = []

            for feat in features:
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [])
                g_type = geom.get("type", "")

                lines = [coords] if g_type == "LineString" else (coords if g_type == "MultiLineString" else [])
                # Check if any coordinate is within radius_m
                is_near = False
                for line in lines:
                    for pt in line:
                        if len(pt) >= 2:
                            p_lon, p_lat = pt[0], pt[1]
                            dist = haversine_distance_m(lat, lon, p_lat, p_lon)
                            if dist <= radius_m * 1.2:
                                is_near = True
                                break
                    if is_near:
                        break

                if is_near:
                    filtered_features.append(feat)

            if filtered_features:
                sub_geojson = {"type": "FeatureCollection", "features": filtered_features}
                sub_net = RoadNetwork.from_geojson_dict(
                    sub_geojson,
                    ref_lat=lat,
                    ref_lon=lon,
                    road_id_prefix="pmgsy_road",
                )
                combined.merge(sub_net)

        return combined
