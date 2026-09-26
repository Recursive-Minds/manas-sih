"""
sih.map.cache
-------------
Persistent spatial disk cache and LRU memory cache for road network vector tiles.
Stores spatial tiles as compact GeoJSON files on disk, indexed by geographic grid keys.
"""

from __future__ import annotations
import os
import json
from collections import OrderedDict
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from sih.map.network import RoadNetwork


class SpatialDiskCache:
    """
    Spatial disk and memory cache for road network vector tiles.
    Partitions the geographic plane into grid tiles (default 0.05° ~ 5.5 km).
    """

    def __init__(
        self,
        cache_dir: str = "data/maps/cache",
        tile_size_deg: float = 0.05,
        max_lru_memory_tiles: int = 16,
    ) -> None:
        self.cache_dir = cache_dir
        self.tile_size_deg = tile_size_deg
        self.max_lru_tiles = max_lru_memory_tiles
        self._memory_cache: OrderedDict[str, RoadNetwork] = OrderedDict()

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass

    def compute_tile_key(self, lat: float, lon: float) -> str:
        """Computes the deterministic string key for a geographic point."""
        lat_idx = int(np.floor(lat / self.tile_size_deg))
        lon_idx = int(np.floor(lon / self.tile_size_deg))
        return f"tile_{lat_idx}_{lon_idx}"

    def get_tile_keys_for_bounds(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> List[str]:
        """Returns all tile keys intersecting the given bounding box."""
        lat_min_idx = int(np.floor(min_lat / self.tile_size_deg))
        lat_max_idx = int(np.floor(max_lat / self.tile_size_deg))
        lon_min_idx = int(np.floor(min_lon / self.tile_size_deg))
        lon_max_idx = int(np.floor(max_lon / self.tile_size_deg))

        keys = []
        for la in range(lat_min_idx, lat_max_idx + 1):
            for lo in range(lon_min_idx, lon_max_idx + 1):
                keys.append(f"tile_{la}_{lo}")
        return keys

    def get_tile_keys_for_radius(self, lat: float, lon: float, radius_m: float) -> List[str]:
        """Calculates all tile keys covering a circular radius around a point."""
        # 1 deg latitude ~ 111,000 meters
        d_lat = radius_m / 111000.0
        # 1 deg longitude ~ 111,000 * cos(lat) meters
        d_lon = radius_m / max(1000.0, 111000.0 * np.cos(np.radians(lat)))

        return self.get_tile_keys_for_bounds(
            min_lat=lat - d_lat,
            min_lon=lon - d_lon,
            max_lat=lat + d_lat,
            max_lon=lon + d_lon,
        )

    def _get_tile_path(self, tile_key: str) -> str:
        return os.path.join(self.cache_dir, f"{tile_key}.geojson")

    def is_tile_cached_on_disk(self, tile_key: str) -> bool:
        """Returns True if the tile file exists on disk."""
        return os.path.exists(self._get_tile_path(tile_key))

    def is_area_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        """Returns True if all tiles covering the requested radius exist on disk."""
        keys = self.get_tile_keys_for_radius(lat, lon, radius_m)
        return all(self.is_tile_cached_on_disk(k) for k in keys)

    def save_tile(self, tile_key: str, network: RoadNetwork) -> str:
        """
        Saves a RoadNetwork to disk under the specified tile key.
        Updates the in-memory LRU cache.
        """
        geojson_dict = network.to_geojson_dict()
        file_path = self._get_tile_path(tile_key)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f)

        # Update in-memory cache
        self._memory_cache[tile_key] = network
        self._memory_cache.move_to_end(tile_key)
        self._evict_lru()

        return file_path

    def save_network_by_tiles(
        self,
        network: RoadNetwork,
        expected_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Partitions a multi-tile RoadNetwork into individual spatial grid tiles and saves each to disk.
        For tiles within expected_keys with zero roads, writes an empty tile to record negative cache.
        """
        tile_segments: Dict[str, List[Any]] = {}
        if expected_keys is not None:
            for k in expected_keys:
                tile_segments[k] = []

        for seg in network.segments:
            k = self.compute_tile_key(seg.start_lat_lon[0], seg.start_lat_lon[1])
            if k not in tile_segments:
                tile_segments[k] = []
            tile_segments[k].append(seg)

        saved_keys = []
        for k, segs in tile_segments.items():
            sub_net = RoadNetwork(cell_size_m=network.cell_size_m)
            for s in segs:
                sub_net.add_segment(s)
            self.save_tile(k, sub_net)
            saved_keys.append(k)

        return saved_keys

    def load_tile(
        self,
        tile_key: str,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float = 0.0,
    ) -> Optional[RoadNetwork]:
        """
        Loads a RoadNetwork for a tile key, checking RAM cache first, then disk.
        """
        # 1. Check in-memory cache
        if tile_key in self._memory_cache:
            self._memory_cache.move_to_end(tile_key)
            return self._memory_cache[tile_key]

        # 2. Check disk cache
        file_path = self._get_tile_path(tile_key)
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            network = RoadNetwork.from_geojson_dict(
                data,
                ref_lat=ref_lat,
                ref_lon=ref_lon,
                ref_alt=ref_alt,
                road_id_prefix=tile_key,
            )
            self._memory_cache[tile_key] = network
            self._memory_cache.move_to_end(tile_key)
            self._evict_lru()
            return network
        except Exception as e:
            print(f"[SpatialDiskCache] Error loading {file_path}: {e}")
            return None

    def get_combined_network_for_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float = 0.0,
    ) -> RoadNetwork:
        """
        Gathers all cached tiles covering the radius and merges them into a single RoadNetwork.
        """
        keys = self.get_tile_keys_for_radius(lat, lon, radius_m)
        combined = RoadNetwork(cell_size_m=100.0)

        for k in keys:
            tile_net = self.load_tile(k, ref_lat=ref_lat, ref_lon=ref_lon, ref_alt=ref_alt)
            if tile_net is not None:
                combined.merge(tile_net)

        return combined

    def _evict_lru(self) -> None:
        """Enforces maximum in-memory tile capacity."""
        while len(self._memory_cache) > self.max_lru_tiles:
            self._memory_cache.popitem(last=False)

    def clear_memory(self) -> None:
        """Clears in-memory LRU cache."""
        self._memory_cache.clear()

    def get_cache_size_bytes(self) -> int:
        """Returns total bytes stored in the disk cache directory."""
        total = 0
        if os.path.exists(self.cache_dir):
            for root, _, files in os.walk(self.cache_dir):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        return total
