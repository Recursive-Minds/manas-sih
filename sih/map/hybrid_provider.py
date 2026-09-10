"""
sih.map.hybrid_provider
-----------------------
Hybrid multi-tier road network provider for Indian transit environments.
Orchestrates L2 Disk Cache -> Live OSM Overpass -> PMGSY/Bhuvan Local GIS -> Graceful Fallback.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any

from sih.map.network import RoadNetwork
from sih.map.provider import IRoadNetworkProvider
from sih.map.cache import SpatialDiskCache
from sih.map.osm_client import OSMOverpassClient
from sih.map.local_gis import LocalGISProvider


class HybridIndiaMapProvider(IRoadNetworkProvider):
    """
    Production composite provider for India:
    1. Tier 1: Local Disk & RAM Cache (instant O(1), offline).
    2. Tier 2: Live OpenStreetMap Overpass (highways, expressways, city grids).
    3. Tier 3: Local PMGSY / ISRO Bhuvan GIS (rural habitations & panchayat tracks).
    4. Tier 4: Graceful Degradation (unmapped terrain, farmland, private tracks).
    """

    def __init__(
        self,
        cache_dir: str = "data/maps/cache",
        gis_data_dir: str = "data/maps/indian_gis",
        enable_live_osm: bool = True,
        osm_client: Optional[OSMOverpassClient] = None,
        disk_cache: Optional[SpatialDiskCache] = None,
        local_gis: Optional[LocalGISProvider] = None,
    ) -> None:
        self.enable_live_osm = enable_live_osm
        self.cache = disk_cache or SpatialDiskCache(cache_dir=cache_dir)
        self.osm = osm_client or OSMOverpassClient()
        self.gis = local_gis or LocalGISProvider(gis_data_dir=gis_data_dir)

    def is_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        """Checks if the area is already available in the local disk/RAM cache."""
        return self.cache.is_area_cached(lat, lon, radius_m)

    def get_corridor_network(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        bearing_deg: Optional[float] = None,
    ) -> RoadNetwork:
        """
        Multi-tier resolution for road network queries.
        """
        # Tier 1: Check Local Disk & Memory Cache
        if self.cache.is_area_cached(lat, lon, radius_m):
            cached_net = self.cache.get_combined_network_for_radius(
                lat=lat,
                lon=lon,
                radius_m=radius_m,
                ref_lat=lat,
                ref_lon=lon,
            )
            if len(cached_net.segments) > 0:
                return cached_net

        # Tier 2: Query Live OpenStreetMap Overpass if enabled
        if self.enable_live_osm:
            osm_net = self.osm.get_corridor_network(
                lat=lat,
                lon=lon,
                radius_m=radius_m,
                bearing_deg=bearing_deg,
            )
            if len(osm_net.segments) > 0:
                # Cache fetched network partitioned by spatial tiles for future offline use
                covered_keys = self.cache.get_tile_keys_for_radius(lat, lon, radius_m)
                self.cache.save_network_by_tiles(osm_net, expected_keys=covered_keys)
                return osm_net

        # Tier 3: Query Local PMGSY / ISRO Bhuvan GIS Data
        gis_net = self.gis.get_corridor_network(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            bearing_deg=bearing_deg,
        )
        if len(gis_net.segments) > 0:
            covered_keys = self.cache.get_tile_keys_for_radius(lat, lon, radius_m)
            self.cache.save_network_by_tiles(gis_net, expected_keys=covered_keys)
            return gis_net

        # Tier 4: Graceful Degradation (Unmapped rural tracks, farmland)
        # Returns empty network; MapMatcher will detect confidence < 0.25 and disable snapping
        return RoadNetwork(cell_size_m=100.0)
