"""
sih.map.provider
----------------
Abstract interface and contracts for spatial road network providers.
Decouples map matching from data source (OSM Overpass, PMGSY, ISRO Bhuvan, local cache).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from sih.map.network import RoadNetwork


@dataclass(slots=True, frozen=True)
class RoadNetworkMetadata:
    """Metadata describing the origin, coverage, and bounds of a road network."""
    source_name: str                     # e.g., 'osm_overpass', 'pmgsy_geojson', 'disk_cache'
    center_lat: float
    center_lon: float
    radius_m: float
    total_segments: int
    total_length_km: float
    timestamp_epoch_s: float
    bounding_box: Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    extra_attributes: Dict[str, Any] = field(default_factory=dict)


class IRoadNetworkProvider(ABC):
    """
    Abstract interface for spatial road network providers.
    Enables plug-and-play switching between live internet APIs, offline shapefiles, and local caches.
    """

    @abstractmethod
    def get_corridor_network(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        bearing_deg: Optional[float] = None,
    ) -> RoadNetwork:
        """
        Retrieves a connected RoadNetwork around the given coordinates.

        Parameters
        ----------
        lat : float
            Current vehicle latitude in degrees.
        lon : float
            Current vehicle longitude in degrees.
        radius_m : float
            Query radius in meters.
        bearing_deg : Optional[float]
            Optional vehicle heading in degrees (enables forward wedge filtering).

        Returns
        -------
        RoadNetwork
            Spatial hash grid road network ready for O(1) candidate lookup.
        """
        pass

    @abstractmethod
    def is_cached(self, lat: float, lon: float, radius_m: float) -> bool:
        """Checks whether the requested area is already available in local cache."""
        pass
