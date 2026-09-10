"""
Map matching module for SIH IDR pipeline.
"""

from sih.map.network import RoadSegment, RoadNetwork
from sih.map.matcher import HMMMapMatcher
from sih.map.governor import RoadKinematicsGovernor, DualRateRoadGovernor, update_map_measurement
from sih.map.provider import IRoadNetworkProvider, RoadNetworkMetadata
from sih.map.cache import SpatialDiskCache
from sih.map.osm_client import OSMOverpassClient
from sih.map.local_gis import LocalGISProvider
from sih.map.hybrid_provider import HybridIndiaMapProvider
from sih.map.corridor_manager import PredictiveCorridorManager, compute_lookahead_radius

__all__ = [
    "RoadSegment",
    "RoadNetwork",
    "HMMMapMatcher",
    "RoadKinematicsGovernor",
    "DualRateRoadGovernor",
    "update_map_measurement",
    "IRoadNetworkProvider",
    "RoadNetworkMetadata",
    "SpatialDiskCache",
    "OSMOverpassClient",
    "LocalGISProvider",
    "HybridIndiaMapProvider",
    "PredictiveCorridorManager",
    "compute_lookahead_radius",
]

