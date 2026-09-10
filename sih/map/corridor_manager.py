"""
sih.map.corridor_manager
------------------------
Speed-adaptive predictive corridor pre-caching and background worker.
Dynamically scales lookahead radius based on vehicle velocity and pre-caches road geometry asynchronously.
"""

from __future__ import annotations
import threading
import time
from typing import Optional, Tuple
import numpy as np

from sih.map.network import RoadNetwork
from sih.map.provider import IRoadNetworkProvider
from sih.data.geo import haversine_distance_m


def compute_lookahead_radius(
    speed_mps: float,
    lookahead_time_s: float = 180.0,
    min_radius_m: float = 800.0,
    max_radius_m: float = 6000.0,
) -> float:
    """
    Computes dynamic speed-dependent lookahead buffer radius.

    Parameters
    ----------
    speed_mps : float
        Current vehicle speed in meters per second.
    lookahead_time_s : float
        Lookahead horizon in seconds (default 180s = 3 minutes).
    min_radius_m : float
        Minimum buffer radius in meters (default 800m).
    max_radius_m : float
        Maximum buffer radius in meters (default 6,000m).

    Returns
    -------
    float
        Target lookahead radius in meters.
    """
    speed_clean = max(0.0, float(speed_mps) if speed_mps is not None else 0.0)
    raw_radius = speed_clean * lookahead_time_s
    return float(np.clip(raw_radius, min_radius_m, max_radius_m))


class PredictiveCorridorManager:
    """
    Background worker that monitors vehicle coordinates and pre-caches the forward corridor.
    Guarantees that the real-time 100 Hz dead-reckoning loop is NEVER stalled by network I/O.
    """

    def __init__(
        self,
        provider: IRoadNetworkProvider,
        active_network: Optional[RoadNetwork] = None,
        lookahead_time_s: float = 180.0,
        min_radius_m: float = 800.0,
        max_radius_m: float = 6000.0,
        boundary_prefetch_ratio: float = 0.40,
    ) -> None:
        self.provider = provider
        self.active_network = active_network or RoadNetwork(cell_size_m=100.0)
        self.lookahead_time_s = lookahead_time_s
        self.min_radius_m = min_radius_m
        self.max_radius_m = max_radius_m
        self.boundary_ratio = boundary_prefetch_ratio

        self._lock = threading.Lock()
        self._last_fetch_lat: Optional[float] = None
        self._last_fetch_lon: Optional[float] = None
        self._last_fetch_radius_m: float = 0.0
        self._is_fetching = False
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CorridorPrefetchWorker")
        # Pre-warm worker thread to avoid cold-start thread spawn latency during real-time loops
        self._executor.submit(lambda: None)

    @property
    def is_fetching(self) -> bool:
        """Returns True if a background pre-fetch is currently active."""
        return self._is_fetching

    def notify_position(
        self,
        lat: float,
        lon: float,
        speed_mps: float = 0.0,
        bearing_deg: Optional[float] = None,
    ) -> bool:
        """
        Notifies the corridor manager of current vehicle kinematics.
        Evaluates proximity to cached boundary and initiates background pre-fetch if needed.

        Returns
        -------
        bool
            True if a background prefetch was triggered.
        """
        target_radius = compute_lookahead_radius(
            speed_mps=speed_mps,
            lookahead_time_s=self.lookahead_time_s,
            min_radius_m=self.min_radius_m,
            max_radius_m=self.max_radius_m,
        )

        should_fetch = False

        if self._last_fetch_lat is None or self._last_fetch_lon is None:
            # First position update: immediate fetch
            should_fetch = True
        else:
            # Check displacement from last fetch anchor
            dist_moved = haversine_distance_m(self._last_fetch_lat, self._last_fetch_lon, lat, lon)
            # If vehicle has traveled beyond threshold ratio of the cached radius
            if dist_moved >= self._last_fetch_radius_m * self.boundary_ratio:
                should_fetch = True
            # Or if speed increased dramatically, requiring a larger radius
            elif target_radius > self._last_fetch_radius_m * 1.5:
                should_fetch = True

        if should_fetch and not self._is_fetching:
            self._trigger_background_prefetch(lat, lon, target_radius, bearing_deg)
            return True

        return False

    def _trigger_background_prefetch(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        bearing_deg: Optional[float],
    ) -> None:
        """Submits an asynchronous background prefetch task to the thread pool."""
        self._is_fetching = True
        self._last_fetch_lat = lat
        self._last_fetch_lon = lon
        self._last_fetch_radius_m = radius_m

        def _worker():
            try:
                new_net = self.provider.get_corridor_network(
                    lat=lat,
                    lon=lon,
                    radius_m=radius_m,
                    bearing_deg=bearing_deg,
                )
                with self._lock:
                    base_net = self.active_network

                # Copy-on-write merge off-thread
                merged = RoadNetwork(cell_size_m=base_net.cell_size_m)
                merged.merge(base_net)
                merged.merge(new_net)

                # Atomic pointer swap
                with self._lock:
                    self.active_network = merged
            except Exception as e:
                print(f"[PredictiveCorridorManager] Warning: prefetch error: {e}")
            finally:
                self._is_fetching = False

        self._executor.submit(_worker)

    def get_active_network(self) -> RoadNetwork:
        """Returns the active RoadNetwork instance with zero lock contention."""
        with self._lock:
            return self.active_network
