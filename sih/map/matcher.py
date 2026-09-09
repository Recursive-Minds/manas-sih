"""
Hidden Markov Model (HMM) Map Matcher with Off-Road Graceful Degradation.

Implements IMapMatcher behind abstract interface.
Fuses continuous EKF position and heading with road network geometries.
Features explicit Off-Road gating to prevent false snapping on unmapped farmland/tracks.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from sih.core.contracts import FusedPosition, MatchedPosition
from sih.core.interfaces import IMapMatcher
from sih.map.network import RoadNetwork, RoadSegment
from sih.map.governor import RoadKinematicsGovernor
from sih.data.geo import enu_to_geodetic


class HMMMapMatcher(IMapMatcher):
    """
    Probabilistic HMM Map Matcher with Off-Road Graceful Degradation and Curvature Governing.
    """
    def __init__(
        self,
        road_network: Optional[RoadNetwork] = None,
        sigma_dist_m: float = 6.0,
        sigma_heading_deg: float = 20.0,
        max_snap_dist_m: float = 30.0,
        max_heading_diff_deg: float = 50.0,
        smoothing_factor: float = 0.35,
        min_confidence_threshold: float = 0.25,
        reference_lat_deg: float = 0.0,
        reference_lon_deg: float = 0.0,
        reference_alt_m: float = 0.0,
        governor: Optional[RoadKinematicsGovernor] = None,
    ) -> None:
        self.road_network = road_network or RoadNetwork()
        self.sigma_dist_m = sigma_dist_m
        self.sigma_heading_deg = sigma_heading_deg
        self.max_snap_dist_m = max_snap_dist_m
        self.max_heading_diff_deg = max_heading_diff_deg
        self.smoothing_factor = smoothing_factor
        self.min_confidence_threshold = min_confidence_threshold
        self.governor = governor or RoadKinematicsGovernor()

        self.ref_lat = reference_lat_deg
        self.ref_lon = reference_lon_deg
        self.ref_alt = reference_alt_m

        self._active_segment: Optional[RoadSegment] = None
        self._last_matched_enu: Optional[np.ndarray] = None
        self._last_fused_enu: Optional[np.ndarray] = None
        self._last_heading_deg: Optional[float] = None
        self._last_timestamp_ns: Optional[int] = None
        self._consecutive_unmatched_count: int = 0
        self._matched_history: List[np.ndarray] = []
        self._succ_map: dict = {}
        self._build_succ_map()

    def _build_succ_map(self) -> None:
        """Build directed topological successor lookup table for road segments."""
        self._succ_map = {}
        if not self.road_network or not self.road_network.segments:
            return
        for s1 in self.road_network.segments:
            self._succ_map[s1.segment_id] = []
            for s2 in self.road_network.segments:
                if s1.segment_id != s2.segment_id:
                    if np.linalg.norm(s1.end_enu_m - s2.start_enu_m) < 8.0:
                        self._succ_map[s1.segment_id].append(s2)

    def set_reference(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
        self.ref_lat = lat_deg
        self.ref_lon = lon_deg
        self.ref_alt = alt_m

    def set_road_network(self, network: RoadNetwork) -> None:
        self.road_network = network
        self._build_succ_map()
        self.reset()

    def reset(self) -> None:
        self._active_segment = None
        self._last_matched_enu = None
        self._last_fused_enu = None
        self._last_heading_deg = None
        self._last_timestamp_ns = None
        self._consecutive_unmatched_count = 0

    def match(self, position: FusedPosition) -> MatchedPosition:
        """
        Match continuous fused position to road network.
        Guarantees strict road corridor attachment via topological network traversal.
        If off-road or on unmapped track, safely falls back without snapping.
        """
        p_enu = position.position_enu_m[:2]
        v_heading_deg = float(np.degrees(position.heading_rad)) % 360.0
        v_speed = float(np.linalg.norm(position.velocity_enu_mps[:2]))

        # Topological candidate retrieval:
        # Includes active segment, its downstream topological successors, and spatial radius
        cands_dict = {}
        if self._active_segment is not None:
            cands_dict[self._active_segment.segment_id] = (self._active_segment, "active")
            for succ in self._succ_map.get(self._active_segment.segment_id, []):
                cands_dict[succ.segment_id] = (succ, "succ")
                for s2 in self._succ_map.get(succ.segment_id, []):
                    cands_dict[s2.segment_id] = (s2, "succ2")

        for s in self.road_network.find_candidates(p_enu, radius_m=self.max_snap_dist_m * 1.5):
            if s.segment_id not in cands_dict:
                cands_dict[s.segment_id] = (s, "spatial")

        if not cands_dict:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        # Estimate vehicle turning rate
        turn_rate_dps = 0.0
        if self._last_heading_deg is not None and self._last_timestamp_ns is not None:
            dt_s = max(1e-3, (position.timestamp_ns - self._last_timestamp_ns) * 1e-9)
            dh = (v_heading_deg - self._last_heading_deg + 180.0) % 360.0 - 180.0
            turn_rate_dps = abs(dh) / dt_s
        self._last_heading_deg = v_heading_deg
        self._last_timestamp_ns = position.timestamp_ns

        is_turning = turn_rate_dps > 1.5

        # Extract EKF heading uncertainty from covariance matrix
        sigma_yaw_ekf_deg = self.sigma_heading_deg
        if position.covariance is not None and position.covariance.shape[0] >= 9:
            var_yaw = float(position.covariance[8, 8])
            if var_yaw > 0.0:
                sigma_yaw_ekf_deg = float(np.degrees(np.sqrt(var_yaw)))

        sigma_eff = float(np.sqrt(sigma_yaw_ekf_deg**2 + self.sigma_heading_deg**2))
        if is_turning:
            sigma_eff = max(sigma_eff, 45.0)

        scored_candidates: List[Tuple[RoadSegment, np.ndarray, float, float, float]] = []

        for sid, (seg, role) in cands_dict.items():
            proj_enu, d_perp, frac = seg.project_point(p_enu)
            if d_perp > self.max_snap_dist_m:
                continue

            h_diff = abs((v_heading_deg - seg.bearing_deg + 180.0) % 360.0 - 180.0)

            # Curve-tolerant topological gating:
            # Allow connected successors to accommodate turns up to 60 deg
            max_allowed_hdiff = 60.0 if role in ("succ", "succ2") else 45.0
            if v_speed > 1.0 and h_diff > max_allowed_hdiff:
                continue

            # Soft Likelihood
            p_dist = np.exp(-0.5 * (d_perp / self.sigma_dist_m) ** 2)
            p_head = np.exp(-0.5 * (h_diff / sigma_eff) ** 2)
            emission_score = float(p_dist * p_head)

            if emission_score < 1e-4:
                continue

            # Topological transition prior
            topo_bonus = 1.0
            if role == "active":
                topo_bonus = 0.3 if frac >= 0.90 else 1.5
            elif role == "succ":
                active_frac = self._active_segment.project_point(p_enu)[2] if self._active_segment else 0.0
                topo_bonus = 3.0 if active_frac >= 0.75 else 1.2
            elif role == "succ2":
                topo_bonus = 1.0

            total_score = emission_score * topo_bonus
            scored_candidates.append((seg, proj_enu, d_perp, h_diff, total_score))

        if not scored_candidates:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        scored_candidates.sort(key=lambda x: x[4], reverse=True)
        best_seg, best_proj, best_dist, best_h_diff, best_score = scored_candidates[0]

        confidence = min(1.0, max(0.0, best_score))
        if confidence < self.min_confidence_threshold:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        # STRICT ROAD ATTACHMENT:
        # Snaps directly to the road centerline projection
        smooth_enu = best_proj

        self._active_segment = best_seg
        self._last_matched_enu = smooth_enu
        self._last_fused_enu = p_enu
        self._consecutive_unmatched_count = 0

        # Convert matched ENU back to WGS-84 Geodetic Lat/Lon
        lat_deg, lon_deg, _ = enu_to_geodetic(
            smooth_enu[0], smooth_enu[1], position.altitude_m,
            self.ref_lat, self.ref_lon, self.ref_alt
        )

        return MatchedPosition(
            timestamp_ns=position.timestamp_ns,
            latitude_deg=float(lat_deg),
            longitude_deg=float(lon_deg),
            bearing_deg=float(best_seg.bearing_deg),
            road_segment_id=best_seg.segment_id,
            distance_to_road_m=float(best_dist),
            confidence=float(confidence),
            is_matched=True,
        )

    def _create_unmatched(self, position: FusedPosition, p_enu: np.ndarray, v_heading_deg: float) -> MatchedPosition:
        """
        Creates an un-snapped position when off-road, preserving raw dead reckoning.
        """
        self._consecutive_unmatched_count += 1
        if self._consecutive_unmatched_count > 3:
            self._active_segment = None

        self._last_matched_enu = p_enu
        self._last_fused_enu = p_enu

        return MatchedPosition(
            timestamp_ns=position.timestamp_ns,
            latitude_deg=position.latitude_deg,
            longitude_deg=position.longitude_deg,
            bearing_deg=v_heading_deg,
            road_segment_id=None,
            distance_to_road_m=0.0,
            confidence=0.0,
            is_matched=False,
        )
