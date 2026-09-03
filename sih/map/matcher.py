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
from sih.data.geo import enu_to_geodetic


class HMMMapMatcher(IMapMatcher):
    """
    Probabilistic HMM Map Matcher with Off-Road Graceful Degradation.
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
    ) -> None:
        self.road_network = road_network or RoadNetwork()
        self.sigma_dist_m = sigma_dist_m
        self.sigma_heading_deg = sigma_heading_deg
        self.max_snap_dist_m = max_snap_dist_m
        self.max_heading_diff_deg = max_heading_diff_deg
        self.smoothing_factor = smoothing_factor
        self.min_confidence_threshold = min_confidence_threshold

        self.ref_lat = reference_lat_deg
        self.ref_lon = reference_lon_deg
        self.ref_alt = reference_alt_m

        self._active_segment: Optional[RoadSegment] = None
        self._last_matched_enu: Optional[np.ndarray] = None
        self._last_fused_enu: Optional[np.ndarray] = None
        self._last_heading_deg: Optional[float] = None
        self._last_timestamp_ns: Optional[int] = None
        self._consecutive_unmatched_count: int = 0

    def set_reference(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
        self.ref_lat = lat_deg
        self.ref_lon = lon_deg
        self.ref_alt = alt_m

    def set_road_network(self, network: RoadNetwork) -> None:
        self.road_network = network
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
        If off-road or on unmapped track, safely falls back without snapping.
        """
        p_enu = position.position_enu_m[:2]
        v_heading_deg = float(np.degrees(position.heading_rad)) % 360.0
        v_speed = float(np.linalg.norm(position.velocity_enu_mps[:2]))

        # Retrieve candidate segments within spatial radius
        candidates = self.road_network.find_candidates(p_enu, radius_m=self.max_snap_dist_m * 1.5)

        if not candidates:
            # Explicit Level 2 Off-Road Fallback: No roads in radius
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

        # Effective heading uncertainty: broad after turns, tight when settled
        sigma_eff = float(np.sqrt(sigma_yaw_ekf_deg**2 + self.sigma_heading_deg**2))
        if is_turning:
            sigma_eff = max(sigma_eff, 45.0)

        scored_candidates: List[Tuple[RoadSegment, np.ndarray, float, float, float]] = []

        for seg in candidates:
            proj_enu, d_perp, frac = seg.project_point(p_enu)
            if d_perp > self.max_snap_dist_m:
                continue

            # Heading difference modulo 360
            h_diff = abs((v_heading_deg - seg.bearing_deg + 180.0) % 360.0 - 180.0)

            # If vehicle is stationary, relax heading requirement
            if v_speed < 1.0:
                h_diff = min(h_diff, 15.0)

            # Soft Likelihood: covariance-driven without hard rejection gate
            p_dist = np.exp(-0.5 * (d_perp / self.sigma_dist_m) ** 2)
            p_head = np.exp(-0.5 * (h_diff / sigma_eff) ** 2)
            emission_score = float(p_dist * p_head)

            # Discard vanishingly unlikely candidates
            if emission_score < 1e-4:
                continue

            # Topological transition bonus / penalty:
            # If current segment has terminated (frac >= 0.95), downweight staying on dead-end segment
            if frac >= 0.95:
                transition_factor = 0.05
            elif self._active_segment and seg.segment_id == self._active_segment.segment_id:
                transition_factor = 1.25
            else:
                transition_factor = 1.0

            total_score = emission_score * transition_factor
            scored_candidates.append((seg, proj_enu, d_perp, h_diff, total_score))

        if not scored_candidates:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        scored_candidates.sort(key=lambda x: x[4], reverse=True)
        top_cands = scored_candidates[:2]

        # Multi-hypothesis fork resolution:
        # If top 2 candidates are parallel (bearing separation < 15 deg) and both credible,
        # delay hard-commit to avoid latching onto the wrong highway branch
        is_ambiguous_fork = False
        if len(top_cands) >= 2:
            h_sep = abs((top_cands[0][0].bearing_deg - top_cands[1][0].bearing_deg + 180.0) % 360.0 - 180.0)
            if h_sep < 15.0 and top_cands[1][4] > 0.40 * top_cands[0][4]:
                is_ambiguous_fork = True

        if is_ambiguous_fork:
            w0 = top_cands[0][4]
            w1 = top_cands[1][4]
            total_w = w0 + w1
            best_seg = top_cands[0][0]
            best_proj = (w0 * top_cands[0][1] + w1 * top_cands[1][1]) / total_w
            best_dist = float((w0 * top_cands[0][2] + w1 * top_cands[1][2]) / total_w)
            best_h_diff = top_cands[0][3]
            best_score = float(total_w / 2.0)
        else:
            best_seg, best_proj, best_dist, best_h_diff, best_score = scored_candidates[0]

        confidence = min(1.0, max(0.0, best_score))

        if confidence < self.min_confidence_threshold:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        # Smooth position transition to avoid sudden UI jumps
        if self._last_matched_enu is not None:
            smooth_enu = (1.0 - self.smoothing_factor) * p_enu + self.smoothing_factor * best_proj
        else:
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
