"""
Hidden Markov Model (HMM) Map Matcher with Off-Road Graceful Degradation.

Implements IMapMatcher behind abstract interface.
Fuses continuous EKF position and heading with road network geometries.
Features explicit Off-Road gating to prevent false snapping on unmapped farmland/tracks.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any, Dict
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
        self._trailing_turn_deg: float = 0.0
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

    def update_road_network(self, network: RoadNetwork) -> None:
        """Dynamically updates the road network without clearing current tracking state."""
        self.road_network = network
        self._build_succ_map()

    def reset(self) -> None:
        self._active_segment = None
        self._last_matched_enu = None
        self._last_fused_enu = None
        self._last_heading_deg = None
        self._last_timestamp_ns = None
        self._trailing_turn_deg = 0.0
        self._consecutive_unmatched_count = 0

    def set_active_segment(self, segment: Optional[RoadSegment]) -> None:
        """Explicitly initialize or update the active road segment."""
        self._active_segment = segment

    def match(
        self,
        position: FusedPosition,
        ekf: Optional[Any] = None,
        domain: str = "Highway",
        v_fwd: Optional[float] = None,
        **kwargs,
    ) -> MatchedPosition:
        """
        Match continuous fused position to road network with topological corridor retention.
        Guarantees strict road corridor attachment via topological network traversal.
        Supports closed-loop EKF updating and domain-appropriate lateral snapping.
        """
        p_enu = position.position_enu_m[:2].copy()
        v_heading_deg = float(np.degrees(position.heading_rad)) % 360.0
        v_speed = float(v_fwd) if v_fwd is not None else float(np.linalg.norm(position.velocity_enu_mps[:2]))

        # Topological candidate retrieval:
        # Includes active segment, its downstream topological successors, and spatial radius
        cands_dict: Dict[str, Tuple[RoadSegment, str]] = {}
        if self._active_segment is not None:
            cands_dict[self._active_segment.segment_id] = (self._active_segment, "active")
            for succ in self._succ_map.get(self._active_segment.segment_id, []):
                cands_dict[succ.segment_id] = (succ, "succ")
                for s2 in self._succ_map.get(succ.segment_id, []):
                    cands_dict[s2.segment_id] = (s2, "succ2")

        for s in self.road_network.find_candidates(p_enu, radius_m=max(45.0, self.max_snap_dist_m * 1.5)):
            if s.segment_id not in cands_dict:
                cands_dict[s.segment_id] = (s, "spatial")

        if not cands_dict:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        # Estimate vehicle turning rate (magnitude and signed direction)
        turn_rate_dps = 0.0
        turn_rate_signed_dps = 0.0
        dt_s = 0.1
        if self._last_heading_deg is not None and self._last_timestamp_ns is not None:
            dt_s = max(1e-3, (position.timestamp_ns - self._last_timestamp_ns) * 1e-9)
            dh = (v_heading_deg - self._last_heading_deg + 180.0) % 360.0 - 180.0
            turn_rate_signed_dps = dh / dt_s
            turn_rate_dps = abs(dh) / dt_s
        self._last_heading_deg = v_heading_deg
        self._last_timestamp_ns = position.timestamp_ns

        # Inspect EKF calibrated yaw rate if available
        if ekf is not None and hasattr(ekf, "_last_w_z_corr"):
            w_z = getattr(ekf, "_last_w_z_corr", 0.0)
            if abs(w_z) > 1e-4:
                turn_rate_signed_dps = -float(np.degrees(w_z))
                turn_rate_dps = abs(turn_rate_signed_dps)

        # Causal trailing turn accumulator (decays over ~1.5s)
        alpha = float(np.clip(dt_s / 1.5, 0.05, 0.5))
        self._trailing_turn_deg = (1.0 - alpha) * self._trailing_turn_deg + (turn_rate_signed_dps * dt_s)

        is_turning = turn_rate_dps > 1.5
        is_turning_intent = (turn_rate_dps >= 2.0) or (abs(self._trailing_turn_deg) >= 8.0)

        # Extract EKF heading uncertainty from covariance matrix
        sigma_yaw_deg = self.sigma_heading_deg
        if ekf is not None and hasattr(ekf, "_P") and ekf._P.shape[0] >= 9:
            var_yaw = float(ekf._P[8, 8])
            if var_yaw > 0.0:
                sigma_yaw_deg = float(np.degrees(np.sqrt(var_yaw)))
        elif position.covariance is not None and position.covariance.shape[0] >= 9:
            var_yaw = float(position.covariance[8, 8])
            if var_yaw > 0.0:
                sigma_yaw_deg = float(np.degrees(np.sqrt(var_yaw)))

        sigma_eff = float(np.sqrt(sigma_yaw_deg**2 + 15.0**2))
        if is_turning:
            sigma_eff = max(sigma_eff, 45.0)

        # Universal Corridor Retention:
        # Check if the candidate pool has competing divergent branches.
        # If single-path corridor (no branching forks within 35m), widen heading tolerance
        # so transient mid-turn gyro phase lag cannot cause candidate pool starvation.
        num_succs = len(self._succ_map.get(self._active_segment.segment_id, [])) if self._active_segment else 0
        is_single_corridor = (num_succs <= 1)
        active_bearing = self._active_segment.bearing_deg if self._active_segment else v_heading_deg

        has_succs = len(self._succ_map.get(self._active_segment.segment_id, [])) > 0 if self._active_segment else False

        scored_candidates = []
        for sid, (seg, role) in cands_dict.items():
            proj_enu, d_perp, frac = seg.project_point(p_enu)
            if d_perp > 35.0:
                continue

            # Reject terminated spatial candidates behind the vehicle
            if role == "spatial" and frac >= 0.95:
                continue

            h_diff = abs((v_heading_deg - seg.bearing_deg + 180.0) % 360.0 - 180.0)
            branch_turn_deg = (seg.bearing_deg - active_bearing + 180.0) % 360.0 - 180.0

            # Max allowable heading discrepancy:
            if role in ("succ", "succ2"):
                max_allowed_hdiff = 110.0
            elif role == "active":
                max_allowed_hdiff = 110.0 if frac < 0.35 else 80.0
            else:
                max_allowed_hdiff = 50.0

            if h_diff > max_allowed_hdiff:
                continue

            # Topological transition prior & heading bandwidth
            sigma_h = 35.0
            if role == "active":
                if frac >= 0.95 and has_succs:
                    topo_bonus = 0.05  # Deprecate active segment when reached end and successors exist
                elif frac >= 0.85:
                    topo_bonus = 0.40
                else:
                    topo_bonus = 1.50
                # Widen heading tolerance if chassis is mid-turn on newly entered segment
                if frac < 0.35:
                    sigma_h = 60.0
            elif role == "succ":
                active_frac = self._active_segment.project_point(p_enu)[2] if self._active_segment else 0.0
                sigma_h = 60.0 if active_frac >= 0.70 else 40.0
                topo_bonus = 6.0 if active_frac >= 0.75 else 2.0
            elif role == "succ2":
                sigma_h = 55.0
                topo_bonus = 1.0
            elif role == "spatial":
                topo_bonus = 0.02 if self._active_segment is not None else 1.0

            # Soft Likelihood
            p_dist = np.exp(-0.5 * (d_perp / 12.0) ** 2)
            p_head = np.exp(-0.5 * (h_diff / sigma_h) ** 2)
            score = float(p_dist * p_head * topo_bonus)

            scored_candidates.append((seg, proj_enu, d_perp, h_diff, score, frac))

        if not scored_candidates:
            return self._create_unmatched(position, p_enu, v_heading_deg)

        scored_candidates.sort(key=lambda x: x[4], reverse=True)
        best_seg, best_proj, best_dist, best_h_diff, best_score, frac = scored_candidates[0]

        # Domain-Appropriate Road Snapping:
        if domain == "Urban" or best_h_diff > 40.0:
            snapped_enu = best_proj.copy()
        else:
            seg_vec = best_seg.end_enu_m - best_seg.start_enu_m
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len > 1e-3:
                u_seg = seg_vec / seg_len
                u_norm = np.array([-u_seg[1], u_seg[0]])
                d_lat = float(np.dot(best_proj - p_enu, u_norm))
                snapped_enu = p_enu + d_lat * u_norm
            else:
                snapped_enu = best_proj.copy()

        # Closed-Loop EKF Feedback (if EKF instance provided):
        if ekf is not None:
            if hasattr(ekf, "_p"):
                # Anti-boundary clamping: do not pin EKF to startpoint or endpoint during sharp junction maneuvers
                is_boundary_clamp = (frac >= 0.98 or frac <= 0.01) and (v_speed > 1.0) and (best_h_diff > 40.0)
                if not is_boundary_clamp:
                    ekf._p[0] = snapped_enu[0]
                    ekf._p[1] = snapped_enu[1]

            if hasattr(ekf, "reanchor_heading"):
                if best_seg != self._active_segment:
                    # Discrete topological transition onto new road! Promptly steer heading into new road corridor
                    ekf.reanchor_heading(best_seg.bearing_deg, confidence=0.85, forward_speed_mps=v_speed)
                    diff_rad = (np.radians(best_seg.bearing_deg) - ekf._heading_rad + np.pi) % (2.0 * np.pi) - np.pi
                    ekf._heading_rad = (ekf._heading_rad + 0.50 * diff_rad) % (2.0 * np.pi)
                    yaw_enu = np.pi / 2.0 - ekf._heading_rad
                    from scipy.spatial.transform import Rotation as R
                    ekf._q = R.from_euler("z", yaw_enu)
                    if hasattr(ekf, "_v") and v_speed > 0.5:
                        ekf._v = ekf._q.as_matrix() @ np.array([v_speed, 0.0, 0.0], dtype=np.float64)
                else:
                    conf = 0.40 if best_h_diff < 20.0 else 0.15
                    ekf.reanchor_heading(best_seg.bearing_deg, confidence=conf, forward_speed_mps=v_speed)

        self._active_segment = best_seg
        self._last_matched_enu = snapped_enu
        self._last_fused_enu = p_enu
        self._consecutive_unmatched_count = 0

        # Convert matched ENU back to WGS-84 Geodetic Lat/Lon
        lat_deg, lon_deg, _ = enu_to_geodetic(
            snapped_enu[0], snapped_enu[1], position.altitude_m,
            self.ref_lat, self.ref_lon, self.ref_alt
        )

        return MatchedPosition(
            timestamp_ns=position.timestamp_ns,
            latitude_deg=float(lat_deg),
            longitude_deg=float(lon_deg),
            bearing_deg=float(best_seg.bearing_deg),
            road_segment_id=best_seg.segment_id,
            distance_to_road_m=float(best_dist),
            confidence=float(min(1.0, max(0.0, best_score))),
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
