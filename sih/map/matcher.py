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
        min_emission_score: Optional[float] = None,
        hard_dist_k: float = 2.5,
        hard_heading_k: float = 2.0,
        min_ambiguity_ratio: float = 1.5,
        hysteresis_count: int = 2,
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
        # Legacy min_emission_score kept for backward compat but no longer the primary gate
        self.min_emission_score = min_emission_score if min_emission_score is not None else float(np.exp(-4.5))

        # --- Three-Stage Confidence Gate Config ---
        # Stage 1: Hard geometric reject thresholds (multiples of sigma)
        self.hard_dist_k = hard_dist_k          # reject if d_perp > hard_dist_k * sigma_dist_m
        self.hard_heading_k = hard_heading_k      # reject if h_diff > hard_heading_k * sigma_heading_deg
        # Stage 2: Ambiguity ratio gate
        self.min_ambiguity_ratio = min_ambiguity_ratio  # skip snap if score_1/score_2 < this
        # Stage 3: Hysteresis — consecutive confident steps required to resume after suppression
        self.hysteresis_count = hysteresis_count

        # --- Gate Diagnostics ---
        self.last_emission_score: Optional[float] = None
        self.total_match_steps: int = 0
        self.gate_suppressed_steps: int = 0           # total (any gate)
        self.hard_gate_suppressed_steps: int = 0      # stage 1 hard reject
        self.ambiguity_gate_suppressed_steps: int = 0  # stage 2 ambiguity
        self.hysteresis_suppressed_steps: int = 0      # stage 3 hysteresis hold
        self.single_candidate_steps: int = 0          # single candidate survived (no ambiguity check)
        self.score_ratios: List[float] = []            # score_1/score_2 per step (only when >= 2 candidates)

        # Hysteresis state
        self._suppressed: bool = False
        self._confident_streak: int = 0

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
        """
        Build directed topological successor lookup table for road segments.
        Uses exact node identity (seg_a.end_node_id == seg_b.start_node_id) when node IDs are present.
        Falls back to 8.0m endpoint distance heuristic with spatial hash grid when node IDs are absent.
        """
        self._succ_map = {}
        if not self.road_network or not self.road_network.segments:
            return

        # Check for cached topological successor map on the road network
        cached = getattr(self.road_network, "_cached_succ_map", None)
        if cached is not None:
            self._succ_map = cached
            return

        # Check whether segments have node IDs
        has_node_ids = any(
            s.start_node_id is not None and s.end_node_id is not None
            for s in self.road_network.segments
        )

        if has_node_ids:
            # Primary topology: exact node identity matching
            start_node_map: Dict[Any, List[RoadSegment]] = {}
            for s in self.road_network.segments:
                if s.start_node_id is not None:
                    if s.start_node_id not in start_node_map:
                        start_node_map[s.start_node_id] = []
                    start_node_map[s.start_node_id].append(s)

            for s1 in self.road_network.segments:
                succs = []
                if s1.end_node_id is not None and s1.end_node_id in start_node_map:
                    for s2 in start_node_map[s1.end_node_id]:
                        if s1.segment_id != s2.segment_id:
                            # Exclude immediate U-turn onto own reverse segment or exact opposite node edge
                            if s2.segment_id == f"{s1.segment_id}_rev" or s1.segment_id == f"{s2.segment_id}_rev":
                                continue
                            if s1.start_node_id is not None and s1.start_node_id == s2.end_node_id and s1.end_node_id == s2.start_node_id:
                                continue
                            succs.append(s2)
                self._succ_map[s1.segment_id] = succs
        else:
            # Fallback: 8.0m endpoint distance heuristic with spatial hash grid
            dist_threshold_m = 8.0
            dist_threshold_sq = dist_threshold_m ** 2
            cell_size = max(15.0, dist_threshold_m * 1.5)
            start_grid: Dict[Tuple[int, int], List[RoadSegment]] = {}
            for s in self.road_network.segments:
                cx = int(np.floor(s.start_enu_m[0] / cell_size))
                cy = int(np.floor(s.start_enu_m[1] / cell_size))
                key = (cx, cy)
                if key not in start_grid:
                    start_grid[key] = []
                start_grid[key].append(s)

            for s1 in self.road_network.segments:
                succs = []
                ecx = int(np.floor(s1.end_enu_m[0] / cell_size))
                ecy = int(np.floor(s1.end_enu_m[1] / cell_size))
                e_pt = s1.end_enu_m
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cand_list = start_grid.get((ecx + dx, ecy + dy), None)
                        if cand_list is not None:
                            for s2 in cand_list:
                                if s1.segment_id != s2.segment_id:
                                    if s2.segment_id == f"{s1.segment_id}_rev" or s1.segment_id == f"{s2.segment_id}_rev":
                                        continue
                                    d2 = (e_pt[0] - s2.start_enu_m[0])**2 + (e_pt[1] - s2.start_enu_m[1])**2
                                    if d2 <= dist_threshold_sq:
                                        succs.append(s2)
                self._succ_map[s1.segment_id] = succs

        # Cache on road network object for instant reuse across scenario evaluations
        setattr(self.road_network, "_cached_succ_map", self._succ_map)

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
        self.last_emission_score = None
        self.total_match_steps = 0
        self.gate_suppressed_steps = 0
        self.hard_gate_suppressed_steps = 0
        self.ambiguity_gate_suppressed_steps = 0
        self.hysteresis_suppressed_steps = 0
        self.single_candidate_steps = 0
        self.score_ratios = []
        self._suppressed = False
        self._confident_streak = 0

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

        # --- Stage 1: Hard Geometric Reject ---
        # Pre-filter candidates before emission scoring.
        # Reject any candidate with d_perp > hard_dist_k * sigma_d
        # OR h_diff > hard_heading_k * sigma_h.
        hard_dist_limit = self.hard_dist_k * self.sigma_dist_m
        hard_heading_limit = self.hard_heading_k * self.sigma_heading_deg

        scored_candidates = []
        for sid, (seg, role) in cands_dict.items():
            proj_enu, d_perp, frac = seg.project_point(p_enu)
            if d_perp > 35.0:
                continue

            h_diff = abs((v_heading_deg - seg.bearing_deg + 180.0) % 360.0 - 180.0)
            branch_turn_deg = (seg.bearing_deg - active_bearing + 180.0) % 360.0 - 180.0

            # Hard geometric reject (Stage 1): skip candidate entirely
            if d_perp > hard_dist_limit or (v_speed > 1.0 and h_diff > hard_heading_limit):
                continue

            # Max allowable heading discrepancy:
            if role in ("succ", "succ2"):
                max_allowed_hdiff = 105.0
            elif is_single_corridor:
                max_allowed_hdiff = 85.0
            else:
                max_allowed_hdiff = 70.0 if is_turning_intent else 50.0

            # Dynamic Turn-Intent Prior at Diverging Junctions:
            # If the driver is actively steering into a turn:
            topo_bonus = 1.0
            if is_turning_intent and not is_single_corridor:
                # Check if this road branch aligns with the driver's turn direction
                if abs(branch_turn_deg) > 20.0:
                    if (self._trailing_turn_deg < -4.0 and branch_turn_deg < -15.0) or \
                       (self._trailing_turn_deg > 4.0 and branch_turn_deg > 15.0):
                        # Branch direction matches driver's turn intent!
                        topo_bonus *= 3.5
                        max_allowed_hdiff = max(max_allowed_hdiff, 110.0)
                    else:
                        # Branch turns in opposite direction
                        topo_bonus *= 0.2
                elif abs(branch_turn_deg) <= 15.0:
                    # Straight continuation branch: downweight when driver is actively turning
                    if abs(self._trailing_turn_deg) >= 10.0:
                        topo_bonus *= 0.25

            if v_speed > 1.0 and h_diff > max_allowed_hdiff:
                continue

            # Topological transition prior
            if role == "active":
                topo_bonus *= 0.3 if frac >= 0.90 else 1.5
            elif role == "succ":
                active_frac = self._active_segment.project_point(p_enu)[2] if self._active_segment else 0.0
                topo_bonus *= 3.0 if active_frac >= 0.75 else 1.2
            elif role == "succ2":
                topo_bonus *= 1.0

            # Soft Likelihood
            p_dist = np.exp(-0.5 * (d_perp / 12.0) ** 2)
            p_head = np.exp(-0.5 * (h_diff / 35.0) ** 2)
            score = float(p_dist * p_head * topo_bonus)

            scored_candidates.append((seg, proj_enu, d_perp, h_diff, score, frac))

        self.total_match_steps += 1

        # If no candidates survived the hard gate, return unsnapped
        if not scored_candidates:
            self.last_emission_score = 0.0
            self.gate_suppressed_steps += 1
            self.hard_gate_suppressed_steps += 1
            self._suppressed = True
            self._confident_streak = 0
            return self._create_unmatched(position, p_enu, v_heading_deg)

        scored_candidates.sort(key=lambda x: x[4], reverse=True)
        best_seg, best_proj, best_dist, best_h_diff, best_score, frac = scored_candidates[0]
        self.last_emission_score = float(best_score)

        # --- Stage 2: Ambiguity Gate ---
        # If there are >= 2 surviving candidates, compute score ratio.
        # If best / second-best < min_ambiguity_ratio, the match is ambiguous — skip snap.
        ambiguity_ok = True
        if len(scored_candidates) >= 2:
            second_score = scored_candidates[1][4]
            ratio = best_score / max(second_score, 1e-12)
            self.score_ratios.append(float(ratio))
            if ratio < self.min_ambiguity_ratio:
                ambiguity_ok = False
        else:
            # Single candidate survived hard gate: track in own category, exclude from ratio distribution
            self.single_candidate_steps += 1

        if not ambiguity_ok:
            self.gate_suppressed_steps += 1
            self.ambiguity_gate_suppressed_steps += 1
            self._suppressed = True
            self._confident_streak = 0
            return self._create_unmatched(position, p_enu, v_heading_deg)

        # --- Stage 3: Hysteresis ---
        # Once suppressed, require `hysteresis_count` consecutive confident steps
        # before resuming snapping, to prevent flicker.
        if self._suppressed:
            self._confident_streak += 1
            if self._confident_streak < self.hysteresis_count:
                self.gate_suppressed_steps += 1
                self.hysteresis_suppressed_steps += 1
                return self._create_unmatched(position, p_enu, v_heading_deg)
            else:
                # Hysteresis satisfied — resume snapping
                self._suppressed = False
                self._confident_streak = 0

        # Domain-Appropriate Road Snapping:
        if domain == "Urban" or best_h_diff > 40.0 or is_turning_intent:
            # Urban grid intersections & sharp turns: project to corner point
            snapped_enu = best_proj.copy()
        else:
            # Highway / Arterial corridor: strictly perpendicular lateral snap
            # Preserves along-track DR integration without coordinate teleportation
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
                ekf._p[0] = snapped_enu[0]
                ekf._p[1] = snapped_enu[1]
            if hasattr(ekf, "reanchor_heading"):
                # Only re-anchor heading if NOT actively cornering across a junction.
                # If the vehicle is mid-turn (|h_diff| > 20 deg or is_turning_intent),
                # let the gyroscope continue to integrate the turn without fighting the steering!
                if not (is_turning_intent and best_h_diff > 20.0):
                    conf = 0.5 if turn_rate_dps < 1.5 else 0.15
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
