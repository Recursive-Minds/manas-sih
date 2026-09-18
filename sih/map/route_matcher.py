"""
sih.map.route_matcher
---------------------
Route-level topological matching for GNSS blackouts.
Hypothesis testing over enumerated DFS candidate paths:
1. Turn Sequence Extraction: Integrates calibrated yaw rate and forward speed.
2. Route Enumeration: Depth-limited DFS over succ_map bounded by DR distance (0.7x-1.4x).
3. Route Scoring: Turn alignment (greedy) with penalties for turn and length mismatch, plus weak d_perp tiebreak.
4. Arclength Projection: Coordinates reconstructed along winning route by along-track distance.
5. Two-Level Fallback: Falls back to 3-stage gated per-step matcher if ratio < 1.8 or cost > max_acceptable_cost.

OPERATIONAL STATUS - DISABLED BY DEFAULT:
Route matching is DISABLED by default across the entire SIH pipeline (enable_route_matching = False).
Diagnostic investigations on real-world Indian road data established:
1. Arclength Tangent Overshoot: Speed scale estimation errors (10-20%) cause arclengths to exceed
   the winning route length, extrapolating along the tangent of the final segment into empty space.
   This severely degraded accuracy on true roads (e.g. Scenario 12 regressed from 10.5% to 41.8%).
2. Lack of Absolute Cost Discrimination: Checking only relative ratio (score_1 / score_2) allowed false wins
   on high-cost routes far from ground truth (Scenario 25 regressed from 4.9% to 59.3%).
3. Benchmark Validation: Disabling route matching yields 11.59% median drift across 40 scenarios
   (vs 12.78% with route matching active). The per-step 3-stage gated matcher outperforms Pure DR
   on 35 / 40 scenarios (87.5% win rate) when running alone.
All code, regression tests, and log-space fixes are maintained intact for reference and research.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
import tracemalloc
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from sih.map.network import RoadSegment, RoadNetwork


@dataclass
class TurnEvent:
    """Represents a discrete vehicle or road turn event."""
    delta_heading_deg: float   # Signed turn angle in degrees (+right / clockwise, -left / CCW)
    arclength_m: float         # Along-track distance from blackout entry at which turn occurs
    timestamp_ns: int = 0      # Timestamp of turn culmination (for observed turns)


@dataclass
class TurnSequence:
    """Sequence of turn events observed during a blackout or extracted from a route."""
    turns: List[TurnEvent] = field(default_factory=list)
    total_distance_m: float = 0.0


@dataclass
class CandidateRoute:
    """An enumerated candidate path through the road network."""
    segments: List[RoadSegment]
    cum_lengths: List[float]   # Cumulative length up to the end of each segment
    total_length_m: float
    turn_sequence: TurnSequence = field(default_factory=TurnSequence)
    score: float = 0.0
    cost: float = 0.0
    turn_cost: float = 0.0
    length_cost: float = 0.0
    d_perp_cost: float = 0.0


@dataclass
class RouteMatchResult:
    """Outcome of route-level matching hypothesis evaluation."""
    won: bool
    confidence_ratio: float
    winning_route: Optional[CandidateRoute]
    second_route: Optional[CandidateRoute]
    projected_pts_enu: Optional[np.ndarray]
    total_routes_evaluated: int
    enumeration_time_ms: float
    scoring_time_ms: float
    observed_turns_count: int
    fallback_reason: str = ""
    peak_memory_kb: float = 0.0


def extract_turn_sequence_from_imu(
    timestamps_ns: np.ndarray,
    yaw_rates_rad_s: np.ndarray,
    speeds_mps: np.ndarray,
    window_s: float = 5.0,
    turn_threshold_deg: float = 30.0,
    min_turn_separation_s: float = 2.0,
) -> TurnSequence:
    """
    Extracts observed discrete turn events during a GNSS blackout by integrating
    calibrated vehicle yaw rate and forward velocity.

    A turn event is detected when the cumulative heading change exceeds turn_threshold_deg
    (default 30 deg) within a sliding window of window_s (default 5.0s).

    Parameters
    ----------
    timestamps_ns : np.ndarray
        Array of timestamps in nanoseconds.
    yaw_rates_rad_s : np.ndarray
        Calibrated vehicle yaw rate about vertical axis (rad/s).
        Sign convention: negative yaw rate = turning right (increasing compass bearing).
    speeds_mps : np.ndarray
        Forward vehicle speed (m/s).
    window_s : float
        Sliding window duration in seconds (default 5.0s).
    turn_threshold_deg : float
        Minimum cumulative turn angle to emit a turn event (default 30.0 deg).
    min_turn_separation_s : float
        Minimum time separation between consecutive distinct turns (default 2.0s).

    Returns
    -------
    TurnSequence
        Extracted observed turn sequence with signed turn angles and along-track distances.
    """
    n = len(timestamps_ns)
    if n < 2:
        return TurnSequence(turns=[], total_distance_m=0.0)

    # Compute continuous cumulative along-track distance and compass heading
    dt_s = np.diff(timestamps_ns) * 1e-9
    dt_s = np.clip(dt_s, 1e-4, 1.0)

    cum_dist = np.zeros(n, dtype=np.float64)
    cum_heading = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        avg_speed = 0.5 * (speeds_mps[i - 1] + speeds_mps[i])
        cum_dist[i] = cum_dist[i - 1] + max(0.0, avg_speed) * dt_s[i - 1]

        # In vehicle coordinates, compass bearing rate = -degrees(w_z)
        # Clockwise / right turn: w_z < 0 -> d(bearing)/dt > 0
        avg_w_z = 0.5 * (yaw_rates_rad_s[i - 1] + yaw_rates_rad_s[i])
        d_bearing = -float(np.degrees(avg_w_z)) * dt_s[i - 1]
        cum_heading[i] = cum_heading[i - 1] + d_bearing

    total_dist = float(cum_dist[-1])
    turns: List[TurnEvent] = []

    # Detect turns using sliding window
    time_s = (timestamps_ns - timestamps_ns[0]) * 1e-9
    last_turn_time_s = -999.0

    in_turn = False
    turn_start_idx = 0
    turn_peak_idx = 0
    max_turn_mag = 0.0

    i = 0
    while i < n:
        t_curr = time_s[i]

        if not in_turn:
            # Check for turn onset across sliding window [t_curr - window_s, t_curr]
            w_start_idx = int(np.searchsorted(time_s, t_curr - window_s))
            dh = cum_heading[i] - cum_heading[w_start_idx]

            if abs(dh) >= turn_threshold_deg and (t_curr - last_turn_time_s) >= min_turn_separation_s:
                in_turn = True
                turn_start_idx = w_start_idx
                turn_peak_idx = i
                max_turn_mag = abs(dh)
        else:
            # During active turn, track maximum cumulative deflection
            dh = cum_heading[i] - cum_heading[turn_start_idx]
            if abs(dh) > max_turn_mag:
                max_turn_mag = abs(dh)
                turn_peak_idx = i

            # Detect turn completion: vehicle returns to straight travel or window elapses
            recent_start = int(np.searchsorted(time_s, t_curr - 1.0))
            recent_rate = abs(cum_heading[i] - cum_heading[recent_start]) / max(0.1, t_curr - time_s[recent_start])

            turn_duration = t_curr - time_s[turn_start_idx]
            if (recent_rate < 3.0 and turn_duration >= 1.5) or turn_duration >= (window_s * 1.5) or i == n - 1:
                signed_total_turn = float(cum_heading[turn_peak_idx] - cum_heading[turn_start_idx])
                if abs(signed_total_turn) >= turn_threshold_deg:
                    turns.append(TurnEvent(
                        delta_heading_deg=signed_total_turn,
                        arclength_m=float(cum_dist[turn_peak_idx]),
                        timestamp_ns=int(timestamps_ns[turn_peak_idx]),
                    ))
                    last_turn_time_s = time_s[turn_peak_idx]

                in_turn = False
                i = max(i, turn_peak_idx + 5)

        i += 1

    return TurnSequence(turns=turns, total_distance_m=total_dist)


def extract_turn_sequence_from_route(
    route: CandidateRoute,
    turn_threshold_deg: float = 30.0,
    merge_distance_m: float = 25.0,
) -> TurnSequence:
    """
    Extracts topological turn events along a CandidateRoute from segment bearing changes.
    Consecutive sub-segment bearing changes in the same direction within merge_distance_m
    are merged into a single turn event (handling curves split into multiple segments).

    Parameters
    ----------
    route : CandidateRoute
        Candidate route with ordered segments and cumulative lengths.
    turn_threshold_deg : float
        Minimum cumulative turn angle to register as a turn event.
    merge_distance_m : float
        Maximum distance between consecutive bends to merge into one turn.

    Returns
    -------
    TurnSequence
        Extracted turn sequence for the candidate route.
    """
    segs = route.segments
    if len(segs) < 2:
        return TurnSequence(turns=[], total_distance_m=route.total_length_m)

    raw_turns: List[Tuple[float, float]] = []
    for i in range(len(segs) - 1):
        s1 = segs[i]
        s2 = segs[i + 1]
        dh = (s2.bearing_deg - s1.bearing_deg + 180.0) % 360.0 - 180.0
        s_m = route.cum_lengths[i]
        if abs(dh) >= 5.0:
            raw_turns.append((dh, s_m))

    if not raw_turns:
        return TurnSequence(turns=[], total_distance_m=route.total_length_m)

    merged_turns: List[TurnEvent] = []
    curr_turn, curr_s = raw_turns[0]

    for i in range(1, len(raw_turns)):
        next_turn, next_s = raw_turns[i]
        if (next_s - curr_s) <= merge_distance_m and (curr_turn * next_turn) > 0:
            curr_turn += next_turn
            curr_s = 0.5 * (curr_s + next_s)
        else:
            if abs(curr_turn) >= turn_threshold_deg:
                merged_turns.append(TurnEvent(delta_heading_deg=curr_turn, arclength_m=curr_s))
            curr_turn, curr_s = next_turn, next_s

    if abs(curr_turn) >= turn_threshold_deg:
        merged_turns.append(TurnEvent(delta_heading_deg=curr_turn, arclength_m=curr_s))

    return TurnSequence(turns=merged_turns, total_distance_m=route.total_length_m)


def enumerate_routes_dfs(
    entry_segment: RoadSegment,
    succ_map: Dict[str, List[RoadSegment]],
    dr_distance_m: float,
    min_dist_ratio: float = 0.70,
    max_dist_ratio: float = 1.40,
    max_routes: int = 200,
    max_depth: int = 60,
) -> List[CandidateRoute]:
    """
    Depth-limited search over succ_map from the entry segment.
    Prunes paths exceeding max_dist_ratio * dr_distance_m and retains paths
    whose length falls within [min_dist_ratio, max_dist_ratio] of DR distance.

    Parameters
    ----------
    entry_segment : RoadSegment
        The active road segment at blackout entry.
    succ_map : Dict[str, List[RoadSegment]]
        Directed successor adjacency map from segment_id to list of downstream RoadSegments.
    dr_distance_m : float
        Estimated along-track travel distance under dead reckoning.
    min_dist_ratio : float
        Minimum length ratio (default 0.70).
    max_dist_ratio : float
        Maximum length ratio (default 1.40).
    max_routes : int
        Maximum number of candidate routes to enumerate (default 200).
    max_depth : int
        Maximum DFS recursion depth to guard stack.

    Returns
    -------
    List[CandidateRoute]
        List of enumerated candidate routes.
    """
    min_len_m = max(15.0, min_dist_ratio * dr_distance_m)
    max_len_m = max_dist_ratio * dr_distance_m

    routes: List[CandidateRoute] = []

    initial_len = entry_segment.length_m
    stack: List[Tuple[RoadSegment, List[RoadSegment], List[float], float, set]] = [
        (entry_segment, [entry_segment], [initial_len], initial_len, {entry_segment.segment_id})
    ]

    while stack and len(routes) < max_routes:
        curr_seg, path_segs, cum_lens, curr_len, visited = stack.pop()
        succs = succ_map.get(curr_seg.segment_id, [])

        if curr_len >= min_len_m:
            route = CandidateRoute(
                segments=list(path_segs),
                cum_lengths=list(cum_lens),
                total_length_m=curr_len,
            )
            route.turn_sequence = extract_turn_sequence_from_route(route)
            routes.append(route)

        if curr_len >= max_len_m or len(path_segs) >= max_depth:
            continue

        if not succs and curr_len < min_len_m and curr_len >= (0.5 * min_len_m):
            route = CandidateRoute(
                segments=list(path_segs),
                cum_lengths=list(cum_lens),
                total_length_m=curr_len,
            )
            route.turn_sequence = extract_turn_sequence_from_route(route)
            routes.append(route)
            continue

        for succ in succs:
            if succ.segment_id in visited:
                continue

            next_len = curr_len + succ.length_m
            if next_len <= max_len_m:
                new_visited = visited.copy()
                new_visited.add(succ.segment_id)
                stack.append((
                    succ,
                    path_segs + [succ],
                    cum_lens + [next_len],
                    next_len,
                    new_visited,
                ))

    return routes


def score_candidate_route(
    route: CandidateRoute,
    observed_seq: TurnSequence,
    dr_distance_m: float,
    dr_trajectory_enu: Optional[np.ndarray] = None,
) -> float:
    """
    Scores a candidate route against the observed blackout turn sequence.
    Aligns turns, penalizes turn and length mismatches, and adds a weak
    tiebreak based on mean perpendicular distance to the DR trajectory.

    Returns
    -------
    float
        Hypothesis likelihood score in range (0.0, 1.0].
    """
    obs_turns = observed_seq.turns
    route_turns = route.turn_sequence.turns

    n_obs = len(obs_turns)
    n_route = len(route_turns)

    turn_cost = 0.0
    if n_obs == 0 and n_route == 0:
        turn_cost = 0.0
    elif n_obs == 0 and n_route > 0:
        turn_cost = 3.5 * n_route
    elif n_obs > 0 and n_route == 0:
        turn_cost = 3.5 * n_obs
    else:
        matched_obs = set()
        matched_route = set()

        for i, o_turn in enumerate(obs_turns):
            best_j = None
            best_cost = 999.0

            for j, r_turn in enumerate(route_turns):
                if j in matched_route:
                    continue

                same_dir = (o_turn.delta_heading_deg * r_turn.delta_heading_deg) > 0
                if not same_dir:
                    continue

                h_diff = abs(o_turn.delta_heading_deg - r_turn.delta_heading_deg) / 30.0
                s_diff = abs(o_turn.arclength_m - r_turn.arclength_m) / max(25.0, 0.20 * dr_distance_m)
                pair_cost = h_diff + s_diff

                if pair_cost < best_cost:
                    best_cost = pair_cost
                    best_j = j

            if best_j is not None and best_cost < 3.0:
                turn_cost += best_cost
                matched_obs.add(i)
                matched_route.add(best_j)
            else:
                turn_cost += 3.5

        unmatched_route_count = n_route - len(matched_route)
        turn_cost += 3.5 * unmatched_route_count

    len_diff = abs(route.total_length_m - dr_distance_m)
    length_cost = len_diff / max(30.0, 0.25 * dr_distance_m)

    d_perp_cost = 0.0
    if dr_trajectory_enu is not None and len(dr_trajectory_enu) >= 3:
        indices = np.linspace(0, len(dr_trajectory_enu) - 1, min(8, len(dr_trajectory_enu)), dtype=int)
        sample_pts = dr_trajectory_enu[indices]

        d_perps = []
        for pt in sample_pts:
            min_d = min(seg.project_point(pt)[1] for seg in route.segments)
            d_perps.append(min_d)

        mean_d = float(np.mean(d_perps))
        d_perp_cost = 0.015 * min(60.0, mean_d)

    total_cost = turn_cost + 0.6 * length_cost + d_perp_cost
    score = float(np.exp(-total_cost))

    route.cost = total_cost
    route.turn_cost = turn_cost
    route.length_cost = length_cost
    route.d_perp_cost = d_perp_cost
    route.score = score

    return score


def project_by_arclength(
    route: CandidateRoute,
    arclengths_m: np.ndarray,
) -> np.ndarray:
    """
    Projects vehicle trajectory coordinates onto a winning CandidateRoute
    strictly by along-track arclength from integrated speed.

    Parameters
    ----------
    route : CandidateRoute
        The winning road route.
    arclengths_m : np.ndarray
        Array of cumulative along-track distances traveled at each timestamp.

    Returns
    -------
    np.ndarray
        (N, 2) array of ENU coordinates projected along the route.
    """
    n = len(arclengths_m)
    out_enu = np.zeros((n, 2), dtype=np.float64)

    segs = route.segments
    cum_lens = route.cum_lengths
    total_len = route.total_length_m
    seg_starts_m = [0.0] + cum_lens[:-1]

    for idx, s_val in enumerate(arclengths_m):
        s = float(s_val)

        if s <= 0.0:
            out_enu[idx] = segs[0].start_enu_m
            continue

        if s >= total_len:
            last_seg = segs[-1]
            overshoot = s - total_len
            v_dir = last_seg.end_enu_m - last_seg.start_enu_m
            v_norm = np.linalg.norm(v_dir)
            if v_norm > 1e-3:
                u_dir = v_dir / v_norm
                out_enu[idx] = last_seg.end_enu_m + overshoot * u_dir
            else:
                out_enu[idx] = last_seg.end_enu_m
            continue

        seg_idx = int(np.searchsorted(cum_lens, s))
        seg_idx = min(seg_idx, len(segs) - 1)

        s_seg = segs[seg_idx]
        s_start = seg_starts_m[seg_idx]
        s_len = max(1e-3, s_seg.length_m)

        frac = float(np.clip((s - s_start) / s_len, 0.0, 1.0))
        out_enu[idx] = (1.0 - frac) * s_seg.start_enu_m + frac * s_seg.end_enu_m

    return out_enu


class RouteMatcher:
    """
    High-level coordinator for route-level matching during GNSS blackouts.
    Executes depth-limited DFS enumeration, hypothesis scoring, confidence ratio gating,
    and arclength projection.
    """

    def __init__(
        self,
        min_route_ratio: float = 1.80,
        max_acceptable_cost: float = 8.0,
        min_dist_ratio: float = 0.70,
        max_dist_ratio: float = 1.40,
        max_routes: int = 200,
    ) -> None:
        self.min_route_ratio = min_route_ratio
        self.max_acceptable_cost = max_acceptable_cost
        self.min_dist_ratio = min_dist_ratio
        self.max_dist_ratio = max_dist_ratio
        self.max_routes = max_routes

    def match_blackout_route(
        self,
        entry_segment: Optional[RoadSegment],
        succ_map: Dict[str, List[RoadSegment]],
        timestamps_ns: np.ndarray,
        yaw_rates_rad_s: np.ndarray,
        speeds_mps: np.ndarray,
        dr_trajectory_enu: Optional[np.ndarray] = None,
    ) -> RouteMatchResult:
        """
        Executes full route-level hypothesis matching over the blackout interval.

        Returns
        -------
        RouteMatchResult
            Result containing winning route, confidence ratio, projected ENU trajectory,
            and timing diagnostics.
        """
        if entry_segment is None or not succ_map:
            return RouteMatchResult(
                won=False,
                confidence_ratio=0.0,
                winning_route=None,
                second_route=None,
                projected_pts_enu=None,
                total_routes_evaluated=0,
                enumeration_time_ms=0.0,
                scoring_time_ms=0.0,
                observed_turns_count=0,
                fallback_reason="No entry segment or empty succ_map",
            )

        # 1. Turn Sequence Extraction
        obs_seq = extract_turn_sequence_from_imu(
            timestamps_ns=timestamps_ns,
            yaw_rates_rad_s=yaw_rates_rad_s,
            speeds_mps=speeds_mps,
        )
        dr_distance_m = obs_seq.total_distance_m
        if dr_distance_m < 15.0:
            return RouteMatchResult(
                won=False,
                confidence_ratio=0.0,
                winning_route=None,
                second_route=None,
                projected_pts_enu=None,
                total_routes_evaluated=0,
                enumeration_time_ms=0.0,
                scoring_time_ms=0.0,
                observed_turns_count=len(obs_seq.turns),
                fallback_reason=f"Insufficient blackout distance ({dr_distance_m:.1f}m < 15m)",
            )

        # 2. Route Enumeration
        tracemalloc.start()
        t_enum_0 = time.perf_counter()
        routes = enumerate_routes_dfs(
            entry_segment=entry_segment,
            succ_map=succ_map,
            dr_distance_m=dr_distance_m,
            min_dist_ratio=self.min_dist_ratio,
            max_dist_ratio=self.max_dist_ratio,
            max_routes=self.max_routes,
        )
        enum_time_ms = (time.perf_counter() - t_enum_0) * 1000.0

        if not routes:
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return RouteMatchResult(
                won=False,
                confidence_ratio=0.0,
                winning_route=None,
                second_route=None,
                projected_pts_enu=None,
                total_routes_evaluated=0,
                enumeration_time_ms=enum_time_ms,
                scoring_time_ms=0.0,
                observed_turns_count=len(obs_seq.turns),
                fallback_reason="No topological routes enumerated within length bounds",
                peak_memory_kb=peak_mem / 1024.0,
            )

        # 3. Route Scoring
        t_score_0 = time.perf_counter()
        for r in routes:
            score_candidate_route(r, obs_seq, dr_distance_m, dr_trajectory_enu=dr_trajectory_enu)

        routes.sort(key=lambda r: r.score, reverse=True)
        score_time_ms = (time.perf_counter() - t_score_0) * 1000.0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_kb = peak_mem / 1024.0

        best_route = routes[0]
        second_route = routes[1] if len(routes) > 1 else None

        # 4. Confidence Ratio Evaluation (Computed strictly in log space to prevent underflow)
        # Scores are exp(-cost). When total_cost > 27.63, scores underflow IEEE double (< 1e-12).
        # In log space, score_1 / score_2 = exp(cost_2 - cost_1).
        if second_route is not None:
            cost_diff = second_route.cost - best_route.cost
            ratio = float(np.exp(min(50.0, max(-50.0, cost_diff))))
        else:
            ratio = 3.0 if best_route.cost <= self.max_acceptable_cost else 1.0

        # Absolute cost gate: A route must have low absolute penalty (good physical fit)
        # in addition to beating the second-best route by ratio >= min_route_ratio.
        # This prevents declaring false-positive "confident wins" on high-cost bad routes (e.g. Scenario 25).
        cost_acceptable = (best_route.cost <= self.max_acceptable_cost)
        won = (ratio >= self.min_route_ratio) and cost_acceptable

        # 5. Arclength Projection (if winning)
        projected_enu = None
        if won:
            dt_s = np.diff(timestamps_ns) * 1e-9
            dt_s = np.clip(dt_s, 1e-4, 1.0)
            arclengths = np.zeros(len(timestamps_ns), dtype=np.float64)
            for k in range(1, len(timestamps_ns)):
                avg_v = 0.5 * (speeds_mps[k - 1] + speeds_mps[k])
                arclengths[k] = arclengths[k - 1] + max(0.0, avg_v) * dt_s[k - 1]

            projected_enu = project_by_arclength(best_route, arclengths)

        if won:
            fallback_msg = ""
        elif not cost_acceptable:
            fallback_msg = f"Best route cost {best_route.cost:.2f} > max acceptable {self.max_acceptable_cost:.2f}"
        else:
            fallback_msg = f"Confidence ratio {ratio:.2f} < threshold {self.min_route_ratio:.2f}"

        return RouteMatchResult(
            won=won,
            confidence_ratio=float(ratio),
            winning_route=best_route,
            second_route=second_route,
            projected_pts_enu=projected_enu,
            total_routes_evaluated=len(routes),
            enumeration_time_ms=enum_time_ms,
            scoring_time_ms=score_time_ms,
            observed_turns_count=len(obs_seq.turns),
            fallback_reason=fallback_msg,
            peak_memory_kb=peak_memory_kb,
        )
