"""
T8 - Turn-at-junction along-track anchoring.

87.9 % of the squared end-point error is along-track (speed/distance). The map
matcher fixes cross-track but has no information about WHERE along the road we
are. A completed turn does: a 90 deg turn can only have happened at a corner of
the road graph. So when the gyro reports a completed turn we

  1. find road corners near the turn apex: pairs of road lines whose directions
     match the pre-turn heading and the post-turn heading, and whose lines meet
     (the intersection C lies on/near both segments);
  2. if exactly one corner is plausible (2nd best clearly farther), the vehicle
     position NOW should be  C + (d_since_apex + R*(tan(theta/2) - theta/2)) * u_out
     (second term: a turn of radius R and angle theta cuts the corner)
     (d_since_apex = distance driven since the turn apex, u_out = exit direction);
  3. correct only the along-track component, with a gain < 1:
       p <- p + gain * ((C + d * u_out - p) . u_out) * u_out

Headings are clockwise from north (as in ErrorStateEKF); a positive yaw rate
(counter-clockwise) decreases the heading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sih.round1.config import JunctionAnchorParams


def _unit_from_bearing(b_deg: float) -> np.ndarray:
    r = np.radians(b_deg)
    return np.array([np.sin(r), np.cos(r)], dtype=np.float64)


def _axis_diff_deg(a: float, b: float) -> float:
    """Angle between two undirected lines, 0..90."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _line_intersection(p1, d1, p2, d2) -> Optional[np.ndarray]:
    m = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    det = np.linalg.det(m)
    if abs(det) < 1e-6:
        return None
    t = np.linalg.solve(m, p2 - p1)
    return p1 + t[0] * d1


def _dist_point_segment(p, a, b) -> float:
    ab = b - a
    L = float(np.dot(ab, ab))
    if L < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / L, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def find_corners(segments: List[Any], h_in_deg: float, h_out_deg: float, apex: np.ndarray,
                 radius_m: float, tol_deg: float, meet_tol_m: float = 35.0,
                 min_line_len_m: float = 12.0, cluster_m: float = 15.0) -> List[Tuple[float, np.ndarray]]:
    """Distinct road corners near `apex`, nearest first. Short pieces of a drawn curve are not
    used as lines; corners closer than cluster_m are one junction (longest supporting lines win)."""
    def _len(s):
        return float(np.linalg.norm(np.asarray(s.end_enu_m, float) - np.asarray(s.start_enu_m, float)))
    ins = [s for s in segments if _axis_diff_deg(s.bearing_deg, h_in_deg) <= tol_deg and _len(s) >= min_line_len_m]
    outs = [s for s in segments if _axis_diff_deg(s.bearing_deg, h_out_deg) <= tol_deg and _len(s) >= min_line_len_m]
    raw: List[Tuple[float, float, np.ndarray]] = []
    for si in ins:
        a1, b1 = np.asarray(si.start_enu_m, float), np.asarray(si.end_enu_m, float)
        for so in outs:
            if so is si:
                continue
            a2, b2 = np.asarray(so.start_enu_m, float), np.asarray(so.end_enu_m, float)
            d1, d2 = b1 - a1, b2 - a2
            if np.linalg.norm(d1) < 1e-3 or np.linalg.norm(d2) < 1e-3:
                continue
            if _axis_diff_deg(si.bearing_deg, so.bearing_deg) < 30.0:
                continue
            c = _line_intersection(a1, d1, a2, d2)
            if c is None:
                continue
            if _dist_point_segment(c, a1, b1) > meet_tol_m or _dist_point_segment(c, a2, b2) > meet_tol_m:
                continue
            d = float(np.linalg.norm(c - apex))
            if d > radius_m:
                continue
            raw.append((min(_len(si), _len(so)), d, c))
    raw.sort(key=lambda x: -x[0])                     # strongest support first
    corners: List[Tuple[float, np.ndarray]] = []
    for _, d, c in raw:
        if any(np.linalg.norm(c - c0) < cluster_m for _, c0 in corners):
            continue
        corners.append((d, c))
    corners.sort(key=lambda x: x[0])
    return corners


class TurnJunctionAnchor:
    def __init__(self, p: JunctionAnchorParams):
        self.p = p
        self.dist_since_entry = 0.0
        self.events: List[Dict[str, Any]] = []
        self._reset_turn()

    def _reset_turn(self) -> None:
        self.in_turn = False
        self._psi = 0.0
        self._dur = 0.0
        self._quiet = 0.0
        self._h_in = 0.0
        self._trace: List[Tuple[float, np.ndarray, float]] = []
        self._v_sum = 0.0

    def update(self, w_z_corr: float, v_fwd: float, dt: float, ekf: Any, road_network: Any) -> Optional[Dict[str, Any]]:
        p = self.p
        self.dist_since_entry += max(0.0, v_fwd) * dt
        heading_deg = float(np.degrees(ekf._heading_rad)) % 360.0

        if not self.in_turn:
            if abs(w_z_corr) >= p.turn_on_rad_s:
                self.in_turn = True
                self._h_in = (heading_deg - np.degrees(-w_z_corr * dt)) % 360.0
            else:
                return None

        self._psi += -np.degrees(w_z_corr * dt)
        self._dur += dt
        self._v_sum += v_fwd * dt
        self._trace.append((self._psi, ekf._p[:2].copy(), self.dist_since_entry))
        self._quiet = self._quiet + dt if abs(w_z_corr) < p.turn_off_rad_s else 0.0

        if self._dur > p.max_turn_duration_s + 5.0:
            self._reset_turn()
            return None
        if self._quiet < p.t_quiet_s:
            return None

        # turn closed
        psi, dur, v_mean, trace, h_in = self._psi, self._dur, self._v_sum / max(self._dur, 1e-3), self._trace, self._h_in
        self._reset_turn()
        ev = {"psi_deg": psi, "dur_s": dur, "v_mean": v_mean, "applied": False}
        if not (p.min_turn_deg <= abs(psi) <= p.max_turn_deg) or dur > p.max_turn_duration_s or v_mean > p.max_speed_mps:
            ev["reason"] = "not a junction turn"
            self.events.append(ev)
            return ev
        if road_network is None:
            ev["reason"] = "no road network"
            self.events.append(ev)
            return ev

        half = 0.5 * abs(psi)
        apex_psi, apex_pos, apex_dist = next(t for t in trace if abs(t[0]) >= half)
        h_out = heading_deg
        radius = min(p.max_radius_m, p.base_radius_m + p.radius_frac * self.dist_since_entry)
        segs = road_network.find_candidates(apex_pos, radius_m=radius)
        corners = find_corners(segs, h_in, h_out, apex_pos, radius, p.line_tol_deg)
        ev.update({"n_corners": len(corners), "radius_m": radius})
        if not corners:
            ev["reason"] = "no corner"
            self.events.append(ev)
            return ev
        if len(corners) >= 2 and corners[1][0] < p.ambiguity_ratio * max(corners[0][0], 5.0):
            ev["reason"] = "ambiguous"
            self.events.append(ev)
            return ev

        c = corners[0][1]
        u_out = _unit_from_bearing(h_out)
        d_since_apex = self.dist_since_entry - apex_dist
        # arc vs corner geometry: from the apex the arc to the exit tangent point is R*theta/2,
        # the straight-line distance from corner C to that tangent point is R*tan(theta/2)
        theta = np.radians(abs(psi))
        r_turn = (v_mean * max(dur - p.t_quiet_s, 0.5)) / max(theta, 1e-3)
        arc_offset = r_turn * (np.tan(0.5 * theta) - 0.5 * theta) if theta < np.radians(150.0) else 0.0
        expected = c + (d_since_apex + arc_offset) * u_out
        cur = ekf._p[:2].copy()
        delta_along = float(np.dot(expected - cur, u_out))
        ev.update({"delta_along_m": delta_along, "corner_dist_m": corners[0][0]})
        if abs(delta_along) > p.max_correction_m:
            ev["reason"] = "correction too large"
            self.events.append(ev)
            return ev
        ekf._p[0] += p.gain * delta_along * u_out[0]
        ekf._p[1] += p.gain * delta_along * u_out[1]
        ev["applied"] = True
        self.events.append(ev)
        return ev
