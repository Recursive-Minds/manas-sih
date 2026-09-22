"""
server/evaluator.py
-------------------
Live Ground-Truth Dead Reckoning Evaluator.

Runs alongside the dead-reckoning engine to compare real-time estimated
trajectories against ground-truth GNSS fixes (which are retained on the
server for validation while hidden from the dead-reckoning engine).

Calculates in real time:
- Blackout elapsed duration (s)
- Cumulative DR distance (m) & GNSS distance (m)
- Instantaneous Euclidean horizontal error (m)
- Along-track (speed scale) and Cross-track (heading) error decomposition (m)
- Real-time drift percentage: error_m / max(gnss_dist_m, 1.0) * 100
- Compliance tier classification: Tier-1 (<10%), Tier-2 (<20%), Tier-3 (>=20%)
- Speed & heading tracking comparison
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from sih.core.contracts import GNSSSample, FusedPosition
from sih.data.geo import geodetic_to_enu


@dataclass
class TrajectoryPoint:
    lat: float
    lon: float
    timestamp_ns: int
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    error_m: Optional[float] = None
    drift_pct: Optional[float] = None


def classify_speed_regime(speed_kmh: float) -> str:
    """
    Classifies SIH speed regimes by mean vehicle velocity:
    - Crawl (< 20 km/h)
    - City (20 - 50 km/h)
    - Highway (> 50 km/h)
    """
    if speed_kmh < 20.0:
        return "Crawl (<20 km/h)"
    elif speed_kmh <= 50.0:
        return "City (20-50 km/h)"
    else:
        return "Highway (>50 km/h)"


@dataclass
class LiveMetrics:
    elapsed_s: float = 0.0
    dr_dist_m: float = 0.0
    gnss_dist_m: float = 0.0
    horizontal_error_m: float = 0.0
    along_track_m: float = 0.0
    cross_track_m: float = 0.0
    drift_pct: float = 0.0
    speed_regime: str = "City (20-50 km/h)"
    drift_label: str = "Drift: 0.00% (target <10%)"
    target_met: bool = True
    tier: str = "City (20-50 km/h)"
    dr_speed_mps: float = 0.0
    gnss_speed_mps: float = 0.0
    dr_heading_deg: float = 0.0
    gnss_bearing_deg: float = 0.0
    max_horizontal_error_m: float = 0.0
    gnss_accuracy_h_m: float = 5.0


class LiveEvaluator:
    """
    Stateful real-time evaluation comparator.
    """
    def __init__(self, ref_lat: float = 0.0, ref_lon: float = 0.0, ref_alt: float = 0.0) -> None:
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.has_ref: bool = (ref_lat != 0.0 or ref_lon != 0.0)

        # Full chronological log of GNSS fixes
        self.gnss_fixes: List[GNSSSample] = []
        # Full chronological log of DR fused positions
        self.dr_fused: List[FusedPosition] = []

        # Blackout state
        self.is_blackout: bool = False
        self.blackout_start_ns: Optional[int] = None
        self.blackout_start_enu: Optional[np.ndarray] = None

        # Cumulative distance tracking during blackout
        self.last_dr_enu: Optional[np.ndarray] = None
        self.last_gnss_enu: Optional[np.ndarray] = None
        self.last_interpolated_gt: Optional[np.ndarray] = None
        self.gnss_enu_history: List[Tuple[int, float, float, float, float]] = []
        self.cum_dr_dist_m: float = 0.0
        self.cum_gnss_dist_m: float = 0.0
        self.max_horizontal_error_m: float = 0.0
        self.latest_gnss_accuracy_m: float = 5.0

        # Latest metrics snapshot
        self.latest_metrics = LiveMetrics()
        self.last_completed_summary: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self.ref_alt = 0.0
        self.has_ref = False
        self.gnss_fixes.clear()
        self.gnss_enu_history.clear()
        self.dr_fused.clear()
        self.is_blackout = False
        self.blackout_start_ns = None
        self.blackout_start_enu = None
        self.last_dr_enu = None
        self.last_gnss_enu = None
        self.last_interpolated_gt = None
        self.cum_dr_dist_m = 0.0
        self.cum_gnss_dist_m = 0.0
        self.max_horizontal_error_m = 0.0
        self.latest_gnss_accuracy_m = 5.0
        self.latest_metrics = LiveMetrics()
        self.last_completed_summary = None

    def start_blackout(self, timestamp_ns: Optional[int] = None) -> None:
        """Triggers the start of GNSS blackout evaluation."""
        self.is_blackout = True
        self.blackout_start_ns = timestamp_ns
        self.cum_dr_dist_m = 0.0
        self.cum_gnss_dist_m = 0.0
        self.max_horizontal_error_m = 0.0
        self.last_interpolated_gt = None

        if self.dr_fused:
            self.last_dr_enu = self.dr_fused[-1].position_enu_m.copy()
            self.blackout_start_enu = self.dr_fused[-1].position_enu_m.copy()
            if self.blackout_start_ns is None:
                self.blackout_start_ns = self.dr_fused[-1].timestamp_ns
        elif self.gnss_fixes and self.has_ref:
            g = self.gnss_fixes[-1]
            enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, g.altitude_m or 0.0, self.ref_lat, self.ref_lon, self.ref_alt)
            self.last_gnss_enu = enu.copy()
            self.blackout_start_enu = enu.copy()
            if self.blackout_start_ns is None:
                self.blackout_start_ns = g.timestamp_ns

        if self.blackout_start_ns is not None:
            gt_start = self._interpolate_gnss_at(self.blackout_start_ns)
            if gt_start is not None:
                self.last_interpolated_gt = np.array([gt_start[0], gt_start[1]], dtype=np.float64)

    def stop_blackout(self, timestamp_ns: Optional[int] = None) -> None:
        """
        Finalizes GNSS blackout evaluation.
        Computes ground truth distance and endpoint error EXACTLY matching the benchmark:
        1. gt_dist = sum of Euclidean distances between valid blackout GNSS fixes.
        2. Endpoint error evaluated by interpolating DR at the timestamp of the last valid GNSS fix.
        3. Drift % = (final_err / gt_dist) * 100.0.
        4. Along-track and cross-track error decomposition using exit track heading.
        """
        if not self.is_blackout:
            return

        bo_start_ns = self.blackout_start_ns
        bo_end_ns = timestamp_ns or (self.dr_fused[-1].timestamp_ns if self.dr_fused else None)

        # 1. Collect valid GNSS fixes strictly inside the blackout window
        bo_gnss: List[GNSSSample] = []
        if bo_start_ns is not None:
            bo_gnss = [
                g for g in self.gnss_fixes
                if g.is_valid and g.timestamp_ns >= bo_start_ns and (bo_end_ns is None or g.timestamp_ns <= bo_end_ns)
            ]

        # 2. Compute ground-truth distance and endpoint
        gt_dist = 0.0
        gt_end_enu = None
        eval_ts = None
        last_g = None
        gt_pts = None

        if len(bo_gnss) >= 2 and self.has_ref:
            gt_pts = np.array([
                geodetic_to_enu(g.latitude_deg, g.longitude_deg, g.altitude_m or 0.0, self.ref_lat, self.ref_lon, self.ref_alt)[:2]
                for g in bo_gnss
            ])
            gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1)))
            gt_end_enu = gt_pts[-1]
            eval_ts = bo_gnss[-1].timestamp_ns
            last_g = bo_gnss[-1]
        elif self.cum_gnss_dist_m > 0:
            gt_dist = self.cum_gnss_dist_m

        if gt_dist <= 0.0:
            gt_dist = max(self.cum_dr_dist_m, 1.0)

        # 3. Interpolate DR position at the timestamp of the last valid blackout GNSS fix
        final_err = self.latest_metrics.horizontal_error_m
        along_track = self.latest_metrics.along_track_m
        cross_track = self.latest_metrics.cross_track_m
        eval_pt = None

        if eval_ts is not None and len(self.dr_fused) >= 2 and bo_start_ns is not None:
            bo_dr = [f for f in self.dr_fused if f.timestamp_ns >= bo_start_ns]
            if len(bo_dr) >= 2:
                dr_ts = np.array([f.timestamp_ns for f in bo_dr], dtype=np.float64)
                dr_e = np.array([f.position_enu_m[0] for f in bo_dr], dtype=np.float64)
                dr_n = np.array([f.position_enu_m[1] for f in bo_dr], dtype=np.float64)

                eval_e = float(np.interp(eval_ts, dr_ts, dr_e))
                eval_n = float(np.interp(eval_ts, dr_ts, dr_n))
                eval_pt = np.array([eval_e, eval_n])

                if gt_end_enu is not None:
                    final_err = float(np.linalg.norm(eval_pt - gt_end_enu))

                    # Track tangent and normal at outage exit
                    if last_g is not None and last_g.bearing_deg is not None and (last_g.speed_mps or 0.0) > 0.5:
                        b_rad = math.radians(last_g.bearing_deg)
                        te_end = math.sin(b_rad)
                        tn_end = math.cos(b_rad)
                    elif gt_pts is not None and len(gt_pts) >= 2:
                        d_end = gt_pts[-1] - gt_pts[-2]
                        d_norm = float(np.linalg.norm(d_end))
                        if d_norm > 1e-4:
                            te_end = float(d_end[0] / d_norm)
                            tn_end = float(d_end[1] / d_norm)
                        else:
                            te_end, tn_end = 1.0, 0.0
                    else:
                        te_end, tn_end = 1.0, 0.0

                    ne_end = -tn_end
                    nn_end = te_end
                    err_vec = eval_pt - gt_end_enu
                    along_track = float(err_vec[0] * te_end + err_vec[1] * tn_end)
                    cross_track = float(err_vec[0] * ne_end + err_vec[1] * nn_end)

        final_drift_pct = (final_err / max(gt_dist, 1.0)) * 100.0

        self.last_completed_summary = {
            "final_error_m": round(final_err, 2),
            "drift_pct": round(final_drift_pct, 2),
            "drift_label": f"Drift: {final_drift_pct:.2f}% (target <10%)",
            "target_met": bool(final_drift_pct < 10.0),
            "max_error_m": round(self.max_horizontal_error_m, 2),
            "duration_s": round(self.latest_metrics.elapsed_s, 1),
            "dr_dist_m": round(self.cum_dr_dist_m, 1),
            "gnss_dist_m": round(gt_dist, 1),
            "along_track_m": round(along_track, 1),
            "cross_track_m": round(cross_track, 1),
            "speed_regime": self.latest_metrics.speed_regime,
            "tier": self.latest_metrics.speed_regime,
        }
        self.is_blackout = False

    def on_gnss(self, gnss: GNSSSample) -> None:
        """Receives ground truth GNSS sample."""
        if not self.has_ref:
            self.ref_lat = gnss.latitude_deg
            self.ref_lon = gnss.longitude_deg
            self.ref_alt = gnss.altitude_m or 0.0
            self.has_ref = True

        self.gnss_fixes.append(gnss)
        self.latest_gnss_accuracy_m = gnss.accuracy_h_m or 5.0

        if self.has_ref:
            enu = geodetic_to_enu(
                gnss.latitude_deg,
                gnss.longitude_deg,
                gnss.altitude_m or 0.0,
                self.ref_lat,
                self.ref_lon,
                self.ref_alt,
            )
            spd = float(gnss.speed_mps) if gnss.speed_mps is not None else 0.0
            brg = float(gnss.bearing_deg) if gnss.bearing_deg is not None else 0.0
            self.gnss_enu_history.append((gnss.timestamp_ns, float(enu[0]), float(enu[1]), spd, brg))
            self.gnss_enu_history.sort(key=lambda x: x[0])

    def _interpolate_gnss_at(self, t_ns: int) -> Optional[Tuple[float, float, float, float]]:
        """Returns (east_m, north_m, speed_mps, bearing_deg) interpolated at t_ns."""
        if not self.gnss_enu_history:
            return None
        if len(self.gnss_enu_history) == 1:
            h = self.gnss_enu_history[0]
            return (h[1], h[2], h[3], h[4])

        if t_ns <= self.gnss_enu_history[0][0]:
            h = self.gnss_enu_history[0]
            return (h[1], h[2], h[3], h[4])

        if t_ns >= self.gnss_enu_history[-1][0]:
            last = self.gnss_enu_history[-1]
            dt = (t_ns - last[0]) * 1e-9
            if dt < 20.0 and last[3] > 0.1:
                b_rad = math.radians(last[4])
                e = last[1] + dt * last[3] * math.sin(b_rad)
                n = last[2] + dt * last[3] * math.cos(b_rad)
                return (e, n, last[3], last[4])
            return (last[1], last[2], last[3], last[4])

        # Find interval
        for i in range(len(self.gnss_enu_history) - 2, -1, -1):
            t0, e0, n0, s0, b0 = self.gnss_enu_history[i]
            t1, e1, n1, s1, b1 = self.gnss_enu_history[i + 1]
            if t0 <= t_ns <= t1:
                dt_total = t1 - t0
                if dt_total <= 0:
                    return (e1, n1, s1, b1)
                alpha = (t_ns - t0) / dt_total
                e = (1.0 - alpha) * e0 + alpha * e1
                n = (1.0 - alpha) * n0 + alpha * n1
                s = (1.0 - alpha) * s0 + alpha * s1
                de = e1 - e0
                dn = n1 - n0
                if math.hypot(de, dn) > 0.1:
                    b = (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0
                else:
                    b = b1
                return (e, n, s, b)

        last = self.gnss_enu_history[-1]
        return (last[1], last[2], last[3], last[4])

    def on_dr(self, fused: FusedPosition) -> LiveMetrics:
        """
        Receives dead reckoning position estimate and computes live comparison metrics.
        """
        if not self.has_ref and fused.latitude_deg != 0.0:
            self.ref_lat = fused.latitude_deg
            self.ref_lon = fused.longitude_deg
            self.ref_alt = fused.altitude_m
            self.has_ref = True

        self.dr_fused.append(fused)

        dr_enu = fused.position_enu_m
        dr_speed = float(np.linalg.norm(fused.velocity_enu_mps[:2]))
        dr_heading = float(math.degrees(fused.heading_rad) % 360.0)

        # Update DR distance during blackout
        if self.is_blackout:
            if self.last_dr_enu is not None:
                step_d = float(np.linalg.norm(dr_enu[:2] - self.last_dr_enu[:2]))
                self.cum_dr_dist_m += step_d
            else:
                self.last_dr_enu = dr_enu.copy()
                if self.blackout_start_enu is None:
                    self.blackout_start_enu = dr_enu.copy()
            self.last_dr_enu = dr_enu.copy()

        # Compute error against time-interpolated ground truth GNSS
        elapsed_s = 0.0
        if self.is_blackout and self.blackout_start_ns is not None:
            elapsed_s = max(0.0, (fused.timestamp_ns - self.blackout_start_ns) * 1e-9)

        horiz_err = 0.0
        along_track = 0.0
        cross_track = 0.0
        gnss_spd = 0.0
        gnss_brg = 0.0

        gt_interp = self._interpolate_gnss_at(fused.timestamp_ns)
        if gt_interp is not None:
            g_east, g_north, gnss_spd, gnss_brg = gt_interp

            # Update continuous interpolated GNSS distance
            if self.is_blackout:
                curr_gt_pt = np.array([g_east, g_north], dtype=np.float64)
                if self.last_interpolated_gt is not None:
                    step_gt_d = float(np.linalg.norm(curr_gt_pt - self.last_interpolated_gt))
                    self.cum_gnss_dist_m += step_gt_d
                self.last_interpolated_gt = curr_gt_pt.copy()

            # Error vector in ENU meters (east, north)
            diff_e = float(dr_enu[0] - g_east)
            diff_n = float(dr_enu[1] - g_north)
            horiz_err = float(math.sqrt(diff_e**2 + diff_n**2))

            # Along-track and cross-track decomposition
            bearing_rad = math.radians(gnss_brg)
            t_e, t_n = math.sin(bearing_rad), math.cos(bearing_rad)
            n_e, n_n = math.cos(bearing_rad), -math.sin(bearing_rad)

            along_track = diff_e * t_e + diff_n * t_n
            cross_track = diff_e * n_e + diff_n * n_n

        if self.is_blackout and horiz_err > self.max_horizontal_error_m:
            self.max_horizontal_error_m = horiz_err

        if self.cum_gnss_dist_m >= 15.0:
            dist_ref = self.cum_gnss_dist_m
        else:
            dist_ref = max(self.cum_dr_dist_m, 1.0)
        drift_pct = (horiz_err / dist_ref) * 100.0 if self.is_blackout else 0.0

        # SIH speed regimes by mean speed
        mean_speed_mps = (self.cum_gnss_dist_m / elapsed_s) if elapsed_s > 0.1 else dr_speed
        mean_speed_kmh = mean_speed_mps * 3.6
        speed_regime = classify_speed_regime(mean_speed_kmh)
        drift_label = f"Drift: {drift_pct:.2f}% (target <10%)"
        target_met = bool(drift_pct < 10.0)

        self.latest_metrics = LiveMetrics(
            elapsed_s=round(elapsed_s, 2),
            dr_dist_m=round(self.cum_dr_dist_m, 2),
            gnss_dist_m=round(self.cum_gnss_dist_m, 2),
            horizontal_error_m=round(horiz_err, 2),
            along_track_m=round(along_track, 2),
            cross_track_m=round(cross_track, 2),
            drift_pct=round(drift_pct, 2),
            speed_regime=speed_regime,
            drift_label=drift_label,
            target_met=target_met,
            tier=speed_regime,
            dr_speed_mps=round(dr_speed, 2),
            gnss_speed_mps=round(gnss_spd, 2),
            dr_heading_deg=round(dr_heading, 1),
            gnss_bearing_deg=round(gnss_brg, 1),
            max_horizontal_error_m=round(self.max_horizontal_error_m, 2),
            gnss_accuracy_h_m=round(self.latest_gnss_accuracy_m, 1),
        )
        return self.latest_metrics

    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        if self.last_completed_summary is not None:
            return self.last_completed_summary
        return {
            "final_error_m": self.latest_metrics.horizontal_error_m,
            "drift_pct": self.latest_metrics.drift_pct,
            "drift_label": self.latest_metrics.drift_label,
            "target_met": self.latest_metrics.target_met,
            "max_error_m": round(self.max_horizontal_error_m, 2),
            "duration_s": self.latest_metrics.elapsed_s,
            "dr_dist_m": self.latest_metrics.dr_dist_m,
            "gnss_dist_m": self.latest_metrics.gnss_dist_m,
            "along_track_m": self.latest_metrics.along_track_m,
            "cross_track_m": self.latest_metrics.cross_track_m,
            "speed_regime": self.latest_metrics.speed_regime,
            "tier": self.latest_metrics.speed_regime,
        }

    def get_summary_dict(self) -> Dict[str, Any]:
        if not self.is_blackout and self.last_completed_summary is not None:
            s = self.last_completed_summary
            return {
                "elapsed_s": s.get("duration_s", self.latest_metrics.elapsed_s),
                "dr_dist_m": s.get("dr_dist_m", self.latest_metrics.dr_dist_m),
                "gnss_dist_m": s.get("gnss_dist_m", self.latest_metrics.gnss_dist_m),
                "horizontal_error_m": s.get("final_error_m", self.latest_metrics.horizontal_error_m),
                "along_track_m": s.get("along_track_m", self.latest_metrics.along_track_m),
                "cross_track_m": s.get("cross_track_m", self.latest_metrics.cross_track_m),
                "drift_pct": s.get("drift_pct", self.latest_metrics.drift_pct),
                "speed_regime": s.get("speed_regime", self.latest_metrics.speed_regime),
                "drift_label": s.get("drift_label", self.latest_metrics.drift_label),
                "target_met": s.get("target_met", self.latest_metrics.target_met),
                "tier": s.get("tier", self.latest_metrics.tier),
                "dr_speed_mps": self.latest_metrics.dr_speed_mps,
                "gnss_speed_mps": self.latest_metrics.gnss_speed_mps,
                "dr_heading_deg": self.latest_metrics.dr_heading_deg,
                "gnss_bearing_deg": self.latest_metrics.gnss_bearing_deg,
                "max_horizontal_error_m": s.get("max_error_m", self.latest_metrics.max_horizontal_error_m),
                "gnss_accuracy_h_m": self.latest_metrics.gnss_accuracy_h_m,
                "is_blackout": False,
                "session_summary": s,
            }

        return {
            "elapsed_s": self.latest_metrics.elapsed_s,
            "dr_dist_m": self.latest_metrics.dr_dist_m,
            "gnss_dist_m": self.latest_metrics.gnss_dist_m,
            "horizontal_error_m": self.latest_metrics.horizontal_error_m,
            "along_track_m": self.latest_metrics.along_track_m,
            "cross_track_m": self.latest_metrics.cross_track_m,
            "drift_pct": self.latest_metrics.drift_pct,
            "speed_regime": self.latest_metrics.speed_regime,
            "drift_label": self.latest_metrics.drift_label,
            "target_met": self.latest_metrics.target_met,
            "tier": self.latest_metrics.tier,
            "dr_speed_mps": self.latest_metrics.dr_speed_mps,
            "gnss_speed_mps": self.latest_metrics.gnss_speed_mps,
            "dr_heading_deg": self.latest_metrics.dr_heading_deg,
            "gnss_bearing_deg": self.latest_metrics.gnss_bearing_deg,
            "max_horizontal_error_m": self.latest_metrics.max_horizontal_error_m,
            "gnss_accuracy_h_m": self.latest_metrics.gnss_accuracy_h_m,
            "is_blackout": self.is_blackout,
            "session_summary": None,
        }
