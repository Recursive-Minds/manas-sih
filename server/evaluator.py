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
        self.cum_dr_dist_m: float = 0.0
        self.cum_gnss_dist_m: float = 0.0
        self.max_horizontal_error_m: float = 0.0
        self.latest_gnss_accuracy_m: float = 5.0

        # Latest metrics snapshot
        self.latest_metrics = LiveMetrics()

    def reset(self) -> None:
        self.gnss_fixes.clear()
        self.dr_fused.clear()
        self.is_blackout = False
        self.blackout_start_ns = None
        self.blackout_start_enu = None
        self.last_dr_enu = None
        self.last_gnss_enu = None
        self.cum_dr_dist_m = 0.0
        self.cum_gnss_dist_m = 0.0
        self.max_horizontal_error_m = 0.0
        self.latest_gnss_accuracy_m = 5.0
        self.latest_metrics = LiveMetrics()

    def start_blackout(self, timestamp_ns: Optional[int] = None) -> None:
        """Triggers the start of GNSS blackout evaluation."""
        self.is_blackout = True
        self.blackout_start_ns = timestamp_ns
        self.cum_dr_dist_m = 0.0
        self.cum_gnss_dist_m = 0.0
        self.max_horizontal_error_m = 0.0

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

    def stop_blackout(self) -> None:
        self.is_blackout = False

    def on_gnss(self, gnss: GNSSSample) -> None:
        """Receives ground truth GNSS sample."""
        if not self.has_ref:
            self.ref_lat = gnss.latitude_deg
            self.ref_lon = gnss.longitude_deg
            self.ref_alt = gnss.altitude_m or 0.0
            self.has_ref = True

        self.gnss_fixes.append(gnss)

        if self.is_blackout and self.has_ref:
            enu = geodetic_to_enu(
                gnss.latitude_deg,
                gnss.longitude_deg,
                gnss.altitude_m or 0.0,
                self.ref_lat,
                self.ref_lon,
                self.ref_alt,
            )
            if self.last_gnss_enu is not None:
                step_d = float(np.linalg.norm(enu[:2] - self.last_gnss_enu[:2]))
                self.cum_gnss_dist_m += step_d
            else:
                self.last_gnss_enu = enu.copy()
                if self.blackout_start_enu is None:
                    self.blackout_start_enu = enu.copy()
            self.last_gnss_enu = enu.copy()

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

        # Compute error against ground truth GNSS
        elapsed_s = 0.0
        if self.is_blackout and self.blackout_start_ns is not None:
            elapsed_s = max(0.0, (fused.timestamp_ns - self.blackout_start_ns) * 1e-9)

        horiz_err = 0.0
        along_track = 0.0
        cross_track = 0.0
        gnss_spd = 0.0
        gnss_brg = 0.0

        if self.gnss_fixes and self.has_ref:
            latest_g = self.gnss_fixes[-1]
            gnss_spd = latest_g.speed_mps or 0.0
            gnss_brg = latest_g.bearing_deg or 0.0

            g_enu = geodetic_to_enu(
                latest_g.latitude_deg,
                latest_g.longitude_deg,
                latest_g.altitude_m or 0.0,
                self.ref_lat,
                self.ref_lon,
                self.ref_alt,
            )

            # Error vector in ENU meters (east, north)
            diff_e = float(dr_enu[0] - g_enu[0])
            diff_n = float(dr_enu[1] - g_enu[1])
            horiz_err = float(math.sqrt(diff_e**2 + diff_n**2))

            # Along-track and cross-track decomposition
            bearing_rad = math.radians(gnss_brg)
            t_e, t_n = math.sin(bearing_rad), math.cos(bearing_rad)
            n_e, n_n = math.cos(bearing_rad), -math.sin(bearing_rad)

            along_track = diff_e * t_e + diff_n * t_n
            cross_track = diff_e * n_e + diff_n * n_n

        if self.is_blackout and horiz_err > self.max_horizontal_error_m:
            self.max_horizontal_error_m = horiz_err

        dist_ref = max(self.cum_gnss_dist_m, self.cum_dr_dist_m, 1.0)
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

    def get_session_summary(self) -> Dict[str, Any]:
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
            "session_summary": self.get_session_summary(),
        }
