"""
Dead Reckoning Execution Engine.

Encapsulates complete scenario simulation, pre-blackout filter conditioning,
ES-EKF propagation, Road Kinematics governing, HMM map matching, and metric computation.
Decoupled from benchmark reporting and plotting scripts.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from sih.fusion.es_ekf import ErrorStateEKF
from sih.map.network import RoadNetwork, RoadSegment
from sih.map.governor import RoadKinematicsGovernor
from sih.map.matcher import HMMMapMatcher
from sih.map.route_matcher import RouteMatcher
from sih.data.geo import geodetic_to_enu
from sih.core.contracts import VelocityEstimate, GNSSSample, FusedPosition
from sih.engine.speed_observer import KinematicSpeedObserver
# [ROUND1] feature-flagged hooks (all OFF by default -> r1 is None -> baseline path)
from sih.round1.engine_hooks import Round1EngineHooks
from sih.round1.history import build_pre_blackout_history


class SteppableStepResult:
    """Encapsulates the state and outputs of a single dead-reckoning IMU step."""
    def __init__(
        self,
        pure_pos: np.ndarray,
        map_pos: np.ndarray,
        pure_speed: float,
        map_speed: float,
        v_map_fwd: float,
        yaw_rate: float,
        fused_pure: Any,
        fused_map: Any,
        matched_pos: Any,
    ):
        self.pure_pos = pure_pos
        self.map_pos = map_pos
        self.pure_speed = pure_speed
        self.map_speed = map_speed
        self.v_map_fwd = v_map_fwd
        self.yaw_rate = yaw_rate
        self.fused_pure = fused_pure
        self.fused_map = fused_map
        self.matched_pos = matched_pos


class SteppableDeadReckoningEngine:
    """
    Steppable Dead Reckoning Engine.
    Encapsulates exact stateful sample-by-sample causal dead reckoning:
    - Pre-blackout GNSS & IMU conditioning / initialization
    - Blackout initialization (heading seeding, alpha scaling, speed observer reset, entry segment)
    - Sample-by-sample propagation: step(cal, v_pred, t_curr)
    Guarantees bit-identical execution between batch benchmark and real-time streaming adapter.
    """
    def __init__(
        self,
        reference_lat_deg: float,
        reference_lon_deg: float,
        reference_alt_m: float = 0.0,
        road_network: Optional[RoadNetwork] = None,
        domain: str = "Highway",
        turn_threshold_rad_s: float = np.radians(1.5),
        cooldown_duration_s: float = 0.5,
        max_gyro_bias_rad_s: float = np.radians(0.1),
        smoothing_factor: float = 0.35,
        enable_speed_scale: bool = True,
    ):
        self.ref_lat = reference_lat_deg
        self.ref_lon = reference_lon_deg
        self.ref_alt = reference_alt_m
        self.road_network = road_network
        self.domain = domain
        self.turn_threshold_rad_s = turn_threshold_rad_s
        self.cooldown_duration_s = cooldown_duration_s
        self.max_gyro_bias_rad_s = max_gyro_bias_rad_s
        self.smoothing_factor = smoothing_factor
        self.enable_speed_scale = enable_speed_scale

        self.ekf_pure = ErrorStateEKF(
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            initial_speed_scale=1.00,
        )
        self.ekf_map = ErrorStateEKF(
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            initial_speed_scale=1.00,
        )
        self.governor = RoadKinematicsGovernor(
            a_lat_max=2.2 if domain == "Highway" else 3.5,
            speed_limit_mps=33.3,
        )
        self.matcher = HMMMapMatcher(
            road_network=road_network,
            reference_lat_deg=self.ref_lat,
            reference_lon_deg=self.ref_lon,
            smoothing_factor=smoothing_factor,
        ) if road_network is not None else None

        self.speed_obs_pure = KinematicSpeedObserver()
        self.speed_obs_map = KinematicSpeedObserver()

        self.speed_scale: float = 1.00
        self.v_entry: float = 10.0
        self.entry_segment: Optional[RoadSegment] = None
        self.acq_radius_m: Optional[float] = None
        self.acq_h_diff_deg: Optional[float] = None
        self.entry_acq_info: str = "no entry segment"
        self.blackout_active: bool = False
        # [ROUND1] None when every round-1 flag is OFF
        self.r1: Optional[Round1EngineHooks] = Round1EngineHooks.create(domain)

    def init_from_gnss(self, warmup_gnss: GNSSSample) -> None:
        self.ekf_pure.init_from_gnss(warmup_gnss, self.ref_lat, self.ref_lon, self.ref_alt)
        self.ekf_map.init_from_gnss(warmup_gnss, self.ref_lat, self.ref_lon, self.ref_alt)

    def update_gnss_warmup(self, gnss: GNSSSample) -> None:
        if gnss.is_valid and self.ekf_pure._initialised:
            self.ekf_pure.update_gnss(gnss)
            self.ekf_map.update_gnss(gnss)

    def predict_warmup(self, cal: Any, v_raw: float, t_curr: int) -> None:
        vel_warm = VelocityEstimate(
            timestamp_ns=t_curr,
            forward_speed_mps=v_raw,
            speed_variance=0.3,
            motion_state="STATIONARY" if v_raw < 0.2 else "DRIVING",
        )
        self.ekf_pure.predict(cal, vel_warm)
        self.ekf_map.predict(cal, vel_warm)

    @staticmethod
    def synthesize_1hz_gnss_window(
        valid_hist_gnss: List[GNSSSample],
        bo_start_ns: int,
        ref_lat: float,
        ref_lon: float,
    ) -> List[GNSSSample]:
        """
        Synthesizes 1.0 Hz historical GNSS stream for heading seeding from sparse fixes.
        Emulates real-world Android FusedLocationProvider 1 Hz stream with ZERO future lookahead.
        """
        recent_hist = [
            g for g in valid_hist_gnss
            if bo_start_ns - int(25.0 * 1e9) <= g.timestamp_ns <= bo_start_ns
        ]
        if len(recent_hist) < 2 and len(valid_hist_gnss) >= 2:
            recent_hist = valid_hist_gnss[-2:]

        pre_gnss_window = []
        if len(recent_hist) >= 2:
            t_hist = np.array([g.timestamp_ns for g in recent_hist], dtype=np.float64)
            lat_hist = np.array([g.latitude_deg for g in recent_hist], dtype=np.float64)
            lon_hist = np.array([g.longitude_deg for g in recent_hist], dtype=np.float64)
            alt_hist = np.array([g.altitude_m if g.altitude_m is not None else 0.0 for g in recent_hist], dtype=np.float64)
            spd_hist = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in recent_hist], dtype=np.float64)

            t_start_grid = max(t_hist[0], bo_start_ns - int(15.0 * 1e9))
            t_1hz = np.arange(t_start_grid, t_hist[-1] + int(1e6), int(1e9))
            if len(t_1hz) >= 2:
                lats_1hz = np.interp(t_1hz, t_hist, lat_hist)
                lons_1hz = np.interp(t_1hz, t_hist, lon_hist)
                alts_1hz = np.interp(t_1hz, t_hist, alt_hist)
                spds_1hz = np.interp(t_1hz, t_hist, spd_hist)

                enu_list = [
                    geodetic_to_enu(lats_1hz[k], lons_1hz[k], 0.0, ref_lat, ref_lon, 0.0)[:2]
                    for k in range(len(t_1hz))
                ]
                for k in range(len(t_1hz)):
                    b_k = None
                    if k > 0:
                        de = enu_list[k][0] - enu_list[k - 1][0]
                        dn = enu_list[k][1] - enu_list[k - 1][1]
                        dist = float(np.sqrt(de**2 + dn**2))
                        if dist > 0.5:
                            b_k = float((np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0)
                    pre_gnss_window.append(GNSSSample(
                        timestamp_ns=int(t_1hz[k]),
                        latitude_deg=float(lats_1hz[k]),
                        longitude_deg=float(lons_1hz[k]),
                        altitude_m=float(alts_1hz[k]),
                        speed_mps=float(spds_1hz[k]),
                        bearing_deg=b_k,
                        accuracy_h_m=3.0,
                        is_valid=True,
                    ))

        if not pre_gnss_window:
            pre_gnss_window = (
                recent_hist
                if recent_hist
                else [g for g in valid_hist_gnss if bo_start_ns - int(15.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
            )
        return pre_gnss_window

    def start_blackout(
        self,
        entry_pos_enu: np.ndarray,
        pre_gnss_window: List[GNSSSample],
        recent_ai_speeds: List[float],
        recent_imu_calib: List[Any],
        cal_entry: Any,
        t_entry_ns: int,
    ) -> None:
        self.blackout_active = True
        self.ekf_pure._p[0] = entry_pos_enu[0]
        self.ekf_pure._p[1] = entry_pos_enu[1]
        self.ekf_map._p[0] = entry_pos_enu[0]
        self.ekf_map._p[1] = entry_pos_enu[1]

        # Dynamic pre-blackout speed scale factor learning from healthy GNSS fixes
        if self.enable_speed_scale and pre_gnss_window:
            self.v_entry = float(pre_gnss_window[-1].speed_mps) if pre_gnss_window[-1].speed_mps is not None else 8.0
            g_speeds = [g.speed_mps for g in pre_gnss_window if g.speed_mps is not None and g.speed_mps > 2.0]
            ai_speeds = recent_ai_speeds[max(0, len(recent_ai_speeds) - len(g_speeds) * 10):]
            if len(g_speeds) >= 3 and len(ai_speeds) >= 10:
                scale = np.mean(g_speeds) / max(0.5, np.mean(ai_speeds))
                self.speed_scale_raw = float(scale)  # [ROUND1] T10: unclipped ratio, read by hooks only
                self.speed_scale = float(np.clip(scale, 0.85, 1.35 if self.domain == "Highway" else 1.25))

        self.speed_obs_pure.reset(initial_speed_mps=self.v_entry, initial_ts_ns=t_entry_ns)
        self.speed_obs_map.reset(initial_speed_mps=self.v_entry, initial_ts_ns=t_entry_ns)

        # Locate reference fix used for heading seeding and integrate gyro forward from that fix
        valid_moving = [g for g in pre_gnss_window if g.is_valid and g.speed_mps is not None and g.speed_mps > 0.5]
        if valid_moving and valid_moving[-1].speed_mps >= 2.5:
            ref_fix = valid_moving[-1]
        else:
            stable = [g for g in pre_gnss_window if g.is_valid and g.speed_mps is not None and g.speed_mps >= 2.0 and g.bearing_deg is not None]
            ref_fix = stable[-1] if stable else (valid_moving[-1] if valid_moving else (pre_gnss_window[-1] if pre_gnss_window else None))

        delta_gyro_deg = 0.0
        if ref_fix is not None and recent_imu_calib:
            last_g_ts = ref_fix.timestamp_ns
            for k in range(1, len(recent_imu_calib)):
                c_prev = recent_imu_calib[k - 1]
                c_curr = recent_imu_calib[k]
                t_k = c_curr.timestamp_ns
                if last_g_ts < t_k <= t_entry_ns:
                    dt_k = (t_k - c_prev.timestamp_ns) * 1e-9
                    delta_gyro_deg += np.degrees(c_curr.gyro_vehicle[2] * dt_k)

        # Reference heading for initial road segment matching
        valid_hist = [g for g in pre_gnss_window if g.is_valid and g.speed_mps is not None and g.speed_mps > 1.5 and g.bearing_deg is not None]
        ref_motion_hdg = float(valid_hist[-1].bearing_deg) if valid_hist else (float(np.degrees(self.ekf_map._heading_rad)) % 360.0)

        init_road_bearing = None
        if self.road_network is not None and self.matcher is not None:
            best_cand, acq_r, acq_h, acq_info = DeadReckoningEngine.acquire_entry_segment(entry_pos_enu, ref_motion_hdg, self.road_network)
            if best_cand is not None:
                init_road_bearing = best_cand.bearing_deg
                self.matcher.set_active_segment(best_cand)
                self.entry_segment = best_cand
            self.acq_radius_m = acq_r
            self.acq_h_diff_deg = acq_h
            self.entry_acq_info = acq_info

        turn_rate_entry = float(cal_entry.gyro_vehicle[2])
        self.ekf_pure.seed_pre_blackout_heading(
            pre_gnss_window,
            road_bearing_deg=init_road_bearing,
            delta_heading_gyro_deg=delta_gyro_deg,
            current_yaw_rate_rad_s=turn_rate_entry,
        )
        self.ekf_map.seed_pre_blackout_heading(
            pre_gnss_window,
            road_bearing_deg=init_road_bearing,
            delta_heading_gyro_deg=delta_gyro_deg,
            current_yaw_rate_rad_s=turn_rate_entry,
        )

        # [ROUND1] entry-speed bookkeeping for T3/T5
        if self.r1 is not None:
            self.r1.on_blackout_start(self, recent_ai_speeds, t_entry_ns)

    def step(self, cal: Any, v_pred: float, t_curr: int) -> SteppableStepResult:
        # [ROUND1] T4 gyro scale (returns cal unchanged when OFF)
        if self.r1 is not None:
            cal = self.r1.pre_step_cal(cal)
        v_ai_cal = float(v_pred) * self.speed_scale
        # [ROUND1] T7 band factor + T5 speed mode
        if self.r1 is not None:
            v_ai_cal = self.r1.adjust_ai_speed(float(v_pred), v_ai_cal, t_curr)
        v_pure_fwd, is_stat_pure = self.speed_obs_pure.update(cal, v_ai_cal)
        # [ROUND1] T3 sticky stop detector
        if self.r1 is not None:
            v_pure_fwd, is_stat_pure = self.r1.post_observer(self.speed_obs_pure, v_pure_fwd, is_stat_pure, cal, v_ai_cal, t_curr)

        # Apply Road Kinematics Governor ONLY to the map-matched stream
        v_map_fwd = v_pure_fwd
        is_stat_map = is_stat_pure
        turn_rate_yaw = float(cal.gyro_vehicle[2])
        local_kappa = 0.0
        if self.road_network is not None:
            nearest_segs = self.road_network.find_candidates(self.ekf_map._p[:2], radius_m=35.0)
            if nearest_segs and len(nearest_segs) >= 2:
                p1 = nearest_segs[0].start_enu_m
                p2 = nearest_segs[0].end_enu_m
                p3 = nearest_segs[1].end_enu_m
                if np.linalg.norm(p2 - nearest_segs[1].start_enu_m) < 8.0:
                    kappas = self.governor.compute_curvature(np.array([p1, p2, p3]))
                    local_kappa = float(np.max(kappas))
            v_map_fwd, _ = self.governor.govern_speed(v_pure_fwd, curvature=local_kappa, yaw_rate_rad_s=turn_rate_yaw)
            if v_map_fwd < 0.2:
                is_stat_map = True

        vel_pure = VelocityEstimate(
            timestamp_ns=t_curr,
            forward_speed_mps=v_pure_fwd,
            speed_variance=0.3,
            motion_state="STATIONARY" if is_stat_pure or v_pure_fwd < 0.2 else "DRIVING",
        )
        vel_map = VelocityEstimate(
            timestamp_ns=t_curr,
            forward_speed_mps=v_map_fwd,
            speed_variance=0.3,
            motion_state="STATIONARY" if is_stat_map or v_map_fwd < 0.2 else "DRIVING",
        )

        fused_pure = self.ekf_pure.predict(cal, vel_pure)
        fused_map = self.ekf_map.predict(cal, vel_map)

        matched_pos = None
        if v_map_fwd > 1.0 and self.matcher is not None:
            matched_pos = self.matcher.match(fused_map, ekf=self.ekf_map, domain=self.domain, v_fwd=v_map_fwd)

        # [ROUND1] T8 junction along-track anchoring (map stream only)
        if self.r1 is not None:
            self.r1.post_step(self, cal, v_map_fwd, t_curr)

        return SteppableStepResult(
            pure_pos=self.ekf_pure._p[:2].copy(),
            map_pos=self.ekf_map._p[:2].copy(),
            pure_speed=float(np.linalg.norm(fused_pure.velocity_enu_mps)),
            map_speed=float(np.linalg.norm(self.ekf_map._v)),
            v_map_fwd=float(v_map_fwd),
            yaw_rate=turn_rate_yaw,
            fused_pure=fused_pure,
            fused_map=fused_map,
            matched_pos=matched_pos,
        )


class DeadReckoningEngine:
    """
    Unified execution engine for vehicle dead reckoning during GNSS blackouts.
    Orchestrates ES-EKF filtering, AI velocity integration, road kinematics governance,
    topological HMM map matching, and route-level hypothesis evaluation.
    """

    def __init__(
        self,
        turn_threshold_rad_s: float = np.radians(1.5),
        cooldown_duration_s: float = 0.5,
        max_gyro_bias_rad_s: float = np.radians(0.1),
        smoothing_factor: float = 0.35,
        min_route_ratio: float = 1.80,
        enable_route_matching: bool = False,
        enable_speed_scale: bool = True,
    ):
        self.turn_threshold_rad_s = turn_threshold_rad_s
        self.cooldown_duration_s = cooldown_duration_s
        self.max_gyro_bias_rad_s = max_gyro_bias_rad_s
        self.smoothing_factor = smoothing_factor
        self.enable_route_matching = enable_route_matching
        self.enable_speed_scale = enable_speed_scale
        self.route_matcher = RouteMatcher(min_route_ratio=min_route_ratio)

    @staticmethod
    def acquire_entry_segment(
        entry_pos_enu: np.ndarray,
        ref_heading_deg: float,
        road_net: RoadNetwork,
        search_radii: Optional[List[float]] = None,
        max_heading_diff_deg: float = 45.0,
    ) -> Tuple[Optional[RoadSegment], Optional[float], Optional[float], str]:
        """
        Acquires initial road segment via widening search (35m -> 75m -> 150m),
        filtering candidates by heading alignment (<= 45 deg) rather than purely spatial proximity.
        """
        if search_radii is None:
            search_radii = [35.0, 75.0, 150.0]

        best_cand = None
        acq_radius_m = None
        acq_h_diff_deg = None

        for r_search in search_radii:
            cands = road_net.find_candidates(entry_pos_enu, radius_m=r_search)
            valid_cands = []
            for s in cands:
                b_diff = abs((s.bearing_deg - ref_heading_deg + 180.0) % 360.0 - 180.0)
                if b_diff <= max_heading_diff_deg:
                    valid_cands.append((b_diff, s))
            if valid_cands:
                valid_cands.sort(key=lambda x: x[0])
                acq_h_diff_deg, best_cand = valid_cands[0]
                acq_radius_m = r_search
                break

        if best_cand is not None:
            acq_info = f"Acquired at radius {acq_radius_m:.0f}m (heading diff {acq_h_diff_deg:.1f} deg)"
        else:
            acq_info = f"no entry segment (none found within {search_radii[-1]:.0f}m, {max_heading_diff_deg:.0f} deg heading match)"

        return best_cand, acq_radius_m, acq_h_diff_deg, acq_info

    def run_scenario(
        self,
        trip: Any,
        calib_samples: List[Any],
        v_preds: np.ndarray,
        road_net: RoadNetwork,
        g_entry: Any,
        duration_s: float,
        domain: str = "Highway",
        can_speeds: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a single blackout evaluation scenario by orchestrating SteppableDeadReckoningEngine.

        Parameters
        ----------
        trip : GenericTrip
            Input drive containing IMU and GNSS sequences.
        calib_samples : list of CalibratedSample
            Mount-calibrated IMU samples.
        v_preds : np.ndarray
            AI forward speed estimates for each IMU sample.
        road_net : RoadNetwork
            Topological road network for corridor map matching.
        g_entry : GNSSSample
            GNSS fix marking the entrance to satellite blackout.
        duration_s : float
            Blackout duration in seconds.
        domain : str
            Operational domain ("Highway", "Arterial", "Urban").
        can_speeds : np.ndarray, optional
            CAN-bus wheel speed ground truth sequence.

        Returns
        -------
        dict or None
            Scenario trajectory, error decomposition, and evaluation metrics.
        """
        t0_ns = trip.imu_samples[0].timestamp_ns
        bo_start_ns = g_entry.timestamp_ns
        bo_end_ns = bo_start_ns + int(duration_s * 1e9)

        valid_gnss = [g for g in trip.gnss_samples if g.is_valid]
        bo_gnss = [x for x in valid_gnss if bo_start_ns <= x.timestamp_ns <= bo_end_ns]
        if len(bo_gnss) < 3:
            return None

        gt_pts = [
            geodetic_to_enu(
                g.latitude_deg, g.longitude_deg, 0.0,
                trip.reference_lat_deg, trip.reference_lon_deg, 0.0
            )[:2]
            for g in bo_gnss
        ]
        gt_pts = np.array(gt_pts)
        gt_start_enu = gt_pts[0]
        gt_end_enu = gt_pts[-1]
        gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1)))

        if gt_dist < 15.0:
            return None

        warmup_start_ns = max(t0_ns, bo_start_ns - int(30.0 * 1e9))
        warmup_gnss = min(
            [g for g in valid_gnss if g.timestamp_ns <= bo_start_ns],
            key=lambda g: abs(g.timestamp_ns - warmup_start_ns),
            default=valid_gnss[0],
        )

        session = SteppableDeadReckoningEngine(
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0,
            road_network=road_net,
            domain=domain,
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            smoothing_factor=self.smoothing_factor,
            enable_speed_scale=self.enable_speed_scale,
        )
        session.init_from_gnss(warmup_gnss)

        # [ROUND1] causal pre-blackout history (t < bo_start) for T3/T4/T7 learners
        if session.r1 is not None and session.r1.needs_history:
            session.r1.set_history(build_pre_blackout_history(
                trip, calib_samples, v_preds, bo_start_ns, session.r1.cfg.history_s))

        n_gnss = len(trip.gnss_samples)
        gnss_idx = 0
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns < warmup_start_ns:
            gnss_idx += 1

        pure_pts = []
        map_pts = []
        map_ts_list = []
        pure_speeds = []
        map_speeds = []
        blackout_started = False
        bo_timestamps_ns = []
        bo_yaw_rates = []
        bo_speeds = []
        bo_dr_enu = []

        valid_hist_gnss = [g for g in valid_gnss if g.timestamp_ns <= bo_start_ns]
        pre_gnss_window = SteppableDeadReckoningEngine.synthesize_1hz_gnss_window(
            valid_hist_gnss, bo_start_ns, trip.reference_lat_deg, trip.reference_lon_deg
        )

        for j, imu in enumerate(trip.imu_samples):
            t_curr = imu.timestamp_ns
            if t_curr < warmup_start_ns:
                continue
            if t_curr > bo_end_ns:
                break

            while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                g = trip.gnss_samples[gnss_idx]
                if g.timestamp_ns <= bo_start_ns:
                    session.update_gnss_warmup(g)
                gnss_idx += 1

            cal = calib_samples[j]

            if not blackout_started and t_curr >= bo_start_ns:
                blackout_started = True
                session.start_blackout(
                    entry_pos_enu=gt_start_enu,
                    pre_gnss_window=pre_gnss_window,
                    recent_ai_speeds=list(v_preds[:j]),
                    recent_imu_calib=calib_samples[:j+1],
                    cal_entry=cal,
                    t_entry_ns=t_curr,
                )

            if not blackout_started:
                session.predict_warmup(cal, float(v_preds[j]), t_curr)
            else:
                step_res = session.step(cal, float(v_preds[j]), t_curr)
                pure_pts.append(step_res.pure_pos.copy())
                map_pts.append(step_res.map_pos.copy())
                map_ts_list.append(t_curr)
                pure_speeds.append(step_res.pure_speed)
                map_speeds.append(step_res.map_speed)

                bo_timestamps_ns.append(t_curr)
                bo_yaw_rates.append(step_res.yaw_rate)
                bo_speeds.append(step_res.v_map_fwd)
                bo_dr_enu.append(step_res.pure_pos.copy())

        pure_pts = np.array(pure_pts)
        map_pts = np.array(map_pts)

        if len(pure_pts) < 2 or len(map_pts) < 2:
            return None

        entry_segment = session.entry_segment
        matcher = session.matcher
        ekf_pure = session.ekf_pure
        ekf_map = session.ekf_map
        speed_scale = session.speed_scale
        v_entry = session.v_entry
        acq_radius_m = session.acq_radius_m
        acq_h_diff_deg = session.acq_h_diff_deg
        entry_acq_info = session.entry_acq_info

        seeded_hdg = float(np.degrees(ekf_pure._heading_rad)) % 360.0
        if g_entry.bearing_deg is not None:
            gt_hdg_entry = float(g_entry.bearing_deg)
        elif len(gt_pts) >= 2:
            v_gt_start = gt_pts[1] - gt_pts[0]
            gt_hdg_entry = float(np.degrees(np.arctan2(v_gt_start[0], v_gt_start[1])) % 360.0)
        else:
            gt_hdg_entry = seeded_hdg
        hdg_seed_err = float(abs((seeded_hdg - gt_hdg_entry + 180.0) % 360.0 - 180.0))

        # Route-level matching evaluation at blackout exit
        final_route_res = None
        if self.enable_route_matching and entry_segment is not None and len(bo_timestamps_ns) >= 5:
            final_route_res = self.route_matcher.match_blackout_route(
                entry_segment=entry_segment,
                succ_map=matcher._succ_map,
                timestamps_ns=np.array(bo_timestamps_ns),
                yaw_rates_rad_s=np.array(bo_yaw_rates),
                speeds_mps=np.array(bo_speeds),
                dr_trajectory_enu=np.array(bo_dr_enu),
            )

        route_won = False
        route_ratio = 0.0
        route_count = 0
        route_time_ms = 0.0
        route_memory_kb = 0.0
        route_turns = 0
        if not self.enable_route_matching:
            route_fallback_reason = "route matching disabled"
        elif entry_segment is None:
            route_fallback_reason = "no entry segment"
        else:
            route_fallback_reason = "Evaluation skipped (<5 samples)"

        if final_route_res is not None:
            route_ratio = final_route_res.confidence_ratio
            route_count = final_route_res.total_routes_evaluated
            route_time_ms = final_route_res.enumeration_time_ms + final_route_res.scoring_time_ms
            route_memory_kb = final_route_res.peak_memory_kb
            route_turns = final_route_res.observed_turns_count
            route_fallback_reason = final_route_res.fallback_reason

            if final_route_res.won and final_route_res.projected_pts_enu is not None:
                route_won = True
                map_pts = final_route_res.projected_pts_enu.copy()

        # Ground truth timestamp synchronization: evaluate at last valid blackout GNSS fix
        map_ts_arr = np.array(map_ts_list, dtype=np.float64)
        gt_ts_arr = np.array([g.timestamp_ns for g in bo_gnss], dtype=np.float64)
        gt_spd_arr = np.array([g.speed_mps for g in bo_gnss], dtype=np.float64)

        eval_east = float(np.interp(gt_ts_arr[-1], map_ts_arr, map_pts[:, 0]))
        eval_north = float(np.interp(gt_ts_arr[-1], map_ts_arr, map_pts[:, 1]))
        eval_pt = np.array([eval_east, eval_north])

        eval_pure_east = float(np.interp(gt_ts_arr[-1], map_ts_arr, pure_pts[:, 0]))
        eval_pure_north = float(np.interp(gt_ts_arr[-1], map_ts_arr, pure_pts[:, 1]))
        eval_pure_pt = np.array([eval_pure_east, eval_pure_north])

        final_err_pure = float(np.linalg.norm(eval_pure_pt - gt_end_enu))
        final_err_map = float(np.linalg.norm(eval_pt - gt_end_enu))
        pure_drift_pct = (final_err_pure / gt_dist) * 100.0
        map_drift_pct = (final_err_map / gt_dist) * 100.0

        # Ground-truth track unit tangent and normal at outage exit
        if bo_gnss[-1].bearing_deg is not None and bo_gnss[-1].speed_mps is not None and bo_gnss[-1].speed_mps > 0.5:
            b_rad = np.radians(bo_gnss[-1].bearing_deg)
            te_end = float(np.sin(b_rad))
            tn_end = float(np.cos(b_rad))
        elif len(gt_pts) >= 2:
            d_end = gt_pts[-1] - gt_pts[-2]
            k_step = -2
            while np.linalg.norm(d_end) < 0.1 and k_step >= -len(gt_pts):
                d_end = gt_pts[-1] - gt_pts[k_step]
                k_step -= 1
            d_norm = float(np.linalg.norm(d_end))
            if d_norm > 1e-4:
                te_end = float(d_end[0] / d_norm)
                tn_end = float(d_end[1] / d_norm)
            else:
                te_end, tn_end = 1.0, 0.0
        else:
            te_end, tn_end = 1.0, 0.0

        t_end_norm = float(np.hypot(te_end, tn_end))
        if t_end_norm > 1e-6:
            te_end /= t_end_norm
            tn_end /= t_end_norm
        else:
            te_end, tn_end = 1.0, 0.0

        ne_end = -tn_end
        nn_end = te_end

        err_vec_map = eval_pt - gt_end_enu
        final_at_map = float(err_vec_map[0] * te_end + err_vec_map[1] * tn_end)
        final_ct_map = float(err_vec_map[0] * ne_end + err_vec_map[1] * nn_end)

        err_vec_pure = eval_pure_pt - gt_end_enu
        final_at_pure = float(err_vec_pure[0] * te_end + err_vec_pure[1] * tn_end)
        final_ct_pure = float(err_vec_pure[0] * ne_end + err_vec_pure[1] * nn_end)

        assert abs(np.hypot(final_at_map, final_ct_map) - final_err_map) < 1e-4, (
            f"Decomposition invariant failed: sqrt({final_at_map}^2 + {final_ct_map}^2) != {final_err_map}"
        )

        # Time-series error decomposition along the blackout duration
        gt_interp_e = np.interp(map_ts_arr, gt_ts_arr, gt_pts[:, 0])
        gt_interp_n = np.interp(map_ts_arr, gt_ts_arr, gt_pts[:, 1])
        gt_spd_interp = np.interp(map_ts_arr, gt_ts_arr, gt_spd_arr)

        # Use true 10 Hz continuous vehicle CAN wheel speed as ground truth when available
        if can_speeds is not None and len(can_speeds) >= len(trip.imu_samples):
            imu_ts_arr = np.array([imu.timestamp_ns for imu in trip.imu_samples], dtype=np.float64)
            can_spd_interp = np.interp(map_ts_arr, imu_ts_arr, can_speeds).astype(np.float32)
        else:
            can_spd_interp = gt_spd_interp.astype(np.float32)

        err_pure_series = np.hypot(pure_pts[:, 0] - gt_interp_e, pure_pts[:, 1] - gt_interp_n)
        err_map_series = np.hypot(map_pts[:, 0] - gt_interp_e, map_pts[:, 1] - gt_interp_n)

        de = np.gradient(gt_interp_e)
        dn = np.gradient(gt_interp_n)
        ds = np.hypot(de, dn)

        # Fill unit tangent vectors for stationary or clamped steps
        valid_ds = ds > 1e-3
        te = np.zeros_like(ds)
        tn = np.zeros_like(ds)
        if np.any(valid_ds):
            te[valid_ds] = de[valid_ds] / ds[valid_ds]
            tn[valid_ds] = dn[valid_ds] / ds[valid_ds]
            for idx in range(len(te)):
                if not valid_ds[idx]:
                    prev_valid = np.where(valid_ds[:idx])[0]
                    next_valid = np.where(valid_ds[idx + 1:])[0]
                    if len(prev_valid) > 0:
                        te[idx] = te[prev_valid[-1]]
                        tn[idx] = tn[prev_valid[-1]]
                    elif len(next_valid) > 0:
                        te[idx] = te[idx + 1 + next_valid[0]]
                        tn[idx] = tn[idx + 1 + next_valid[0]]
                    else:
                        te[idx] = te_end
                        tn[idx] = tn_end
        else:
            te[:] = te_end
            tn[:] = tn_end

        # Ensure the final point matches the exact endpoint decomposition
        te[-1] = te_end
        tn[-1] = tn_end

        diff_e = map_pts[:, 0] - gt_interp_e
        diff_n = map_pts[:, 1] - gt_interp_n
        along_track_series = diff_e * te + diff_n * tn
        cross_track_series = diff_e * (-tn) + diff_n * te
        along_track_series[-1] = final_at_map
        cross_track_series[-1] = final_ct_map
        time_rel_s = (map_ts_arr - bo_start_ns) * 1e-9

        result = {
            "t_start_s": (bo_start_ns - t0_ns) * 1e-9,
            "duration_s": duration_s,
            "dist_m": gt_dist,
            "pure_err_m": final_err_pure,
            "pure_drift_pct": pure_drift_pct,
            "map_err_m": final_err_map,
            "map_drift_pct": map_drift_pct,
            "pure_pts": pure_pts,
            "map_pts": map_pts,
            "gt_pts": gt_pts,
            "time_rel_s": time_rel_s,
            "pure_speeds": np.array(pure_speeds),
            "map_speeds": np.array(map_speeds),
            "gt_speeds": can_spd_interp,
            "gt_can_speeds": can_spd_interp,
            "gt_gps_speeds": gt_spd_interp,
            "err_pure_series": err_pure_series,
            "err_map_series": err_map_series,
            "along_track_series": along_track_series,
            "cross_track_series": cross_track_series,
            "final_at_m": final_at_map,
            "final_ct_m": final_ct_map,
            "along_track_m": final_at_map,
            "cross_track_m": final_ct_map,
            "pure_at_m": final_at_pure,
            "pure_ct_m": final_ct_pure,
            "hdg_seed_err": hdg_seed_err,
            "total_match_steps": matcher.total_match_steps,
            "gate_suppressed_steps": matcher.gate_suppressed_steps,
            "gate_suppressed_pct": (matcher.gate_suppressed_steps / max(1, matcher.total_match_steps)) * 100.0,
            "hard_gate_suppressed_steps": matcher.hard_gate_suppressed_steps,
            "hard_gate_suppressed_pct": (matcher.hard_gate_suppressed_steps / max(1, matcher.total_match_steps)) * 100.0,
            "ambiguity_gate_suppressed_steps": matcher.ambiguity_gate_suppressed_steps,
            "ambiguity_gate_suppressed_pct": (matcher.ambiguity_gate_suppressed_steps / max(1, matcher.total_match_steps)) * 100.0,
            "hysteresis_suppressed_steps": matcher.hysteresis_suppressed_steps,
            "hysteresis_suppressed_pct": (matcher.hysteresis_suppressed_steps / max(1, matcher.total_match_steps)) * 100.0,
            "single_candidate_steps": matcher.single_candidate_steps,
            "single_candidate_pct": (matcher.single_candidate_steps / max(1, matcher.total_match_steps)) * 100.0,
            "score_ratio_median": float(np.median(matcher.score_ratios)) if matcher.score_ratios else 0.0,
            "score_ratio_p10": float(np.percentile(matcher.score_ratios, 10)) if len(matcher.score_ratios) >= 2 else 0.0,
            "score_ratio_p90": float(np.percentile(matcher.score_ratios, 90)) if len(matcher.score_ratios) >= 2 else 0.0,
            "route_match_won": route_won,
            "route_match_ratio": float(route_ratio),
            "route_count": int(route_count),
            "route_time_ms": float(route_time_ms),
            "route_memory_kb": float(route_memory_kb),
            "route_turns_count": int(route_turns),
            "route_fallback_reason": route_fallback_reason,
            "entry_radius_m": acq_radius_m,
            "entry_h_diff_deg": acq_h_diff_deg,
            "entry_acq_info": entry_acq_info,
        }
        # [ROUND1] diagnostics (keys only added when a round-1 flag is ON)
        if session.r1 is not None:
            result.update(session.r1.summary())
        return result


def run_dead_reckoning_scenario(
    trip: Any,
    calib_samples: List[Any],
    v_preds: np.ndarray,
    road_net: RoadNetwork,
    g_entry: Any,
    duration_s: float,
    domain: str = "Highway",
    can_speeds: Optional[np.ndarray] = None,
    enable_route_matching: bool = False,
    enable_speed_scale: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Convenience functional wrapper for executing a dead-reckoning scenario with default engine settings.
    """
    engine = DeadReckoningEngine(
        enable_route_matching=enable_route_matching,
        enable_speed_scale=enable_speed_scale,
    )
    return engine.run_scenario(
        trip=trip,
        calib_samples=calib_samples,
        v_preds=v_preds,
        road_net=road_net,
        g_entry=g_entry,
        duration_s=duration_s,
        domain=domain,
        can_speeds=can_speeds,
    )
