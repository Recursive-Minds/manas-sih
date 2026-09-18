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
from sih.map.network import RoadNetwork
from sih.map.governor import RoadKinematicsGovernor
from sih.map.matcher import HMMMapMatcher
from sih.map.route_matcher import RouteMatcher
from sih.data.geo import geodetic_to_enu
from sih.core.contracts import VelocityEstimate, GNSSSample
from sih.engine.speed_observer import KinematicSpeedObserver


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
        Executes a single blackout evaluation scenario.

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

        ekf_pure = ErrorStateEKF(
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            initial_speed_scale=1.00,
        )
        ekf_pure.init_from_gnss(
            warmup_gnss,
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0,
        )

        ekf_map = ErrorStateEKF(
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            initial_speed_scale=1.00,
        )
        ekf_map.init_from_gnss(
            warmup_gnss,
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            reference_alt_m=0.0,
        )

        # Road governor: AASHTO/IRC highway comfort limit (2.2 m/s^2 with superelevation) on highway; intersection limit (3.5 m/s^2) elsewhere
        governor = RoadKinematicsGovernor(
            a_lat_max=2.2 if domain == "Highway" else 3.5,
            speed_limit_mps=33.3,
        )
        matcher = HMMMapMatcher(
            road_network=road_net,
            reference_lat_deg=trip.reference_lat_deg,
            reference_lon_deg=trip.reference_lon_deg,
            smoothing_factor=self.smoothing_factor,
        )

        # High-fidelity Kinematic Delta-v speed observers
        speed_obs_pure = KinematicSpeedObserver()
        speed_obs_map = KinematicSpeedObserver()

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
        speed_scale = 1.00
        v_entry = 10.0
        entry_segment = None
        acq_radius_m = None
        acq_h_diff_deg = None
        entry_acq_info = "no entry segment"
        bo_timestamps_ns = []
        bo_yaw_rates = []
        bo_speeds = []
        bo_dr_enu = []

        for j, imu in enumerate(trip.imu_samples):
            t_curr = imu.timestamp_ns
            if t_curr < warmup_start_ns:
                continue
            if t_curr > bo_end_ns + int(1e9):
                break

            while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
                g = trip.gnss_samples[gnss_idx]
                if g.timestamp_ns <= bo_start_ns:
                    ekf_pure.update_gnss(g)
                    ekf_map.update_gnss(g)
                gnss_idx += 1

            cal = calib_samples[j]

            if not blackout_started and t_curr >= bo_start_ns:
                blackout_started = True
                ekf_pure._p[0] = gt_start_enu[0]
                ekf_pure._p[1] = gt_start_enu[1]
                ekf_map._p[0] = gt_start_enu[0]
                ekf_map._p[1] = gt_start_enu[1]

                # 1. Synthesize 1.0 Hz historical GNSS stream for testing on sparse IO-VNBD dataset
                # Emulates real-world Android FusedLocationProvider 1 Hz stream with ZERO future lookahead
                valid_hist_gnss = [g for g in valid_gnss if g.timestamp_ns <= bo_start_ns]
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
                            geodetic_to_enu(lats_1hz[k], lons_1hz[k], 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
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
                        else [g for g in valid_gnss if bo_start_ns - int(15.0 * 1e9) <= g.timestamp_ns <= bo_start_ns]
                    )

                # Dynamic pre-blackout speed scale factor learning from healthy GNSS fixes
                if self.enable_speed_scale and pre_gnss_window:
                    v_entry = float(pre_gnss_window[-1].speed_mps) if pre_gnss_window[-1].speed_mps is not None else 8.0
                    g_speeds = [g.speed_mps for g in pre_gnss_window if g.speed_mps is not None and g.speed_mps > 2.0]
                    ai_speeds = [v_preds[k] for k in range(max(0, j - len(g_speeds) * 10), j)]
                    if len(g_speeds) >= 3 and len(ai_speeds) >= 10:
                        scale = np.mean(g_speeds) / max(0.5, np.mean(ai_speeds))
                        speed_scale = float(np.clip(scale, 0.85, 1.35 if domain == "Highway" else 1.25))

                # Initialize kinematic speed observers anchored at true entry velocity
                speed_obs_pure.reset(initial_speed_mps=v_entry, initial_ts_ns=t_curr)
                speed_obs_map.reset(initial_speed_mps=v_entry, initial_ts_ns=t_curr)

                # Locate reference fix used for heading seeding and integrate gyro forward from that fix
                valid_moving = [g for g in pre_gnss_window if g.is_valid and g.speed_mps is not None and g.speed_mps > 0.5]
                if valid_moving and valid_moving[-1].speed_mps >= 2.5:
                    ref_fix = valid_moving[-1]
                else:
                    stable_fixes = [
                        g for g in pre_gnss_window
                        if g.is_valid and g.speed_mps is not None and g.speed_mps >= 2.0 and g.bearing_deg is not None
                    ]
                    ref_fix = stable_fixes[-1] if stable_fixes else (valid_moving[-1] if valid_moving else (pre_gnss_window[-1] if pre_gnss_window else None))

                delta_gyro_deg = 0.0
                if ref_fix is not None:
                    last_g_ts = ref_fix.timestamp_ns
                    for k_imu in range(len(trip.imu_samples)):
                        t_k = trip.imu_samples[k_imu].timestamp_ns
                        if last_g_ts < t_k <= bo_start_ns:
                            dt_k = (t_k - trip.imu_samples[k_imu - 1].timestamp_ns) * 1e-9
                            delta_gyro_deg += np.degrees(calib_samples[k_imu].gyro_vehicle[2] * dt_k)

                # Reference heading for initial road segment matching:
                # Uses latest moving GNSS Doppler bearing when available, or EKF heading
                valid_hist = [
                    g for g in pre_gnss_window
                    if g.is_valid and g.speed_mps is not None and g.speed_mps > 1.5 and g.bearing_deg is not None
                ]
                ref_motion_hdg = float(valid_hist[-1].bearing_deg) if valid_hist else (float(np.degrees(ekf_map._heading_rad)) % 360.0)

                init_road_bearing = None
                best_cand = None
                acq_radius_m = None
                acq_h_diff_deg = None

                # 1. Entry Segment Acquisition: widening search (35m, 75m, 150m) preferring closest bearing match within 45 deg
                best_cand, acq_radius_m, acq_h_diff_deg, entry_acq_info = self.acquire_entry_segment(
                    gt_start_enu, ref_motion_hdg, road_net
                )
                if best_cand is not None:
                    init_road_bearing = best_cand.bearing_deg
                    matcher.set_active_segment(best_cand)
                    entry_segment = best_cand
                else:
                    entry_segment = None

                turn_rate_entry = float(cal.gyro_vehicle[2])
                ekf_pure.seed_pre_blackout_heading(
                    pre_gnss_window,
                    road_bearing_deg=init_road_bearing,
                    delta_heading_gyro_deg=delta_gyro_deg,
                    current_yaw_rate_rad_s=turn_rate_entry,
                )
                ekf_map.seed_pre_blackout_heading(
                    pre_gnss_window,
                    road_bearing_deg=init_road_bearing,
                    delta_heading_gyro_deg=delta_gyro_deg,
                    current_yaw_rate_rad_s=turn_rate_entry,
                )
                seeded_hdg = float(np.degrees(ekf_pure._heading_rad)) % 360.0
                if g_entry.bearing_deg is not None:
                    gt_hdg_entry = float(g_entry.bearing_deg)
                elif len(gt_pts) >= 2:
                    v_gt_start = gt_pts[1] - gt_pts[0]
                    gt_hdg_entry = float(np.degrees(np.arctan2(v_gt_start[0], v_gt_start[1])) % 360.0)
                else:
                    gt_hdg_entry = seeded_hdg
                hdg_seed_err = float(abs((seeded_hdg - gt_hdg_entry + 180.0) % 360.0 - 180.0))

            v_ai_cal = float(v_preds[j]) * (speed_scale if blackout_started else 1.0)

            if blackout_started:
                v_pure_fwd, is_stat_pure = speed_obs_pure.update(cal, v_ai_cal)
            else:
                v_pure_fwd = v_ai_cal
                is_stat_pure = (v_ai_cal < 0.2)

            # Apply Road Kinematics Governor ONLY to the map-matched stream
            v_map_fwd = v_pure_fwd
            is_stat_map = is_stat_pure
            if blackout_started:
                turn_rate_yaw = float(cal.gyro_vehicle[2])
                local_kappa = 0.0
                nearest_segs = road_net.find_candidates(ekf_map._p[:2], radius_m=35.0)
                if nearest_segs and len(nearest_segs) >= 2:
                    p1 = nearest_segs[0].start_enu_m
                    p2 = nearest_segs[0].end_enu_m
                    p3 = nearest_segs[1].end_enu_m
                    # Verify spatial continuity between segments before computing curvature
                    if np.linalg.norm(p2 - nearest_segs[1].start_enu_m) < 8.0:
                        kappas = governor.compute_curvature(np.array([p1, p2, p3]))
                        local_kappa = float(np.max(kappas))
                v_map_fwd, _ = governor.govern_speed(v_pure_fwd, curvature=local_kappa, yaw_rate_rad_s=turn_rate_yaw)
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

            fused_pure = ekf_pure.predict(cal, vel_pure)
            fused_map = ekf_map.predict(cal, vel_map)

            if bo_start_ns <= t_curr <= bo_end_ns and v_map_fwd > 1.0:
                matcher.match(fused_map, ekf=ekf_map, domain=domain, v_fwd=v_map_fwd)

            if bo_start_ns <= t_curr <= bo_end_ns:
                pure_pts.append(fused_pure.position_enu_m[:2].copy())
                map_pts.append(ekf_map._p[:2].copy())
                map_ts_list.append(t_curr)
                pure_speeds.append(float(np.linalg.norm(fused_pure.velocity_enu_mps)))
                map_speeds.append(float(np.linalg.norm(ekf_map._v)))

                bo_timestamps_ns.append(t_curr)
                bo_yaw_rates.append(float(cal.gyro_vehicle[2]))
                bo_speeds.append(float(v_map_fwd))
                bo_dr_enu.append(fused_pure.position_enu_m[:2].copy())

        pure_pts = np.array(pure_pts)
        map_pts = np.array(map_pts)

        if len(pure_pts) < 2 or len(map_pts) < 2:
            return None

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

        return {
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
