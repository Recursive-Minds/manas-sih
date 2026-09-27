"""
server/session_core.py
----------------------
Standalone session orchestrator for Smartphone Intelligent Dead Reckoning (IDR).

Responsibilities:
1. Session state machine: 'WARMING_UP' <-> 'BLACKOUT'.
2. Strict Zero-Future-Leak GNSS Firewall:
   - Evaluator ALWAYS receives ground-truth GNSS fixes (for live metrics).
   - Engine receives GNSS fixes ONLY while state == 'WARMING_UP'.
   - In state == 'BLACKOUT', GNSS fixes are strictly dropped before reaching the engine.
3. Decoupled Core:
   - Zero dependence on aiohttp, server.replay, or tile code.
   - Zero dependence on pandas, tqdm, or dill at import time.
   - Operates in pure Python/Kotlin embedding environments (e.g. Chaquopy).
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional, Dict, Any, List
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.core.contracts import IMUSample, GNSSSample
from sih.calibration.mount import MountAlignment
from server.engine_adapter import EngineAdapterStageA, EngineAdapterStageB
from server.evaluator import LiveEvaluator


class SessionCore:
    """
    Core IDR session manager controlling dead reckoning engine, evaluator,
    and state transitions.
    """

    def __init__(
        self,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        ref_alt: float = 0.0,
        saved_alignment: Optional[MountAlignment] = None,
        road_network: Any = None,
        domain: str = "Mixed",
        engine_type: str = "stage_b",
        ai_model: Any = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        device: Optional[Any] = None,
        use_speed_smoother: bool = True,
        predictor: Optional[Any] = None,
        enable_handoff: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        self.ref_alt = float(ref_alt)
        self.saved_alignment = saved_alignment
        self.road_network = road_network
        self.domain = domain
        self.engine_type = engine_type
        self.use_speed_smoother = use_speed_smoother
        self.predictor = predictor
        self.cache_dir = cache_dir or "data/maps/cache"

        if ai_model is not None:
            self.ai_model = ai_model
            self.norm_mean = norm_mean
            self.norm_std = norm_std
            self.device = device
        elif predictor is not None and not isinstance(predictor, str):
            self.ai_model, self.norm_mean, self.norm_std, self.device = None, None, None, None
        else:
            try:
                import torch
                from sih.models.inference import load_ai_model
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.ai_model, self.norm_mean, self.norm_std, _ = load_ai_model(self.device)
            except Exception:
                self.ai_model, self.norm_mean, self.norm_std, self.device = None, None, None, None

        if engine_type == "stage_a":
            self.engine = EngineAdapterStageA(
                reference_lat_deg=self.ref_lat,
                reference_lon_deg=self.ref_lon,
                reference_alt_m=self.ref_alt,
                saved_alignment=saved_alignment,
            )
        else:
            self.engine = EngineAdapterStageB(
                reference_lat_deg=self.ref_lat,
                reference_lon_deg=self.ref_lon,
                reference_alt_m=self.ref_alt,
                road_network=road_network,
                saved_alignment=saved_alignment,
                domain=domain,
                model=self.ai_model,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                device=self.device,
                decimate_gnss_for_seeding=False,
                lock_saved_alignment=(saved_alignment is not None),
                use_speed_smoother=self.use_speed_smoother,
                predictor=self.predictor,
            )

        self.evaluator = LiveEvaluator(
            ref_lat=self.ref_lat,
            ref_lon=self.ref_lon,
            ref_alt=self.ref_alt,
        )

        self.state: str = "WARMING_UP"
        self.last_replay_time: float = 0.0
        self.is_preloaded_trip: bool = (saved_alignment is not None or road_network is not None)
        self.benchmark_active: bool = False
        self.current_benchmark_scenario: Optional[int] = None
        self.current_trip_name: str = ""
        self.last_batch_timing: Dict[str, float] = {
            "features_ms": 0.0,
            "model_ms": 0.0,
            "ekf_map_ms": 0.0,
            "total_ms": 0.0,
        }
        self.enable_handoff = enable_handoff
        self.handoff_manager = None
        self.last_valid_gnss: Optional[GNSSSample] = None
        self.benchmark_batches: List[Dict[str, Any]] = []
        self.expected_metrics: Dict[str, Any] = {}
        self.map_matching_status_msg: str = "MAP MATCH: ON"
        # [DEMOFIX] handoff display + benchmark run state
        self._handoff_pending: bool = False
        self._handoff_target_enu = None
        self._last_dr_display_enu = None
        self._last_imu_ts = None
        self.benchmark_running: bool = False
        self._bundle_cache: Dict[str, Any] = {}
        if enable_handoff:
            try:
                from sih.handoff.manager import SeamlessGNSSHandoffManager, HandoffConfig
                self.handoff_manager = SeamlessGNSSHandoffManager(HandoffConfig(blend_duration_s=3.0))  # [DEMOFIX] visible blend
            except Exception:
                self.handoff_manager = None

    def get_last_batch_timing(self) -> Dict[str, float]:
        """Returns the timing breakdown of the most recently processed batch."""
        return dict(self.last_batch_timing)

    def set_blackout(self, active: bool = True, entry_gnss: Optional[GNSSSample] = None, timestamp_ns: Optional[int] = None) -> None:
        was_blackout = (self.state == "BLACKOUT")
        self.state = "BLACKOUT" if active else "WARMING_UP"
        self.engine.set_blackout(active, entry_gnss=entry_gnss)
        self._handoff_on_blackout(active, was_blackout)  # [DEMOFIX]
        if active:
            ts_ns = timestamp_ns or (entry_gnss.timestamp_ns if entry_gnss else None)
            self.evaluator.start_blackout(timestamp_ns=ts_ns)
        else:
            self.evaluator.stop_blackout(timestamp_ns=timestamp_ns or getattr(self, "benchmark_bo_end_ns", None))

    def reset(self, clear_trip: bool = False, keep_ref: Optional[bool] = None) -> None:
        self.state = "WARMING_UP"
        self.benchmark_active = False
        self.is_preloaded_trip = False
        self.current_benchmark_scenario = None
        self.benchmark_bo_start_ns = None
        self.benchmark_bo_end_ns = None
        if clear_trip:
            self.current_trip_name = ""
            self.ref_lat = 0.0
            self.ref_lon = 0.0
            self.ref_alt = 0.0
            self.evaluator.reset(keep_ref=False)
        else:
            kr = keep_ref if keep_ref is not None else (self.ref_lat != 0.0 or self.ref_lon != 0.0)
            self.evaluator.reset(keep_ref=kr)
        self.engine.reset()
        if self.handoff_manager is not None:
            self.handoff_manager.reset()
        self._handoff_pending = False  # [DEMOFIX]
        self._handoff_target_enu = None
        self._last_dr_display_enu = None
        self._last_imu_ts = None

    def set_map_matching(self, enabled: bool) -> None:
        """Enables or disables map matching in the underlying engine."""
        if hasattr(self.engine, "enable_map_matching"):
            self.engine.enable_map_matching = bool(enabled)
        if hasattr(self.engine, "session") and hasattr(self.engine.session, "enable_map_matching"):
            self.engine.session.enable_map_matching = bool(enabled)


    def get_mount_state(self) -> str:
        """Returns mount state string (UNLEVELLED / LEVELLED / YAW_LOCKED / REUSED)."""
        if hasattr(self.engine, "get_mount_state_string"):
            return self.engine.get_mount_state_string()
        return "UNKNOWN"

    def prime_features(
        self,
        imu_samples: List[IMUSample],
        calib_samples: Optional[List[Any]] = None,
    ) -> None:
        """Passthrough pre-rolling features into the underlying engine adapter."""
        if hasattr(self.engine, "prime_features"):
            self.engine.prime_features(imu_samples, calib_samples=calib_samples)


    # ------------------------------------------------------------------
    # [DEMOFIX] Handoff display wiring (DISPLAY ONLY - never feeds the engine)
    # Blackout is defined by the app state (START/STOP or the benchmark
    # batches), so the FSM is driven by set_blackout(), not by GNSS timeouts.
    # On exit: hold the DR marker until the first valid post-blackout fix,
    # then Hermite-blend the display from the last DR position to that fix.
    # ------------------------------------------------------------------
    def _handoff_ref(self):
        e = self.engine
        return (float(getattr(e, "ref_lat", self.ref_lat) or self.ref_lat),
                float(getattr(e, "ref_lon", self.ref_lon) or self.ref_lon),
                float(getattr(e, "ref_alt", self.ref_alt) or 0.0))

    def _to_enu(self, lat: float, lon: float) -> np.ndarray:
        from sih.data.geo import geodetic_to_enu
        rl, ro, ra = self._handoff_ref()
        return np.asarray(geodetic_to_enu(lat, lon, ra, rl, ro, ra), dtype=np.float64)

    def _handoff_on_blackout(self, active: bool, was_blackout: bool) -> None:
        hm = self.handoff_manager
        if hm is None:
            return
        from sih.handoff.manager import HandoffState
        if active:
            hm.reset()
            hm._state = HandoffState.INS_DEAD_RECKONING
            self._handoff_pending = False
            self._handoff_target_enu = None
        elif was_blackout:
            self._handoff_pending = True            # wait for first valid fix after exit
            hm._state = HandoffState.REACQUISITION_VERIFY

    def _handoff_on_gnss(self, gnss: GNSSSample) -> None:
        hm = self.handoff_manager
        if hm is None or not gnss.is_valid or self.state != "WARMING_UP":
            return
        from sih.handoff.manager import HandoffState
        if self._handoff_pending and self._last_dr_display_enu is not None:
            target = self._to_enu(gnss.latitude_deg, gnss.longitude_deg)
            hm.reconciler.initiate_blend(timestamp_ns=gnss.timestamp_ns,
                                         p_dead_reckoning=self._last_dr_display_enu.copy(),
                                         p_fused=target)
            self._handoff_target_enu = target
            self._handoff_pending = False
            hm._state = HandoffState.REACQUISITION_BLENDING if hm.reconciler.is_blending else HandoffState.GNSS_HEALTHY
        elif hm.reconciler.is_blending:
            self._handoff_target_enu = self._to_enu(gnss.latitude_deg, gnss.longitude_deg)
        elif hm._state in (HandoffState.INITIALIZING, HandoffState.GNSS_DEGRADED, HandoffState.REACQUISITION_VERIFY) and not self._handoff_pending:
            hm._state = HandoffState.GNSS_HEALTHY

    def _handoff_on_fused(self, fused: Any) -> None:
        self._last_dr_display_enu = self._to_enu(fused.latitude_deg, fused.longitude_deg)

    def _handoff_display(self) -> Optional[Dict[str, Any]]:
        hm = self.handoff_manager
        if hm is None or self._handoff_pending or self._handoff_target_enu is None or self._last_imu_ts is None:
            return None
        if not hm.reconciler.is_blending:
            return None
        from sih.handoff.manager import HandoffState
        from sih.data.geo import enu_to_geodetic
        p, alpha = hm.reconciler.get_blended_position(int(self._last_imu_ts), self._handoff_target_enu)
        if not hm.reconciler.is_blending:
            hm._state = HandoffState.GNSS_HEALTHY
            return None
        rl, ro, ra = self._handoff_ref()
        lat, lon, _ = enu_to_geodetic(p[0], p[1], p[2], rl, ro, ra)
        return {"lat": float(lat), "lon": float(lon), "alpha": float(alpha)}

    def _dr_visible(self) -> bool:
        hm = self.handoff_manager
        blending = bool(hm is not None and hm.reconciler.is_blending)
        return bool(self.state == "BLACKOUT" or self._handoff_pending or blending)

    def get_hud(self) -> Dict[str, Any]:
        """Returns structured HUD snapshot dictionary."""
        latest_dr = self.engine.latest_fused_position
        latest_gnss = self.evaluator.gnss_fixes[-1] if self.evaluator.gnss_fixes else None

        dr_dict = None
        if latest_dr is not None:
            dr_dict = {
                "lat": float(latest_dr.latitude_deg),
                "lon": float(latest_dr.longitude_deg),
                "speed_mps": round(float(self.evaluator.latest_metrics.dr_speed_mps), 2),
                "heading_deg": round(float(self.evaluator.latest_metrics.dr_heading_deg), 1),
            }

        gnss_dict = None
        if latest_gnss is not None:
            gnss_dict = {
                "lat": float(latest_gnss.latitude_deg),
                "lon": float(latest_gnss.longitude_deg),
                "speed_mps": round(float(latest_gnss.speed_mps or 0.0), 2),
                "bearing_deg": round(float(latest_gnss.bearing_deg or 0.0), 1),
            }

        warmup = self.engine.get_warmup_status()
        if self.handoff_manager is not None:
            warmup["handoff_state"] = self.handoff_manager.state.value
        else:
            warmup["handoff_state"] = "INITIALIZING"
        warmup["map_matching_enabled"] = getattr(self.engine, "enable_map_matching", True)
        warmup["map_matching_status"] = getattr(self, "map_matching_status_msg", "MAP MATCH: ON" if getattr(self.engine, "enable_map_matching", True) else "MAP MATCH: OFF")

        reconciled_dict = self._handoff_display()  # [DEMOFIX] sensor-time Hermite blend
        if self.handoff_manager is not None:
            warmup["handoff_state"] = self.handoff_manager.state.value
        if self.benchmark_active:
            warmup["speed_calib_display"] = "Speed calibration: trip history (bench bundle)"

        return {
            "type": "hud_update",
            "state": self.state,
            "benchmark_active": self.benchmark_active,
            "benchmark_scenario": self.current_benchmark_scenario,
            "mount_status": self.engine.get_mount_status(),
            "warmup": warmup,
            "dr_pos": dr_dict,
            "gnss_pos": gnss_dict,
            "reconciled_pos": reconciled_dict,
            "metrics": self.evaluator.get_summary_dict(),
            "dr_visible": self._dr_visible(),  # [DEMOFIX]
            "benchmark_running": bool(self.benchmark_running),  # [DEMOFIX]
        }

    def push_batch(self, batch_data: Dict[str, Any], source: str = "phone") -> Dict[str, Any]:
        """
        Processes a single sensor batch dictionary (IMU and/or GNSS).
        Enforces zero-future-leak firewall and state transitions.
        Returns the updated HUD dictionary.
        """
        source = batch_data.get("source", "device")

        # CRITICAL FIREWALL: While benchmark is active, strictly drop non-benchmark batches
        if self.benchmark_active and source != "benchmark":
            return self.get_hud()

        # When evaluating preloaded benchmark trip outside benchmark mode, drop local desk phone
        if self.is_preloaded_trip and not self.benchmark_active and source not in ("benchmark", "replay"):
            return self.get_hud()

        now = time.time()
        if source == "replay":
            if (now - self.last_replay_time) > 8.0:
                self.reset()
            self.last_replay_time = now

        t_batch_start = time.perf_counter()
        if hasattr(self.engine, "reset_batch_timing"):
            self.engine.reset_batch_timing()

        # 1. Ingest GNSS samples if present
        gnss_list = batch_data.get("gnss", [])
        latest_g_in_batch = None
        for g_dict in gnss_list:
            gnss = GNSSSample(
                timestamp_ns=int(g_dict["timestamp_ns"]),
                latitude_deg=float(g_dict["latitude_deg"]),
                longitude_deg=float(g_dict["longitude_deg"]),
                altitude_m=float(g_dict.get("altitude_m", 0.0)),
                speed_mps=float(g_dict["speed_mps"]) if g_dict.get("speed_mps") is not None else None,
                bearing_deg=float(g_dict["bearing_deg"]) if g_dict.get("bearing_deg") is not None else None,
                accuracy_h_m=float(g_dict.get("accuracy_h_m", 5.0)),
                is_valid=bool(g_dict.get("is_valid", True)),
            )
            self.evaluator.on_gnss(gnss)

            if self.state == "WARMING_UP":
                self.engine.on_gnss(gnss)
            latest_g_in_batch = gnss
            if gnss.is_valid:
                self.last_valid_gnss = gnss
            self._handoff_on_gnss(gnss)  # [DEMOFIX] display-only

            if gnss.is_valid and (self.road_network is None or len(self.road_network.segments) == 0):
                try:
                    from sih.map.cache import SpatialDiskCache
                    c_dir = getattr(self, "cache_dir", "data/maps/cache")
                    cache = SpatialDiskCache(cache_dir=c_dir)
                    if cache.is_area_cached(gnss.latitude_deg, gnss.longitude_deg, 1500.0):
                        cached_rnet = cache.get_combined_network_for_radius(
                            gnss.latitude_deg, gnss.longitude_deg, 1500.0,
                            ref_lat=self.ref_lat if self.ref_lat != 0.0 else gnss.latitude_deg,
                            ref_lon=self.ref_lon if self.ref_lon != 0.0 else gnss.longitude_deg,
                        )
                        if len(cached_rnet.segments) > 0:
                            self.road_network = cached_rnet
                            if hasattr(self.engine, "update_road_network"):
                                self.engine.update_road_network(cached_rnet, self.ref_lat or gnss.latitude_deg, self.ref_lon or gnss.longitude_deg)
                            elif hasattr(self.engine, "road_network"):
                                self.engine.road_network = cached_rnet
                                self.engine.enable_map_matching = True
                except Exception:
                    pass

        # 2. State transition evaluation
        if "state" in batch_data:
            req_state = batch_data["state"]
            if req_state in ("WARMING_UP", "BLACKOUT") and req_state != self.state:
                entry_g = (
                    latest_g_in_batch
                    or self.last_valid_gnss
                    or (self.evaluator.gnss_fixes[-1] if self.evaluator.gnss_fixes else None)
                )
                self.set_blackout(req_state == "BLACKOUT", entry_gnss=entry_g)

        # 3. Ingest IMU samples
        imu_list = batch_data.get("imu", [])
        for im_dict in imu_list:
            acc = im_dict["accel"]
            gyr = im_dict["gyro"]
            imu = IMUSample(
                timestamp_ns=int(im_dict["timestamp_ns"]),
                accel=np.array([float(acc[0]), float(acc[1]), float(acc[2])], dtype=np.float64),
                gyro=np.array([float(gyr[0]), float(gyr[1]), float(gyr[2])], dtype=np.float64),
            )
            self._last_imu_ts = imu.timestamp_ns  # [DEMOFIX]
            fused = self.engine.on_imu(imu)
            if fused is not None:
                self.evaluator.on_dr(fused)
                self._handoff_on_fused(fused)  # [DEMOFIX] display-only

        t_batch_end = time.perf_counter()
        total_ms = (t_batch_end - t_batch_start) * 1000.0

        timing = {
            "features_ms": 0.0,
            "model_ms": 0.0,
            "ekf_map_ms": 0.0,
            "total_ms": float(total_ms),
        }
        if hasattr(self.engine, "get_batch_timing"):
            split = self.engine.get_batch_timing()
            timing.update(split)
            timing["total_ms"] = float(total_ms)

        self.last_batch_timing = timing
        hud = self.get_hud()
        hud["timing"] = self.last_batch_timing
        return hud

    def control(self, cmd: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a control command on the session core.
        """
        if args is None:
            args = {}

        if cmd == "start_blackout":
            self.set_blackout(True)
            return {"status": "ok", "state": self.state}
        elif cmd == "stop_blackout":
            self.set_blackout(False)
            return {"status": "ok", "state": self.state}
        elif cmd == "set_blackout":
            active = bool(args.get("active", True))
            entry_gnss = args.get("entry_gnss")
            self.set_blackout(active=active, entry_gnss=entry_gnss)
            return {"status": "ok", "state": self.state}
        elif cmd == "reset":
            clear_trip = bool(args.get("clear_trip", False))
            keep_ref = args.get("keep_ref")
            self.reset(clear_trip=clear_trip, keep_ref=keep_ref)
            return {"status": "ok", "state": self.state}
        elif cmd == "prime_features":
            imu_samples = args.get("imu_samples", [])
            calib_samples = args.get("calib_samples")
            self.prime_features(imu_samples, calib_samples=calib_samples)
            return {"status": "ok"}
        elif cmd == "set_map_matching":
            enabled = bool(args.get("enabled", True))
            self.set_map_matching(enabled)
            return {"status": "ok", "enabled": enabled}
        elif cmd == "prefetch_road_network":
            lat = float(args.get("lat", self.ref_lat))
            lon = float(args.get("lon", self.ref_lon))
            radius_m = float(args.get("radius_m", 3000.0))
            cache_dir = args.get("cache_dir")
            return self.prefetch_road_network(lat, lon, radius_m, cache_dir)
        elif cmd == "get_hud":
            return self.get_hud()
        else:
            return {"status": "error", "message": f"Unknown control command: {cmd}"}


    def prefetch_road_network(
        self,
        lat: float,
        lon: float,
        radius_m: float = 3000.0,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prefetches road network around (lat, lon) within radius_m.
        Stores into cache_dir (or self.cache_dir).
        Updates self.road_network and self.engine.
        """
        t0 = time.perf_counter()
        c_dir = cache_dir or getattr(self, "cache_dir", "data/maps/cache")
        try:
            from sih.map.cache import SpatialDiskCache
            from sih.map.hybrid_provider import HybridIndiaMapProvider
            cache = SpatialDiskCache(cache_dir=c_dir)
            gis_dir = os.path.join(c_dir, "indian_gis")
            provider = HybridIndiaMapProvider(
                disk_cache=cache,
                enable_live_osm=True,
                cache_dir=c_dir,
                gis_data_dir=gis_dir,
            )
            rnet = provider.get_corridor_network(lat=lat, lon=lon, radius_m=radius_m)
            fetch_time_s = time.perf_counter() - t0
            seg_count = len(rnet.segments)
            cache_size_bytes = cache.get_cache_size_bytes()

            if seg_count > 0:
                self.road_network = rnet
                if hasattr(self.engine, "update_road_network"):
                    self.engine.update_road_network(rnet, self.ref_lat or lat, self.ref_lon or lon)
                elif hasattr(self.engine, "road_network"):
                    self.engine.road_network = rnet
                    self.engine.enable_map_matching = True
                return {
                    "success": True,
                    "message": f"Prefetched {seg_count} segments ({fetch_time_s:.2f}s, cache: {cache_size_bytes / 1024:.1f} KB)",
                    "segment_count": seg_count,
                    "fetch_time_s": fetch_time_s,
                    "cache_size_bytes": cache_size_bytes,
                }
            else:
                return {
                    "success": False,
                    "message": f"No segments found ({fetch_time_s:.2f}s)",
                    "segment_count": 0,
                    "fetch_time_s": fetch_time_s,
                    "cache_size_bytes": cache_size_bytes,
                }
        except Exception as e:
            import traceback
            print(f"[Prefetch] Exception during prefetch: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Prefetch error: {e}",
                "segment_count": 0,
                "fetch_time_s": time.perf_counter() - t0,
                "cache_size_bytes": 0,
            }

    def setup_benchmark_engine(
        self,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
        saved_alignment: Optional[MountAlignment],
        road_network: Any,
        domain: str,
        scenario_id: int,
        trip_name: str,
    ) -> None:
        """Sets up benchmark engine and evaluator for a specific scenario."""
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.saved_alignment = saved_alignment
        self.road_network = road_network
        self.domain = domain
        self.current_trip_name = trip_name
        self.current_benchmark_scenario = scenario_id

        self.engine = EngineAdapterStageB(
            reference_lat_deg=ref_lat,
            reference_lon_deg=ref_lon,
            reference_alt_m=ref_alt,
            road_network=road_network,
            saved_alignment=saved_alignment,
            domain=domain,
            model=self.ai_model,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            device=self.device,
            decimate_gnss_for_seeding=False,
            lock_saved_alignment=True,
            use_speed_smoother=self.use_speed_smoother,
            predictor=self.predictor,
        )

        if road_network is None or len(getattr(road_network, "segments", [])) == 0:
            import logging
            logging.warning("[Benchmark] WARNING: No road network loaded for Scenario #%s. MAP MATCH: OFF (no roads for this area)", scenario_id)
            print(f"[Benchmark] WARNING: No road network loaded for Scenario #{scenario_id}. MAP MATCH: OFF (no roads for this area)")
            self.engine.enable_map_matching = False
            self.map_matching_status_msg = "MAP MATCH: OFF (no roads for this area)"
        else:
            self.engine.enable_map_matching = True
            self.map_matching_status_msg = f"MAP MATCH: ON ({len(road_network.segments)} segments)"

        self.evaluator = LiveEvaluator(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
        )
        self.benchmark_active = True
        self.is_preloaded_trip = True

    def setup_benchmark_from_bundle(self, bundle_source: Any) -> Dict[str, Any]:
        """
        Loads and sets up a complete, deterministic, parity-tested benchmark session
        from an exported benchmark bundle (bench_<id>.bin or dict).
        
        Args:
            bundle_source: File path (str), raw gzip bytes (bytes), or parsed bundle dict.
            
        Returns:
            Dict containing scenario metadata, expected metrics, and batch count.
        """
        import gzip
        import json
        try:
            from scipy.spatial.transform import Rotation as R
        except ImportError:
            from sih.core.scipy_shim import Rotation as R
        from sih.map.network import RoadNetwork, RoadSegment
        from sih.calibration.mount import MountAlignment, CalibratedSample

        if isinstance(bundle_source, str):
            if not os.path.exists(bundle_source):
                raise FileNotFoundError(f"Benchmark bundle not found: {bundle_source}")
            _key = f"{bundle_source}:{os.path.getsize(bundle_source)}"  # [DEMOFIX] parse once, re-setup fast
            if _key not in self._bundle_cache:
                with gzip.open(bundle_source, "rt", encoding="utf-8") as f:
                    self._bundle_cache = {_key: json.load(f)}
            bundle = self._bundle_cache[_key]
        elif isinstance(bundle_source, bytes):
            bundle = json.loads(gzip.decompress(bundle_source).decode("utf-8"))
        elif isinstance(bundle_source, dict):
            bundle = bundle_source
        else:
            raise TypeError(f"Unsupported bundle_source type: {type(bundle_source)}")

        scenario_id = int(bundle["scenario_id"])
        trip_name = str(bundle.get("trip", ""))
        domain = str(bundle.get("domain", "Mixed"))
        ref_lat = float(bundle["reference_lat_deg"])
        ref_lon = float(bundle["reference_lon_deg"])
        ref_alt = float(bundle.get("reference_alt_m", 0.0))
        self.benchmark_bo_start_ns = bundle.get("bo_start_ns")
        self.benchmark_bo_end_ns = bundle.get("bo_end_ns")

        # 1. Deserialize MountAlignment
        align_dict = bundle.get("saved_alignment")
        saved_alignment = None
        if align_dict:
            rot = R.from_quat(align_dict["quat"])
            saved_alignment = MountAlignment(
                is_calibrated=bool(align_dict.get("is_calibrated", True)),
                R_phone_to_vehicle=rot,
                forward_axis_phone=np.array(align_dict["forward_axis_phone"], dtype=np.float64),
                lateral_axis_phone=np.array(align_dict["lateral_axis_phone"], dtype=np.float64),
                vertical_axis_phone=np.array(align_dict["vertical_axis_phone"], dtype=np.float64),
                yaw_axis_index=int(align_dict["yaw_axis_index"]),
                yaw_axis_sign=float(align_dict["yaw_axis_sign"]),
                mount_yaw_offset_rad=float(align_dict.get("mount_yaw_offset_rad", 0.0)),
                pitch_deg=float(align_dict.get("pitch_deg", 0.0)),
                roll_deg=float(align_dict.get("roll_deg", 0.0)),
            )

        # 2. Deserialize RoadNetwork
        rnet_dict = bundle.get("road_network")
        road_network = None
        if rnet_dict and "segments" in rnet_dict:
            cell_size = float(rnet_dict.get("cell_size_m", 100.0))
            road_network = RoadNetwork(cell_size_m=cell_size)
            for s in rnet_dict["segments"]:
                seg = RoadSegment(
                    segment_id=str(s["id"]),
                    start_enu_m=np.array(s["s_enu"], dtype=np.float64),
                    end_enu_m=np.array(s["e_enu"], dtype=np.float64),
                    start_lat_lon=(float(s["s_ll"][0]), float(s["s_ll"][1])),
                    end_lat_lon=(float(s["e_ll"][0]), float(s["e_ll"][1])),
                    bearing_deg=float(s["brg"]),
                    length_m=float(s["len"]),
                    road_type=str(s.get("type", "motorway")),
                    speed_limit_mps=float(s.get("spd", 25.0)),
                    is_oneway=bool(s.get("ow", False)),
                    start_node_id=s.get("sn"),
                    end_node_id=s.get("en"),
                )
                road_network.add_segment(seg)

        # 3. Setup Benchmark Engine
        self.setup_benchmark_engine(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
            saved_alignment=saved_alignment,
            road_network=road_network,
            domain=domain,
            scenario_id=scenario_id,
            trip_name=trip_name,
        )

        # 3b. Load normalization vectors from bundle
        if "norm_mean" in bundle and "norm_std" in bundle:
            nm = np.array(bundle["norm_mean"], dtype=np.float32).reshape(-1, 1)
            ns = np.array(bundle["norm_std"], dtype=np.float32).reshape(-1, 1)
            self.norm_mean = nm
            self.norm_std = ns
            if hasattr(self.engine, "norm_mean"):
                self.engine.norm_mean = nm
                self.engine.norm_std = ns

        # 4. Initialize engine from warmup GNSS
        warmup_g_dict = bundle.get("warmup_gnss")
        if warmup_g_dict:
            warmup_g = GNSSSample(
                timestamp_ns=int(warmup_g_dict["t"]),
                latitude_deg=float(warmup_g_dict["lat"]),
                longitude_deg=float(warmup_g_dict["lon"]),
                altitude_m=float(warmup_g_dict.get("alt", 0.0)),
                speed_mps=float(warmup_g_dict["spd"]) if warmup_g_dict.get("spd") is not None else None,
                bearing_deg=float(warmup_g_dict["brg"]) if warmup_g_dict.get("brg") is not None else None,
                accuracy_h_m=float(warmup_g_dict.get("acc", 5.0)),
                is_valid=bool(warmup_g_dict.get("valid", True)),
            )
            self.engine.session.init_from_gnss(warmup_g)
            self.evaluator.on_gnss(warmup_g)
            self.last_valid_gnss = warmup_g

        # 5. Prime feature extractor with pre-roll IMU + calib
        preroll_imu_dicts = bundle.get("preroll_imu", [])
        preroll_calib_dicts = bundle.get("preroll_calib", [])
        if preroll_imu_dicts:
            preroll_imu = [
                IMUSample(
                    timestamp_ns=int(im["t"]),
                    accel=np.array(im["a"], dtype=np.float64),
                    gyro=np.array(im["g"], dtype=np.float64),
                )
                for im in preroll_imu_dicts
            ]
            R_mat = saved_alignment.R_phone_to_vehicle.as_matrix() if saved_alignment else np.eye(3, dtype=np.float64)
            grav_v = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
            preroll_calib = [
                CalibratedSample(
                    timestamp_ns=int(c["t"]),
                    accel_vehicle=np.array(c["av"], dtype=np.float64),
                    gyro_vehicle=np.array(c["gv"], dtype=np.float64),
                    rotation_body_to_vehicle=R_mat,
                    gravity_vehicle=grav_v,
                    is_calibrated=True,
                )
                for c in preroll_calib_dicts
            ]
            _rp = bundle.get("preroll_v_raw")  # [DEMOFIX] exporter-computed speeds -> seconds, not minutes
            self.engine.prime_features(preroll_imu, calib_samples=preroll_calib, raw_preds=_rp)

        # 6. Seed GNSS history buffer for T7 online speed calibration
        gnss_history_dicts = bundle.get("gnss_history", [])
        gnss_history = [
            GNSSSample(
                timestamp_ns=int(g["t"]),
                latitude_deg=float(g["lat"]),
                longitude_deg=float(g["lon"]),
                altitude_m=float(g.get("alt", 0.0)),
                speed_mps=float(g["spd"]) if g.get("spd") is not None else None,
                bearing_deg=float(g["brg"]) if g.get("brg") is not None else None,
                accuracy_h_m=float(g.get("acc", 5.0)),
                is_valid=bool(g.get("valid", True)),
            )
            for g in gnss_history_dicts
        ]
        self.engine.recent_gnss_window = list(gnss_history)
        if gnss_history:
            self.engine.moving_gnss_fixes_count = max(len(gnss_history), 180)
            pass  # [DEMOFIX] history feeds the engine only; the evaluator (truth display) starts at the warm-up fix

        # Store batches in memory ready for replay
        self.benchmark_batches = bundle.get("batches", [])
        self.state = "WARMING_UP"  # [DEMOFIX] every setup starts a clean run
        if self.handoff_manager is not None:
            self.handoff_manager.reset()
        self._handoff_pending = False
        self._handoff_target_enu = None
        self._last_dr_display_enu = None
        self._last_imu_ts = None
        self.benchmark_running = False
        self.expected_metrics = bundle.get("expected", {})

        return {
            "success": True,
            "scenario_id": scenario_id,
            "trip": trip_name,
            "segments_count": len(road_network.segments) if road_network else 0,
            "batches_count": len(self.benchmark_batches),
            "expected_error_m": self.expected_metrics.get("endpoint_error_m", 0.0),
            "expected_drift_pct": self.expected_metrics.get("drift_pct", 0.0),
            "gt_dist_m": self.expected_metrics.get("gt_dist_m", 0.0),
        }

    def setup_or_restore_benchmark(self, bundle_path: str) -> Dict[str, Any]:
        """[DEMOFIX] First call per bundle: full setup, then snapshot engine+evaluator.
        Later calls (every RUN): restore the snapshot in < 1 s -> each run starts clean."""
        import copy
        key = f"{bundle_path}:{os.path.getsize(bundle_path)}"
        snap = getattr(self, "_bench_snapshot", None)
        if snap is None or snap["key"] != key:
            res = self.setup_benchmark_from_bundle(bundle_path)
            memo = {id(self.predictor): self.predictor}
            if self.road_network is not None:
                memo[id(self.road_network)] = self.road_network
            attrs = {k: getattr(self, k, None) for k in self._BENCH_ATTRS}
            self._bench_snapshot = {"key": key, "state": copy.deepcopy((self.engine, self.evaluator), memo),
                                    "attrs": attrs, "res": res}
            return res
        memo = {id(self.predictor): self.predictor}
        if snap["attrs"].get("road_network") is not None:
            memo[id(snap["attrs"]["road_network"])] = snap["attrs"]["road_network"]
        self.engine, self.evaluator = copy.deepcopy(snap["state"], memo)
        for k, v in snap["attrs"].items():
            setattr(self, k, v)
        self.state = "WARMING_UP"
        if self.handoff_manager is not None:
            self.handoff_manager.reset()
        self._handoff_pending = False
        self._handoff_target_enu = None
        self._last_dr_display_enu = None
        self._last_imu_ts = None
        self.benchmark_running = False
        return snap["res"]

    _BENCH_ATTRS = ("ref_lat", "ref_lon", "ref_alt", "saved_alignment", "road_network", "domain",
                    "current_trip_name", "current_benchmark_scenario", "benchmark_active", "is_preloaded_trip",
                    "benchmark_bo_start_ns", "benchmark_bo_end_ns", "norm_mean", "norm_std",
                    "last_valid_gnss", "benchmark_batches", "expected_metrics")

    def get_benchmark_batch_count(self) -> int:
        """Returns the number of loaded benchmark batches."""
        return len(self.benchmark_batches)

    def push_benchmark_batch_index(self, index: int) -> Dict[str, Any]:
        """Replays the benchmark batch at index and returns updated HUD dictionary."""
        if 0 <= index < len(self.benchmark_batches):
            self.benchmark_running = bool(index < len(self.benchmark_batches) - 1)
            return self.push_batch(self.benchmark_batches[index], source="benchmark")
        return self.get_hud()

    def deload_benchmark(self) -> None:
        """Restores live mode engine and evaluator."""
        self.benchmark_active = False
        self.is_preloaded_trip = False
        self.current_benchmark_scenario = None
        self.current_trip_name = ""
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self.ref_alt = 0.0
        self.engine = EngineAdapterStageB(
            reference_lat_deg=0.0,
            reference_lon_deg=0.0,
            reference_alt_m=0.0,
            road_network=None,
            saved_alignment=None,
            domain="Mixed",
            model=self.ai_model,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            device=self.device,
            decimate_gnss_for_seeding=False,
            lock_saved_alignment=False,
            use_speed_smoother=self.use_speed_smoother,
            predictor=self.predictor,
        )
        self.evaluator = LiveEvaluator(
            ref_lat=0.0,
            ref_lon=0.0,
            ref_alt=0.0,
        )
        self.state = "WARMING_UP"
