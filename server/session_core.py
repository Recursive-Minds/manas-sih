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
        if enable_handoff:
            try:
                from sih.handoff.manager import SeamlessGNSSHandoffManager, HandoffConfig
                self.handoff_manager = SeamlessGNSSHandoffManager(HandoffConfig())
            except Exception:
                self.handoff_manager = None

    def get_last_batch_timing(self) -> Dict[str, float]:
        """Returns the timing breakdown of the most recently processed batch."""
        return dict(self.last_batch_timing)

    def set_blackout(self, active: bool = True, entry_gnss: Optional[GNSSSample] = None) -> None:
        self.state = "BLACKOUT" if active else "WARMING_UP"
        self.engine.set_blackout(active, entry_gnss=entry_gnss)
        if active:
            ts_ns = entry_gnss.timestamp_ns if entry_gnss else None
            self.evaluator.start_blackout(timestamp_ns=ts_ns)
        else:
            self.evaluator.stop_blackout()

    def reset(self, clear_trip: bool = False, keep_ref: Optional[bool] = None) -> None:
        self.state = "WARMING_UP"
        self.benchmark_active = False
        self.is_preloaded_trip = False
        self.current_benchmark_scenario = None
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

    def set_map_matching(self, enabled: bool) -> None:
        """Enables or disables map matching in the underlying engine."""
        if hasattr(self.engine, "enable_map_matching"):
            self.engine.enable_map_matching = bool(enabled)
        if hasattr(self.engine, "session") and hasattr(self.engine.session, "enable_map_matching"):
            self.engine.session.enable_map_matching = bool(enabled)

    def prefetch_road_network(self, lat: float, lon: float, radius_m: float = 3000.0) -> Dict[str, Any]:
        """
        Prefetches road network geometry around (lat, lon) using Overpass client
        and assigns it to the engine.
        """
        try:
            from sih.map.osm_client import OSMOverpassClient
            client = OSMOverpassClient()
            net = client.fetch_road_network(lat, lon, radius_m=radius_m)
            if net is not None:
                self.engine.road_network = net
                if hasattr(self.engine, "session") and hasattr(self.engine.session, "road_network"):
                    self.engine.session.road_network = net
                return {
                    "success": True,
                    "segments": len(net.segments),
                    "message": f"Prefetched {len(net.segments)} segments ({radius_m:.0f}m radius)",
                }
            return {"success": False, "segments": 0, "message": "No roads returned"}
        except Exception as e:
            return {"success": False, "segments": 0, "message": str(e)}

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

        reconciled_dict = None
        if self.handoff_manager is not None and getattr(self.handoff_manager.reconciler, "is_active", False):
            try:
                from sih.data.geo import enu_to_geodetic
                p_disp = self.handoff_manager.reconciler.get_reconciled_position(time.time_ns())
                if p_disp is not None and self.ref_lat != 0.0:
                    r_lat, r_lon, _ = enu_to_geodetic(
                        p_disp[0], p_disp[1], p_disp[2],
                        self.ref_lat, self.ref_lon, self.ref_alt
                    )
                    reconciled_dict = {"lat": float(r_lat), "lon": float(r_lon)}
            except Exception:
                pass

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
            if self.handoff_manager is not None:
                try:
                    self.handoff_manager.on_gnss(gnss)
                except Exception:
                    pass

        # 2. State transition evaluation
        if "state" in batch_data:
            req_state = batch_data["state"]
            if req_state in ("WARMING_UP", "BLACKOUT") and req_state != self.state:
                entry_g = latest_g_in_batch or (self.evaluator.gnss_fixes[-1] if self.evaluator.gnss_fixes else None)
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
            fused = self.engine.on_imu(imu)
            if fused is not None:
                self.evaluator.on_dr(fused)
                if self.handoff_manager is not None:
                    try:
                        self.handoff_manager.on_fused_position(fused)
                    except Exception:
                        pass

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
        elif cmd == "get_hud":
            return self.get_hud()
        else:
            return {"status": "error", "message": f"Unknown control command: {cmd}"}

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
        )
        self.evaluator = LiveEvaluator(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
        )
        self.benchmark_active = True
        self.is_preloaded_trip = True

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
        )
        self.evaluator = LiveEvaluator(
            ref_lat=0.0,
            ref_lon=0.0,
            ref_alt=0.0,
        )
        self.state = "WARMING_UP"
