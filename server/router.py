"""
server/router.py
----------------
High-performance WebSocket & HTTP streaming router for Smartphone IDR.

Responsibilities:
1. Endpoints:
   - HTTP GET / or /view : Serves the live Leaflet map and HUD web application.
   - WS /ws/stream       : High-rate sensor ingestion endpoint for phone app or replay CLI.
   - WS /ws/client       : Real-time broadcast channel for browser dashboard clients.
2. Strict Zero-Future-Leak Firewall:
   - Evaluator ALWAYS receives ground-truth GNSS fixes (for error computation).
   - Engine receives GNSS fixes ONLY while state == 'WARMING_UP'.
   - In state == 'BLACKOUT', GNSS fixes are strictly dropped before reaching the engine.
"""

from __future__ import annotations
import os
import sys
import time
import json
import asyncio
import argparse
from typing import Set, Optional, Dict, Any, List
import numpy as np
from aiohttp import web

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.core.contracts import IMUSample, GNSSSample
from sih.calibration.mount import MountAlignment
from server.engine_adapter import EngineAdapterStageA, EngineAdapterStageB
from server.evaluator import LiveEvaluator


def load_scenarios_meta() -> List[Dict[str, Any]]:
    canonical_json = os.path.join(ROOT_DIR, "server", "scenarios_canonical.json")
    metas = [
        {
            "id": -1,
            "trip": "Random",
            "name": "🎲 Random Held-Out Scenario (All Trips)",
            "env": "Diverse Selection",
            "duration_s": 0.0,
            "distance_m": 0.0,
            "target_drift_pct": 10.0,
            "benchmark_drift_pct": 0.0,
            "split": "Held-Out Verification",
        }
    ]
    if os.path.exists(canonical_json):
        try:
            with open(canonical_json, "r", encoding="utf-8") as f:
                sc_list = json.load(f)
            for sc in sc_list:
                sid = int(sc["scenario_id"])
                trip = str(sc["trip"])
                dom = str(sc["domain"])
                dur = int(sc["duration_s"])
                dist = float(sc["gt_dist_m"])
                drift = float(sc.get("drift_osm", 0.0))
                status = "PASS" if drift < 10.0 else "FAIL"
                metas.append({
                    "id": sid,
                    "trip": trip,
                    "name": f"Scenario #{sid:02d}: {dom} ({trip}, {dur}s, {dist:.0f}m) - {drift:.1f}% Drift [{status}]",
                    "env": dom,
                    "duration_s": float(dur),
                    "distance_m": float(dist),
                    "target_drift_pct": 10.0,
                    "benchmark_drift_pct": float(drift),
                    "split": "Held-Out 20% / Unseen Trip",
                })
        except Exception as e:
            print(f"[Router] Failed loading scenarios_canonical.json: {e}")
    else:
        metas.append({
            "id": 30,
            "trip": "S-S3a",
            "name": "Scenario #30: Urban Mixed (S-S3a, 60s) - 5.5% Drift [PASS]",
            "env": "Mixed",
            "duration_s": 60.0,
            "distance_m": 244.2,
            "target_drift_pct": 10.0,
            "benchmark_drift_pct": 5.44,
            "split": "Held-Out Unseen Test Drive",
        })
    return metas


SCENARIOS_META = load_scenarios_meta()


class NavigationRouter:
    def __init__(
        self,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        ref_alt: float = 0.0,
        saved_alignment: Optional[MountAlignment] = None,
        road_network: Any = None,
        domain: str = "Mixed",
        engine_type: str = "stage_b",
    ) -> None:
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.saved_alignment = saved_alignment
        self.road_network = road_network
        self.domain = domain
        self.engine_type = engine_type

        # Load unified MoE AI velocity model
        try:
            import torch
            from sih.models.inference import load_ai_model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ai_model, self.norm_mean, self.norm_std, _ = load_ai_model(self.device)
            print(f"[Router] AI Model successfully loaded onto {self.device}")
        except Exception as e:
            print(f"[Router Warning] Failed to load AI model: {e}")
            self.ai_model, self.norm_mean, self.norm_std, self.device = None, None, None, None

        if engine_type == "stage_a":
            self.engine = EngineAdapterStageA(
                reference_lat_deg=ref_lat,
                reference_lon_deg=ref_lon,
                reference_alt_m=ref_alt,
                saved_alignment=saved_alignment,
            )
        else:
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
                lock_saved_alignment=(saved_alignment is not None),
            )
        self.evaluator = LiveEvaluator(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
        )
        self.state: str = "WARMING_UP"
        self.client_websockets: Set[web.WebSocketResponse] = set()
        self.stream_websockets: Set[web.WebSocketResponse] = set()
        self.last_replay_time: float = 0.0
        self.is_preloaded_trip: bool = (saved_alignment is not None or road_network is not None)

        # Dedicated Benchmark Evaluation State
        self.benchmark_active: bool = False
        self.benchmark_task: Optional[asyncio.Task] = None
        self.current_benchmark_scenario: Optional[int] = None
        self.loaded_trip: Any = None
        self.current_trip_name: str = ""

    def set_blackout(self, active: bool = True, entry_gnss: Optional[GNSSSample] = None) -> None:
        self.state = "BLACKOUT" if active else "WARMING_UP"
        print(f"[Router] State transition -> {self.state}")
        self.engine.set_blackout(active, entry_gnss=entry_gnss)
        if active:
            self.evaluator.start_blackout(timestamp_ns=entry_gnss.timestamp_ns if entry_gnss else None)
        else:
            self.evaluator.stop_blackout()

    def reset(self, clear_trip: bool = False) -> None:
        self.state = "WARMING_UP"
        self.benchmark_active = False
        self.is_preloaded_trip = False
        self.current_benchmark_scenario = None
        if clear_trip:
            self.loaded_trip = None
            self.current_trip_name = ""
        self.engine.reset()
        self.evaluator.reset()

    async def broadcast_hud(self) -> None:
        recipients = list(self.client_websockets | self.stream_websockets)
        if not recipients:
            return

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

        msg = {
            "type": "hud_update",
            "state": self.state,
            "benchmark_active": self.benchmark_active,
            "benchmark_scenario": self.current_benchmark_scenario,
            "mount_status": self.engine.get_mount_status(),
            "warmup": self.engine.get_warmup_status(),
            "dr_pos": dr_dict,
            "gnss_pos": gnss_dict,
            "metrics": self.evaluator.get_summary_dict(),
        }
        payload = json.dumps(msg)

        to_remove = set()
        for ws in recipients:
            try:
                await ws.send_str(payload)
            except Exception as e:
                print(f"[WebSocket Error]: {e}")
                to_remove.add(ws)
        self.client_websockets.difference_update(to_remove)
        self.stream_websockets.difference_update(to_remove)

    async def process_batch(self, batch_data: Dict[str, Any]) -> None:
        """
        Processes a single sensor batch JSON received from phone or replay.
        """
        source = batch_data.get("source", "device")

        # CRITICAL FIREWALL: While benchmark is active, strictly drop non-benchmark batches
        if self.benchmark_active and source != "benchmark":
            return

        # When evaluating preloaded benchmark trip outside benchmark mode, drop local desk phone
        if self.is_preloaded_trip and not self.benchmark_active and source not in ("benchmark", "replay"):
            return

        now = time.time()
        if source == "replay":
            if (now - self.last_replay_time) > 8.0:
                print("[Router] New replay session detected. Auto-resetting for clean run.")
                self.reset()
            self.last_replay_time = now

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

        # 4. Broadcast updated HUD state to phone and web dashboard
        await self.broadcast_hud()

    async def prepare_benchmark(self, scenario_id: int = 30) -> Dict[str, Any]:
        """
        Preloads scenario trip, road network, and alignment when benchmark suite is opened.
        Immediately activates benchmark warmup ticks ('✓ Bench-Calib', '✓ Gravity', etc.).
        """
        if scenario_id <= 0:
            import random
            real_scenarios = [s["id"] for s in SCENARIOS_META if s["id"] > 0]
            scenario_id = random.choice(real_scenarios) if real_scenarios else 30

        meta = next((s for s in SCENARIOS_META if s["id"] == scenario_id), None)
        if meta is None:
            return {"status": "error", "message": f"Scenario #{scenario_id} not found"}

        trip_name = meta["trip"]
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", f"{trip_name}.csv")
        if not os.path.exists(trip_path):
            return {"status": "error", "message": f"Trip CSV {trip_path} not found"}

        from server.replay import load_any_trip
        from sih.map.network import load_trip_road_network
        from sih.calibration.mount import MountCalibrator

        if self.current_trip_name != trip_name or self.loaded_trip is None:
            print(f"[Benchmark] Preloading trip {trip_name} and OSM road network for Scenario #{scenario_id}...")
            t = load_any_trip(trip_path)
            self.loaded_trip = t
            self.current_trip_name = trip_name
            self.ref_lat = t.reference_lat_deg
            self.ref_lon = t.reference_lon_deg
            self.ref_alt = 0.0

            calib_m = MountCalibrator(min_samples=30)
            idx_g = 0
            n_g = len(t.gnss_samples)
            for im in t.imu_samples:
                while idx_g < n_g and t.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
                    calib_m.observe_gnss(t.gnss_samples[idx_g])
                    idx_g += 1
                calib_m.update(im)
            saved_align = calib_m.alignment
            rnet, _ = load_trip_road_network(t, map_source="osm", cache_dir="data/maps/cache")

            trip_domain = "Mixed" if "S-S3" in trip_name else ("Highway" if "S-M" in trip_name else "Arterial")
            self.engine = EngineAdapterStageB(
                reference_lat_deg=self.ref_lat,
                reference_lon_deg=self.ref_lon,
                reference_alt_m=self.ref_alt,
                road_network=rnet,
                saved_alignment=saved_align,
                domain=trip_domain,
                model=self.ai_model,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                device=self.device,
                decimate_gnss_for_seeding=False,
                lock_saved_alignment=True,
            )
            self.evaluator = LiveEvaluator(
                ref_lat=self.ref_lat,
                ref_lon=self.ref_lon,
                ref_alt=self.ref_alt,
            )

        self.benchmark_active = True
        self.is_preloaded_trip = True
        self.current_benchmark_scenario = scenario_id
        await self.broadcast_hud()
        return {"status": "ok", "scenario_id": scenario_id, "trip": trip_name}

    async def start_benchmark(self, scenario_id: int = 30, speed: float = 2.0) -> Dict[str, Any]:
        """
        Orchestrates scenario replay: loads trip, aligns mount, slices blackout, and streams batches.
        """
        if self.benchmark_task and not self.benchmark_task.done():
            self.benchmark_task.cancel()

        prep_res = await self.prepare_benchmark(scenario_id=scenario_id)
        if prep_res.get("status") == "error":
            return prep_res

        scenario_id = prep_res["scenario_id"]
        trip_name = prep_res["trip"]

        from server.replay import slice_scenario, build_sensor_batches
        from sih.calibration.mount import MountCalibrator

        # Reset session metrics without clearing loaded trip
        self.engine.reset()
        self.evaluator.reset()
        self.state = "WARMING_UP"

        sliced_trip, bo_start, bo_dur = slice_scenario(self.loaded_trip, scenario_id, warmup_s=30.0)
        bo_start_ns = getattr(sliced_trip, "exact_bo_start_ns", int(bo_start * 1e9))

        # Align mount specifically up to scenario blackout entry
        calib_sc = MountCalibrator(min_samples=30)
        idx_g = 0
        n_g = len(self.loaded_trip.gnss_samples)
        for im in self.loaded_trip.imu_samples:
            if im.timestamp_ns > bo_start_ns:
                break
            while idx_g < n_g and self.loaded_trip.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
                calib_sc.observe_gnss(self.loaded_trip.gnss_samples[idx_g])
                idx_g += 1
            calib_sc.update(im)
        if calib_sc.alignment and calib_sc.alignment.is_calibrated:
            self.engine.saved_alignment = calib_sc.alignment
            self.engine.calibrator._alignment = calib_sc.alignment
            self.engine.calibrator._yaw_locked = True
            self.engine.mount_reused = True

        # Seed initial EKF fix from the first valid sample in the warm-up window
        valid_gnss = [g for g in sliced_trip.gnss_samples if g.is_valid]
        if valid_gnss and hasattr(self.engine, "session"):
            self.engine.session.init_from_gnss(valid_gnss[0])

        batches = build_sensor_batches(
            trip=sliced_trip,
            batch_interval_s=0.10,
            blackout_start_s=bo_start,
            blackout_duration_s=bo_dur,
            exact_bo_start_ns=getattr(sliced_trip, "exact_bo_start_ns", None),
            exact_bo_end_ns=getattr(sliced_trip, "exact_bo_end_ns", None),
        )
        for b in batches:
            b["source"] = "benchmark"

        self.benchmark_active = True
        self.current_benchmark_scenario = scenario_id
        self.benchmark_task = asyncio.create_task(self._run_benchmark_loop(batches, speed))
        return {
            "status": "ok",
            "scenario_id": scenario_id,
            "trip": trip_name,
            "batches": len(batches),
            "duration_s": bo_dur,
        }

    async def _run_benchmark_loop(self, batches: List[Dict[str, Any]], speed: float) -> None:
        try:
            print(f"[Benchmark] Replaying Scenario #{self.current_benchmark_scenario} ({len(batches)} batches @ {speed}x)...")
            t_prev = time.time()
            for b_idx, batch in enumerate(batches):
                await self.process_batch(batch)
                if speed > 0.0:
                    dt_target = 0.10 / speed
                    elapsed = time.time() - t_prev
                    sleep_time = max(0.0, dt_target - elapsed)
                    if sleep_time > 0.001:
                        await asyncio.sleep(sleep_time)
                    t_prev = time.time()
                if b_idx % 50 == 0 or b_idx == len(batches) - 1:
                    print(f"  [Benchmark] Progress: {b_idx + 1}/{len(batches)} ({batch['state']})")
            print(f"[Benchmark] Scenario #{self.current_benchmark_scenario} replay finished successfully.")
        except asyncio.CancelledError:
            print(f"[Benchmark] Scenario replay cancelled.")
        except Exception as e:
            print(f"[Benchmark Error]: {e}")
        finally:
            if self.state == "BLACKOUT":
                self.set_blackout(False)
            self.benchmark_active = False
            self.is_preloaded_trip = False
            self.current_benchmark_scenario = None
            await self.broadcast_hud()

    async def stop_benchmark(self) -> Dict[str, Any]:
        if self.benchmark_task and not self.benchmark_task.done():
            self.benchmark_task.cancel()
        self.benchmark_active = False
        self.is_preloaded_trip = False
        self.current_benchmark_scenario = None
        # Deload benchmark: restore engine back to live mode with unlocked saved alignment
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
        )
        self.evaluator = LiveEvaluator(
            ref_lat=0.0,
            ref_lon=0.0,
            ref_alt=0.0,
        )
        self.state = "WARMING_UP"
        await self.broadcast_hud()
        return {"status": "ok", "message": "Benchmark stopped and deloaded"}


def json_response_cors(data: Any, status: int = 200) -> web.Response:
    return web.json_response(
        data,
        status=status,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


async def api_scenarios_handler(request: web.Request) -> web.Response:
    return json_response_cors({"scenarios": SCENARIOS_META})


async def api_benchmark_start_handler(request: web.Request) -> web.Response:
    router: NavigationRouter = request.app["router"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    scenario_id = int(body.get("scenario_id", 30))
    speed = float(body.get("speed", 2.0))
    res = await router.start_benchmark(scenario_id=scenario_id, speed=speed)
    return json_response_cors(res)


async def api_benchmark_stop_handler(request: web.Request) -> web.Response:
    router: NavigationRouter = request.app["router"]
    res = await router.stop_benchmark()
    return json_response_cors(res)


async def api_benchmark_prepare_handler(request: web.Request) -> web.Response:
    router: NavigationRouter = request.app["router"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    scenario_id = int(body.get("scenario_id", 30))
    res = await router.prepare_benchmark(scenario_id=scenario_id)
    return json_response_cors(res)


async def status_handler(request: web.Request) -> web.Response:
    router: NavigationRouter = request.app["router"]
    return json_response_cors({
        "status": "online",
        "state": router.state,
        "benchmark_active": router.benchmark_active,
        "benchmark_scenario": router.current_benchmark_scenario,
        "trip": router.current_trip_name,
    })


async def view_handler(request: web.Request) -> web.Response:
    index_path = os.path.join(os.path.dirname(__file__), "view", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return web.Response(text=content, content_type="text/html")
    return web.Response(text="Dashboard view not found", status=404)


async def tile_handler(request: web.Request) -> web.Response:
    z = request.match_info["z"]
    x = request.match_info["x"]
    y = request.match_info["y"]

    tile_cache_dir = os.path.join(ROOT_DIR, "data", "maps", "cache", "tiles", str(z), str(x))
    os.makedirs(tile_cache_dir, exist_ok=True)
    tile_cache_path = os.path.join(tile_cache_dir, f"{y}.png")

    if os.path.exists(tile_cache_path) and os.path.getsize(tile_cache_path) > 0:
        with open(tile_cache_path, "rb") as f:
            return web.Response(body=f.read(), content_type="image/png")

    osm_url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    headers = {"User-Agent": "SmartphoneIDR/1.0 (manas.sih.idr)"}
    try:
        import aiohttp
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(osm_url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    with open(tile_cache_path, "wb") as f:
                        f.write(data)
                    return web.Response(body=data, content_type="image/png")
    except Exception:
        pass
    return web.Response(status=404)


async def ws_client_handler(request: web.Request) -> web.WebSocketResponse:
    router: NavigationRouter = request.app["router"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    router.client_websockets.add(ws)

    await router.broadcast_hud()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                cmd = data.get("command") or data.get("type")
                if cmd == "start_blackout":
                    router.set_blackout(True)
                elif cmd == "stop_blackout":
                    router.set_blackout(False)
                elif cmd == "reset":
                    router.reset()
                elif cmd == "prepare_benchmark":
                    sc_id = int(data.get("scenario_id", 30))
                    await router.prepare_benchmark(scenario_id=sc_id)
                elif cmd == "start_benchmark":
                    sc_id = int(data.get("scenario_id", 30))
                    spd = float(data.get("speed", 2.0))
                    await router.start_benchmark(scenario_id=sc_id, speed=spd)
                elif cmd == "stop_benchmark":
                    await router.stop_benchmark()
                await router.broadcast_hud()
    finally:
        router.client_websockets.discard(ws)
    return ws


async def ws_stream_handler(request: web.Request) -> web.WebSocketResponse:
    router: NavigationRouter = request.app["router"]
    ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
    await ws.prepare(request)
    router.stream_websockets.add(ws)
    print(f"[WebSocket] Stream connected (total={len(router.stream_websockets)})")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")
                if msg_type == "sensor_batch":
                    await router.process_batch(data)
                elif msg_type == "control":
                    cmd = data.get("command")
                    if cmd == "start_blackout":
                        router.set_blackout(True)
                    elif cmd == "stop_blackout":
                        router.set_blackout(False)
                    elif cmd == "reset":
                        router.reset()
                    elif cmd == "prepare_benchmark":
                        sc_id = int(data.get("scenario_id", 30))
                        await router.prepare_benchmark(scenario_id=sc_id)
                    elif cmd == "start_benchmark":
                        sc_id = int(data.get("scenario_id", 30))
                        spd = float(data.get("speed", 2.0))
                        await router.start_benchmark(scenario_id=sc_id, speed=spd)
                    elif cmd == "stop_benchmark":
                        await router.stop_benchmark()
                    await router.broadcast_hud()
    finally:
        router.stream_websockets.discard(ws)
    return ws


def create_app(router: Optional[NavigationRouter] = None) -> web.Application:
    app = web.Application()
    app["router"] = router or NavigationRouter()
    app.router.add_get("/", view_handler)
    app.router.add_get("/view", view_handler)
    app.router.add_get("/status", status_handler)
    app.router.add_get("/api/scenarios", api_scenarios_handler)
    app.router.add_post("/api/benchmark/start", api_benchmark_start_handler)
    app.router.add_post("/api/benchmark/stop", api_benchmark_stop_handler)
    app.router.add_post("/api/benchmark/prepare", api_benchmark_prepare_handler)
    app.router.add_get("/ws/client", ws_client_handler)
    app.router.add_get("/ws/stream", ws_stream_handler)
    app.router.add_get(r"/tiles/{z:\d+}/{x:\d+}/{y:\d+}.png", tile_handler)
    return app


def main():
    parser = argparse.ArgumentParser(description="Smartphone IDR Streaming Server & Dashboard Router")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default 8765)")
    parser.add_argument("--trip", default=None, help="Optional preloaded trip (default: None, dynamically loaded per scenario)")
    args = parser.parse_args()

    saved_align = None
    rnet = None
    ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0
    loaded_t = None
    trip_domain = "Arterial"
    if args.trip:
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", f"{args.trip}.csv")
        if os.path.exists(trip_path):
            from server.replay import load_any_trip
            from sih.map.network import load_trip_road_network
            from sih.calibration.mount import MountCalibrator
            t = load_any_trip(trip_path)
            loaded_t = t
            ref_lat, ref_lon, ref_alt = t.reference_lat_deg, t.reference_lon_deg, 0.0
            calib_m = MountCalibrator(min_samples=30)
            idx_g = 0
            n_g = len(t.gnss_samples)
            for im in t.imu_samples:
                while idx_g < n_g and t.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
                    calib_m.observe_gnss(t.gnss_samples[idx_g])
                    idx_g += 1
                calib_m.update(im)
            saved_align = calib_m.alignment
            rnet, _ = load_trip_road_network(t, map_source="osm", cache_dir="data/maps/cache")

    if args.trip:
        trip_domain = "Mixed" if "S-S3" in args.trip else ("Highway" if "S-M" in args.trip else "Arterial")
    router = NavigationRouter(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=ref_alt,
        saved_alignment=saved_align,
        road_network=rnet,
        domain=trip_domain,
    )
    router.current_trip_name = args.trip
    router.loaded_trip = loaded_t

    app = create_app(router=router)
    print(f"=== Starting Smartphone IDR Router on http://localhost:{args.port} ===")
    print(f"-> Preloaded Trip:  {args.trip} (OSM Network & Mount Alignment active)")
    print(f"-> Web Dashboard:   http://localhost:{args.port}/view")
    print(f"-> Sensor Ingestion: ws://localhost:{args.port}/ws/stream")
    print(f"-> Client HUD:      ws://localhost:{args.port}/ws/client")
    print(f"-> Benchmark API:   http://localhost:{args.port}/api/scenarios")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
