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

from sih.core.contracts import GNSSSample
from sih.calibration.mount import MountAlignment
from server.session_core import SessionCore


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
                metas.append({
                    "id": sid,
                    "trip": trip,
                    "name": f"Scenario #{sid:02d}: {dom} ({trip}, {dur}s, {dist:.0f}m)",
                    "env": dom,
                    "duration_s": float(dur),
                    "distance_m": float(dist),
                    "target_drift_pct": 10.0,
                    "split": "Held-Out 20% / Unseen Trip",
                })
        except Exception as e:
            print(f"[Router] Failed loading scenarios_canonical.json: {e}")
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
        self.core = SessionCore(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
            saved_alignment=saved_alignment,
            road_network=road_network,
            domain=domain,
            engine_type=engine_type,
            use_speed_smoother=True,
        )

        self.client_websockets: Set[web.WebSocketResponse] = set()
        self.stream_websockets: Set[web.WebSocketResponse] = set()
        self.benchmark_task: Optional[asyncio.Task] = None
        self.loaded_trip: Any = None

    @property
    def engine(self):
        return self.core.engine

    @property
    def evaluator(self):
        return self.core.evaluator

    @property
    def state(self) -> str:
        return self.core.state

    @property
    def benchmark_active(self) -> bool:
        return self.core.benchmark_active

    @property
    def current_benchmark_scenario(self) -> Optional[int]:
        return self.core.current_benchmark_scenario

    @property
    def current_trip_name(self) -> str:
        return self.core.current_trip_name

    def set_blackout(self, active: bool = True, entry_gnss: Optional[GNSSSample] = None) -> None:
        self.core.set_blackout(active, entry_gnss=entry_gnss)

    def reset(self, clear_trip: bool = False) -> None:
        self.core.reset(clear_trip=clear_trip)
        if clear_trip:
            self.loaded_trip = None

    async def broadcast_hud(self) -> None:
        recipients = list(self.client_websockets | self.stream_websockets)
        if not recipients:
            return

        payload = json.dumps(self.core.get_hud())
        to_remove = set()
        for ws in recipients:
            try:
                await ws.send_str(payload)
            except Exception:
                to_remove.add(ws)
        self.client_websockets.difference_update(to_remove)
        self.stream_websockets.difference_update(to_remove)

    async def process_batch(self, batch_data: Dict[str, Any]) -> None:
        """
        Processes a single sensor batch JSON received from phone or replay.
        """
        self.core.push_batch(batch_data)
        await self.broadcast_hud()

    async def prepare_benchmark(self, scenario_id: int = 30) -> Dict[str, Any]:
        """
        Preloads scenario trip, road network, and alignment when benchmark suite is opened.
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

        if self.core.current_trip_name != trip_name or self.loaded_trip is None:
            print(f"[Benchmark] Preloading trip {trip_name} and OSM road network for Scenario #{scenario_id}...")
            t = load_any_trip(trip_path)
            self.loaded_trip = t

            calib_m = MountCalibrator(min_samples=30)
            idx_g = 0
            n_g = len(t.gnss_samples)
            for im in t.imu_samples:
                while idx_g < n_g and t.gnss_samples[idx_g].timestamp_ns <= im.timestamp_ns:
                    calib_m.observe_gnss(t.gnss_samples[idx_g])
                    idx_g += 1
                calib_m.update(im)
            saved_alignment = calib_m.alignment
            road_network, _ = load_trip_road_network(t, map_source="osm", cache_dir="data/maps/cache")
            trip_domain = "Mixed" if "S-S3" in trip_name else ("Highway" if "S-M" in trip_name else "Arterial")

            self.core.setup_benchmark_engine(
                ref_lat=t.reference_lat_deg,
                ref_lon=t.reference_lon_deg,
                ref_alt=0.0,
                saved_alignment=saved_alignment,
                road_network=road_network,
                domain=trip_domain,
                scenario_id=scenario_id,
                trip_name=trip_name,
            )
        else:
            self.core.current_benchmark_scenario = scenario_id
            self.core.benchmark_active = True

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
        align_to_use = calib_sc.alignment if (calib_sc.alignment and calib_sc.alignment.is_calibrated) else self.core.saved_alignment

        self.core.engine.saved_alignment = align_to_use
        self.core.engine.reset()
        if align_to_use is not None and align_to_use.is_calibrated:
            self.core.engine.calibrator._alignment = align_to_use
            self.core.engine.calibrator._yaw_locked = True
            self.core.engine.mount_reused = True

        self.core.evaluator.reset(keep_ref=True)
        self.core.evaluator.set_reference(self.core.ref_lat, self.core.ref_lon, self.core.ref_alt)
        self.core.state = "WARMING_UP"

        valid_gnss = [g for g in sliced_trip.gnss_samples if g.is_valid]
        if valid_gnss and hasattr(self.core.engine, "session"):
            self.core.engine.session.init_from_gnss(valid_gnss[0])

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

        self.core.benchmark_active = True
        self.core.current_benchmark_scenario = scenario_id
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
            print(f"[Benchmark] Replaying Scenario #{self.core.current_benchmark_scenario} ({len(batches)} batches @ {speed}x)...")
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
            print(f"[Benchmark] Scenario #{self.core.current_benchmark_scenario} replay finished successfully.")
        except asyncio.CancelledError:
            print(f"[Benchmark] Scenario replay cancelled.")
        except Exception as e:
            print(f"[Benchmark Error]: {e}")
        finally:
            if self.core.state == "BLACKOUT":
                self.core.set_blackout(False)
            self.core.benchmark_active = False
            self.core.is_preloaded_trip = False
            self.core.current_benchmark_scenario = None
            await self.broadcast_hud()

    async def stop_benchmark(self) -> Dict[str, Any]:
        if self.benchmark_task and not self.benchmark_task.done():
            self.benchmark_task.cancel()
        self.core.deload_benchmark()
        self.loaded_trip = None
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
