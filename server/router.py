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
import json
import asyncio
import argparse
from typing import Set, Optional, Dict, Any
import numpy as np
from aiohttp import web

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.core.contracts import IMUSample, GNSSSample
from sih.calibration.mount import MountAlignment
from server.engine_adapter import EngineAdapterStageA
from server.evaluator import LiveEvaluator


class NavigationRouter:
    def __init__(
        self,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        ref_alt: float = 0.0,
        saved_alignment: Optional[MountAlignment] = None,
    ) -> None:
        self.engine = EngineAdapterStageA(
            reference_lat_deg=ref_lat,
            reference_lon_deg=ref_lon,
            reference_alt_m=ref_alt,
            saved_alignment=saved_alignment,
        )
        self.evaluator = LiveEvaluator(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
        )
        self.state: str = "WARMING_UP"
        self.client_websockets: Set[web.WebSocketResponse] = set()

    def set_blackout(self, active: bool = True) -> None:
        self.state = "BLACKOUT" if active else "WARMING_UP"
        self.engine.set_blackout(active)
        if active:
            self.evaluator.start_blackout()
        else:
            self.evaluator.stop_blackout()

    def reset(self) -> None:
        self.state = "WARMING_UP"
        self.engine.reset()
        self.evaluator.reset()

    async def broadcast_hud(self) -> None:
        if not self.client_websockets:
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
            "mount_status": self.engine.get_mount_status(),
            "warmup": self.engine.get_warmup_status(),
            "dr_pos": dr_dict,
            "gnss_pos": gnss_dict,
            "metrics": self.evaluator.get_summary_dict(),
        }
        payload = json.dumps(msg)

        to_remove = set()
        for ws in self.client_websockets:
            try:
                await ws.send_str(payload)
            except Exception:
                to_remove.add(ws)
        self.client_websockets.difference_update(to_remove)

    async def process_batch(self, batch_data: Dict[str, Any]) -> None:
        """
        Processes a single sensor batch JSON received from phone or replay.
        """
        # Optional external state override from batch header
        if "state" in batch_data:
            req_state = batch_data["state"]
            if req_state in ("WARMING_UP", "BLACKOUT") and req_state != self.state:
                self.set_blackout(req_state == "BLACKOUT")

        # Ingest GNSS samples if present
        gnss_list = batch_data.get("gnss", [])
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
            # Ground truth evaluator ALWAYS receives GNSS
            self.evaluator.on_gnss(gnss)

            # Strict firewall: dead-reckoning engine receives GNSS ONLY when warming up
            if self.state == "WARMING_UP":
                self.engine.on_gnss(gnss)

        # Ingest IMU samples
        imu_list = batch_data.get("imu", [])
        for im_dict in imu_list:
            acc = im_dict["accel"]
            gyr = im_dict["gyro"]
            imu = IMUSample(
                timestamp_ns=int(im_dict["timestamp_ns"]),
                accel=np.array(acc, dtype=np.float64),
                gyro=np.array(gyr, dtype=np.float64),
            )
            fused = self.engine.on_imu(imu)
            if fused is not None:
                self.evaluator.on_dr(fused)

        # Broadcast update to web clients
        await self.broadcast_hud()


async def view_handler(request: web.Request) -> web.Response:
    index_path = os.path.join(os.path.dirname(__file__), "view", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return web.Response(text=content, content_type="text/html")
    return web.Response(text="Dashboard view not found", status=404)


async def ws_client_handler(request: web.Request) -> web.WebSocketResponse:
    router: NavigationRouter = request.app["router"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    router.client_websockets.add(ws)

    # Send initial state snapshot immediately
    await router.broadcast_hud()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "control":
                    cmd = data.get("command")
                    if cmd == "start_blackout":
                        router.set_blackout(True)
                    elif cmd == "stop_blackout":
                        router.set_blackout(False)
                    elif cmd == "reset":
                        router.reset()
                    await router.broadcast_hud()
    finally:
        router.client_websockets.discard(ws)
    return ws


async def ws_stream_handler(request: web.Request) -> web.WebSocketResponse:
    router: NavigationRouter = request.app["router"]
    ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
    await ws.prepare(request)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "sensor_batch":
                    await router.process_batch(data)
                elif data.get("type") == "control":
                    cmd = data.get("command")
                    if cmd == "start_blackout":
                        router.set_blackout(True)
                    elif cmd == "stop_blackout":
                        router.set_blackout(False)
                    elif cmd == "reset":
                        router.reset()
                    await router.broadcast_hud()
    finally:
        pass
    return ws


def create_app(router: Optional[NavigationRouter] = None) -> web.Application:
    app = web.Application()
    app["router"] = router or NavigationRouter()
    app.router.add_get("/", view_handler)
    app.router.add_get("/view", view_handler)
    app.router.add_get("/ws/client", ws_client_handler)
    app.router.add_get("/ws/stream", ws_stream_handler)
    return app


def main():
    parser = argparse.ArgumentParser(description="Smartphone IDR Streaming Server & Dashboard Router")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default 8765)")
    args = parser.parse_args()

    app = create_app()
    print(f"=== Starting Smartphone IDR Router on http://localhost:{args.port} ===")
    print(f"-> Web Dashboard:   http://localhost:{args.port}/view")
    print(f"-> Sensor Ingestion: ws://localhost:{args.port}/ws/stream")
    print(f"-> Client HUD:      ws://localhost:{args.port}/ws/client")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
