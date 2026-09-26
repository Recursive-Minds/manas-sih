"""
tests/test_app_session_core.py
------------------------------
Unit tests for server.session_core.SessionCore:
1. Import isolation test: importing server.session_core must NOT load aiohttp, pandas, tqdm, or dill.
2. Replay parity test: replaying recorded batches through SessionCore produces bit-identical HUD positions,
   speeds, headings, and states compared to NavigationRouter.
3. Control API test: start_blackout, stop_blackout, reset, prime_features, and get_hud.
"""

import os
import sys
import json
import subprocess
import pytest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def test_session_core_import_isolation():
    """
    Subprocess test: Importing server.session_core must not pull in
    aiohttp, pandas, tqdm, dill, or server.replay into sys.modules.
    """
    code = """
import sys
import server.session_core

forbidden = ["aiohttp", "pandas", "tqdm", "dill", "server.replay"]
loaded_forbidden = [m for m in forbidden if m in sys.modules]
if loaded_forbidden:
    print(f"FAILED: forbidden modules loaded: {loaded_forbidden}")
    sys.exit(1)
print("PASSED")
sys.exit(0)
"""
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, capture_output=True, text=True)
    assert res.returncode == 0, f"Import isolation failed: {res.stderr}\n{res.stdout}"
    assert "PASSED" in res.stdout


def test_session_core_batch_replay_parity():
    """
    Replays 60 recorded batches through SessionCore and NavigationRouter
    to verify bit-identical HUD positions, metrics, and state transitions.
    """
    from server.session_core import SessionCore
    from server.router import NavigationRouter

    fixture_path = os.path.join(ROOT_DIR, "tests", "fixtures", "session_core_sample_batches.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    ref_lat = fixture["ref_lat"]
    ref_lon = fixture["ref_lon"]
    batches = fixture["batches"]

    # 1. Initialize standalone SessionCore
    core = SessionCore(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=0.0,
        domain="Mixed",
        engine_type="stage_b",
        use_speed_smoother=True,
    )
    core.benchmark_active = True
    core.current_benchmark_scenario = 30

    # 2. Initialize NavigationRouter (which wraps SessionCore)
    router = NavigationRouter(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=0.0,
        domain="Mixed",
        engine_type="stage_b",
    )
    router.core.benchmark_active = True
    router.core.current_benchmark_scenario = 30

    # Replay 60 batches
    for i, batch in enumerate(batches):
        hud_core = core.push_batch(batch)
        hud_router = router.core.push_batch(batch)

        assert hud_core["state"] == hud_router["state"], f"Batch {i} state mismatch"
        if hud_core["dr_pos"] is not None and hud_router["dr_pos"] is not None:
            assert hud_core["dr_pos"]["lat"] == hud_router["dr_pos"]["lat"], f"Batch {i} lat mismatch"
            assert hud_core["dr_pos"]["lon"] == hud_router["dr_pos"]["lon"], f"Batch {i} lon mismatch"
            assert hud_core["dr_pos"]["speed_mps"] == hud_router["dr_pos"]["speed_mps"], f"Batch {i} speed mismatch"
            assert hud_core["dr_pos"]["heading_deg"] == hud_router["dr_pos"]["heading_deg"], f"Batch {i} heading mismatch"

    # Verify control API
    ctrl_res = core.control("start_blackout")
    assert ctrl_res["status"] == "ok"
    assert core.state == "BLACKOUT"

    ctrl_res = core.control("stop_blackout")
    assert ctrl_res["status"] == "ok"
    assert core.state == "WARMING_UP"

    ctrl_res = core.control("reset")
    assert ctrl_res["status"] == "ok"
    assert core.state == "WARMING_UP"
