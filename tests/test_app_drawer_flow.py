"""
tests/test_app_drawer_flow.py  [DEMOFIX]
Simulates EXACTLY what the on-device benchmark drawer does (LocalChaquopyEngineBridge):
  setup_or_restore_benchmark(bundle) -> push_benchmark_batch_index(0..N-1)   (RUN)
  ... and RUN again, and again after deload_benchmark (close/reopen drawer).
Guards the demo path so a fix elsewhere cannot silently break it.
"""
import glob
import gzip
import json
import math
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLES = sorted(glob.glob(os.path.join(ROOT, "android", "app", "src", "ondevice", "assets", "bench_*.bin")))


def _hav(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    return 2 * R * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


@pytest.fixture(scope="module")
def core():
    from sih.models.predictor import create_predictor
    from server.session_core import SessionCore
    pred = create_predictor("torch", device="cpu")
    return SessionCore(0.0, 0.0, 0.0, None, None, "Mixed", "stage_b", None, None, None, None, True, pred, True, "/tmp/idr_test_maps")


def _run(core, path):
    core.setup_or_restore_benchmark(path)
    n = core.get_benchmark_batch_count()
    assert n > 0
    return [core.push_benchmark_batch_index(i) for i in range(n)]


@pytest.mark.parametrize("path", BUNDLES, ids=[os.path.basename(p) for p in BUNDLES])
def test_drawer_flow(core, path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        b = json.load(f)
    exp = float(b["expected"]["endpoint_error_m"])
    assert len(b.get("preroll_v_raw", [])) == len(b["preroll_imu"]), "bundle must carry preroll_v_raw (fast setup)"

    runs = [_run(core, path), _run(core, path)]
    core.deload_benchmark()
    runs.append(_run(core, path))

    finals = []
    for hs in runs:
        s = hs[-1]["metrics"]["session_summary"]
        finals.append(float(s["final_error_m_raw"]))
        # 1. numbers = laptop expected
        assert abs(float(s["final_error_m"]) - exp) <= 0.05
        # 2. DR marker moves while the car moves (>= 90% distinct positions over moving steps)
        bo = [h for h in hs if h["state"] == "BLACKOUT" and h.get("dr_pos")]
        moving = [h for h in bo if (h.get("dr_pos", {}).get("speed_mps") or 0.0) >= 0.5]
        if len(moving) > 20:
            distinct = len({(h["dr_pos"]["lat"], h["dr_pos"]["lon"]) for h in moving})
            assert distinct >= 0.9 * len(moving), f"DR marker frozen: {distinct}/{len(moving)}"
        # 3. handoff: blackout -> verify -> blend -> healthy, blend visible, DR hidden at the end
        states = []
        for h in hs:
            st = h["warmup"]["handoff_state"]
            if not states or states[-1] != st:
                states.append(st)
        for st in ("INS_DEAD_RECKONING", "REACQUISITION_BLENDING", "GNSS_HEALTHY"):
            assert st in states, f"handoff never reached {st}: {states}"
        assert sum(1 for h in hs if h.get("reconciled_pos")) >= 10
        assert hs[-1]["dr_visible"] is False
        # 4. displayed marker never jumps > 25 m between updates during the handoff blend
        blend = [(h["reconciled_pos"]["lat"], h["reconciled_pos"]["lon"]) for h in hs if h.get("reconciled_pos")]
        for i in range(1, len(blend)):
            assert _hav(blend[i - 1], blend[i]) <= 25.0
        # 5. run-state flag for the RUN button
        assert hs[1]["benchmark_running"] is True and hs[-1]["benchmark_running"] is False
    # 6. every run identical (no state leaks between runs)
    assert max(finals) - min(finals) <= 1e-6, finals
