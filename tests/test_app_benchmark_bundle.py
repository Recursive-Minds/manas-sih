"""
tests/test_app_benchmark_bundle.py
----------------------------------
Automated test suite verifying that on-device benchmark bundles (.bin)
replayed through SessionCore produce exact parity with laptop benchmarks:
- Mount state is REUSED (mount alignment locked)
- Pre-roll features primed and GNSS history >= 180s
- Road network loaded and map-matching active
- Endpoint error matches expected target within 0.05 m
"""

import os
import pytest
from server.session_core import SessionCore

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "android", "app", "src", "ondevice", "assets"))


@pytest.mark.parametrize("scenario_id, expected_err_m, expected_drift_pct", [
    (1, 80.59, 26.73),
    (22, 16.77, 3.53),
    (23, 67.76, 6.01),
    (25, 77.30, 12.58),
    (26, 122.80, 13.75),
    (30, 7.04, 2.88),
])
def test_benchmark_bundle_replay_parity(scenario_id: int, expected_err_m: float, expected_drift_pct: float):
    bundle_path = os.path.join(ASSETS_DIR, f"bench_{scenario_id}.bin")
    assert os.path.exists(bundle_path), f"Bundle not found: {bundle_path}"

    session = SessionCore()
    setup_res = session.setup_benchmark_from_bundle(bundle_path)
    assert setup_res["success"] is True

    # 1. Verify Warmup HUD invariants
    hud = session.get_hud()
    warmup_info = hud.get("warmup", {})
    mount_state = warmup_info.get("mount_state")
    assert mount_state == "REUSED", f"Expected mount_state REUSED, got {mount_state}"
    assert "REUSED" in hud["mount_status"]

    speed_calib_s = warmup_info.get("speed_calib_s", 0)
    assert speed_calib_s >= 180, f"Expected speed_calib_s >= 180, got {speed_calib_s}"

    map_status = warmup_info.get("map_matching_status", "")
    assert "MAP MATCH: ON" in map_status, f"Expected MAP MATCH: ON, got {map_status}"
    assert setup_res["segments_count"] > 0, f"Expected segments > 0, got {setup_res['segments_count']}"

    # 2. Replay all batches
    total_batches = session.get_benchmark_batch_count()
    assert total_batches > 0, "No batches to replay"

    last_hud = None
    for i in range(total_batches):
        last_hud = session.push_benchmark_batch_index(i)

    # 3. Verify Blackout Completion & Metrics
    summary = session.evaluator.last_completed_summary
    assert summary is not None, "Evaluation summary was not completed at end of blackout"

    final_err = summary["final_error_m"]
    drift_pct = summary["drift_pct"]

    # Verify endpoint error within 0.05 m
    err_diff = abs(final_err - expected_err_m)
    assert err_diff <= 0.05, f"Scenario {scenario_id} final error {final_err:.2f} m deviated from expected {expected_err_m:.2f} m by {err_diff:.3f} m"

    # Verify drift within 0.05 pp
    drift_diff = abs(drift_pct - expected_drift_pct)
    assert drift_diff <= 0.05, f"Scenario {scenario_id} drift {drift_pct:.2f}% deviated from expected {expected_drift_pct:.2f}% by {drift_diff:.3f}%"
