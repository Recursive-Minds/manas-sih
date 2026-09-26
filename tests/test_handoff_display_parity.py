"""
tests/test_handoff_display_parity.py
------------------------------------
Verifies that Handoff FSM + Hermite reconciliation is DISPLAY ONLY:
The engine outputs, dead reckoning positions, and evaluation metrics
are 100% bit-identical with handoff enabled (ON) or disabled (OFF).
"""

import os
import sys
import json
import pytest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.session_core import SessionCore


def test_handoff_on_off_bit_identical_parity():
    """
    Replays recorded batches through two SessionCore instances:
    1. Instance A: enable_handoff = True
    2. Instance B: enable_handoff = False
    Asserts that all core engine outputs and metrics are bit-identical.
    """
    fixture_path = os.path.join(ROOT_DIR, "tests", "fixtures", "session_core_sample_batches.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    ref_lat = fixture["ref_lat"]
    ref_lon = fixture["ref_lon"]
    batches = fixture["batches"]

    # Instance A: Handoff ON
    session_on = SessionCore(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=0.0,
        enable_handoff=True,
    )

    # Instance B: Handoff OFF
    session_off = SessionCore(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=0.0,
        enable_handoff=False,
    )

    handoff_states_observed = []

    for i, batch in enumerate(batches):
        hud_on = session_on.push_batch(batch, source="device")
        hud_off = session_off.push_batch(batch, source="device")

        # 1. Assert state is identical
        assert hud_on["state"] == hud_off["state"], f"Batch {i}: state mismatch"

        # 2. Assert dr_pos is bit-identical
        dr_on = hud_on.get("dr_pos")
        dr_off = hud_off.get("dr_pos")
        if dr_on is None:
            assert dr_off is None, f"Batch {i}: dr_pos on is None but off is not None"
        else:
            assert dr_off is not None, f"Batch {i}: dr_pos off is None"
            assert dr_on["lat"] == dr_off["lat"], f"Batch {i}: DR lat mismatch: {dr_on['lat']} vs {dr_off['lat']}"
            assert dr_on["lon"] == dr_off["lon"], f"Batch {i}: DR lon mismatch: {dr_on['lon']} vs {dr_off['lon']}"
            assert dr_on["speed_mps"] == dr_off["speed_mps"], f"Batch {i}: DR speed mismatch"
            assert dr_on["heading_deg"] == dr_off["heading_deg"], f"Batch {i}: DR heading mismatch"

        # 3. Assert metrics are bit-identical
        m_on = hud_on.get("metrics")
        m_off = hud_off.get("metrics")
        if m_on is not None and m_off is not None:
            assert m_on["drift_pct"] == m_off["drift_pct"], f"Batch {i}: drift_pct mismatch"
            assert m_on["horizontal_error_m"] == m_off["horizontal_error_m"], f"Batch {i}: error mismatch"
            assert m_on["dr_dist_m"] == m_off["dr_dist_m"], f"Batch {i}: dr_dist mismatch"

        # 4. Check handoff state transitions in HUD warmup
        h_state = hud_on["warmup"]["handoff_state"]
        handoff_states_observed.append(h_state)

    # Verify that handoff state was active and tracked
    assert len(handoff_states_observed) == len(batches)
    assert "INITIALIZING" in handoff_states_observed or "GNSS_HEALTHY" in handoff_states_observed
    print(f"Handoff display parity: 100% bit-identical across {len(batches)} batches.")
