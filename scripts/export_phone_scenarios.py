"""
scripts/export_phone_scenarios.py
---------------------------------
Generates lightweight pre-sliced benchmark scenario JSON bundles
for canonical test scenarios (1, 22, 23, 25, 26, 30) and copies
them directly into android/app/src/ondevice/assets/ for autonomous on-device replay.
"""

from __future__ import annotations
import os
import sys
import json
from typing import Dict, Any, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from server.replay import slice_scenario, build_sensor_batches

ASSETS_DIR = os.path.join(ROOT_DIR, "android", "app", "src", "ondevice", "assets")
TRIPS_DIR = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips")


def export_scenarios():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    loader = GenericDataLoader()

    # Load canonical scenario specifications
    canonical_json = os.path.join(ROOT_DIR, "server", "scenarios_canonical.json")
    with open(canonical_json, "r", encoding="utf-8") as f:
        canonical_specs = json.load(f)

    target_scenario_ids = [1, 22, 23, 25, 26, 30]
    cached_trips: Dict[str, Any] = {}

    for sc_id in target_scenario_ids:
        spec = next((s for s in canonical_specs if s["scenario_id"] == sc_id), None)
        if not spec:
            print(f"Warning: Scenario {sc_id} not found in canonical specs!")
            continue

        trip_name = spec["trip"]
        if trip_name not in cached_trips:
            csv_path = os.path.join(TRIPS_DIR, f"{trip_name}.csv")
            if not os.path.exists(csv_path):
                print(f"Skipping scenario {sc_id}: {csv_path} not found")
                continue
            print(f"Loading {trip_name}.csv ...")
            cached_trips[trip_name] = loader.load_file(csv_path)

        trip = cached_trips[trip_name]

        # Slice scenario with 30s warmup
        sliced_trip, bo_start_s, bo_dur_s = slice_scenario(trip, sc_id, warmup_s=30.0)

        # Build 100 ms batches
        batches = build_sensor_batches(
            trip=sliced_trip,
            batch_interval_s=0.10,
            blackout_start_s=bo_start_s,
            blackout_duration_s=bo_dur_s,
            exact_bo_start_ns=getattr(sliced_trip, "exact_bo_start_ns", None),
            exact_bo_end_ns=getattr(sliced_trip, "exact_bo_end_ns", None),
        )

        for b in batches:
            b["source"] = "benchmark"

        payload = {
            "scenario_id": sc_id,
            "trip": trip_name,
            "domain": spec.get("domain", "Mixed"),
            "reference_lat_deg": float(trip.reference_lat_deg),
            "reference_lon_deg": float(trip.reference_lon_deg),
            "reference_alt_m": float(trip.reference_alt_m),
            "total_batches": len(batches),
            "blackout_duration_s": bo_dur_s,
            "batches": batches,
        }

        out_path = os.path.join(ASSETS_DIR, f"scenario_{sc_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

        size_kb = os.path.getsize(out_path) / 1024
        print(f"  -> Generated {out_path}: {len(batches)} batches ({size_kb:.1f} KB)")

    print("All canonical scenarios exported to Android assets successfully.")


if __name__ == "__main__":
    export_scenarios()
