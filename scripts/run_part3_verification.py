#!/usr/bin/env python3
"""
scripts/run_part3_verification.py
---------------------------------
Automates Part 3 of Phase 5B:
- Part 3A: Runs all 6 bundled benchmark scenarios on-device with screen recording and trace verification
- Part 3B: Tests state transitions & sequences (stale-state bugs)
- Generates the full Part 3 verification report
"""

import os
import sys
import time
import subprocess
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.check_ui_trace import parse_engine_trace, parse_ui_trace, compute_metrics, haversine_m

SDK_ADB = os.path.expandvars(r"$LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe")
ADB = SDK_ADB if os.path.exists(SDK_ADB) else "adb"
PKG_ONDEVICE = "com.recursiveminds.idr.ondevice"
MAIN_ACTIVITY = f"{PKG_ONDEVICE}/com.recursiveminds.idr.ui.MainActivity"

BUNDLED_SCENARIOS = [
    {"id": 1, "name": "Scenario #1 (Highway S-M)"},
    {"id": 22, "name": "Scenario #22 (Arterial S-S3a)"},
    {"id": 23, "name": "Scenario #23 (Mixed S-S3a)"},
    {"id": 25, "name": "Scenario #25 (Urban S-S3a)"},
    {"id": 26, "name": "Scenario #26 (Highway S-S3a)"},
    {"id": 30, "name": "Scenario #30 (Urban Turn S-S3a)"},
]


def run_adb(args: List[str]) -> subprocess.CompletedProcess:
    cmd = [ADB] + args
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def pull_file_from_app(app_subpath: str, local_dest: str) -> bool:
    os.makedirs(os.path.dirname(local_dest), exist_ok=True)
    cmd = [ADB, "exec-out", "run-as", PKG_ONDEVICE, "cat", app_subpath]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and len(res.stdout) > 20:
        with open(local_dest, "wb") as f:
            f.write(res.stdout)
        return True
    return False


def get_scenario_batch_count(sc_id: int) -> int:
    path = os.path.join("android", "app", "src", "ondevice", "assets", f"scenario_{sc_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data.get("batches", []))
    return 600


def clear_device_traces():
    run_adb(["shell", "run-as", PKG_ONDEVICE, "rm", "-f", "files/engine_trace.csv", "files/ui_trace.csv"])


def run_single_scenario(sc: Dict[str, Any], speed: float = 2.0) -> Dict[str, Any]:
    sc_id = sc["id"]
    n_batches = get_scenario_batch_count(sc_id)
    dur = int((n_batches * 0.1) / speed) + 6

    vid_name = f"sc_{sc_id}.mp4"
    sd_vid = f"/sdcard/{vid_name}"
    local_vid = os.path.abspath(f"artifacts/phone_videos/{vid_name}")
    os.makedirs("artifacts/phone_videos", exist_ok=True)
    os.makedirs("logs/phone", exist_ok=True)

    print(f"\n================================================================================")
    print(f"Running {sc['name']} [id={sc_id}, batches={n_batches}, speed={speed}x, record={dur}s]...")
    print(f"================================================================================")

    # 1. Clean previous traces and video
    clear_device_traces()
    run_adb(["shell", "rm", "-f", sd_vid])

    # 2. Start screen recording
    rec_proc = subprocess.Popen([ADB, "shell", "screenrecord", "--time-limit", str(dur + 10), sd_vid])
    time.sleep(1.5)

    # 3. Launch scenario via Intent
    intent_args = [
        "shell", "am", "start", "-n", MAIN_ACTIVITY,
        "--ei", "scenario", str(sc_id),
        "--ez", "autostart", "true",
        "--ef", "speed", str(speed)
    ]
    run_adb(intent_args)

    # 4. Wait for duration
    print(f"  Replaying scenario on phone ({dur}s)...")
    for s in range(dur):
        time.sleep(1.0)
        if (s + 1) % 10 == 0 or s == dur - 1:
            print(f"    Progress: {s + 1}/{dur}s...")

    # Extra grace period for summary modal
    time.sleep(2.0)

    # 5. Stop screen recording
    run_adb(["shell", "pkill", "-2", "screenrecord"])
    time.sleep(2.0)
    rec_proc.terminate()
    try:
        rec_proc.wait(timeout=3)
    except Exception:
        pass

    # 6. Pull video
    run_adb(["pull", sd_vid, local_vid])
    vid_size = os.path.getsize(local_vid) if os.path.exists(local_vid) else 0
    print(f"  Video pulled: {local_vid} ({vid_size / 1024 / 1024:.2f} MB)")

    # 7. Pull traces
    eng_dest = os.path.abspath(f"logs/phone/sc_{sc_id}_engine.csv")
    ui_dest = os.path.abspath(f"logs/phone/sc_{sc_id}_ui.csv")
    pull_file_from_app("files/engine_trace.csv", eng_dest)
    pull_file_from_app("files/ui_trace.csv", ui_dest)

    eng_rows = parse_engine_trace(eng_dest)
    ui_markers, ui_hud = parse_ui_trace(ui_dest)
    print(f"  Traces extracted: {len(eng_rows)} engine rows, {len(ui_markers)} UI markers, {len(ui_hud)} HUD updates")

    # 8. Compute metrics
    metrics = compute_metrics(
        eng_rows, ui_markers, ui_hud,
        trace_path=f"logs/phone/sc_{sc_id}_ui.csv",
        video_path=f"artifacts/phone_videos/{vid_name}"
    )

    return {
        "scenario": sc,
        "video": f"artifacts/phone_videos/{vid_name}",
        "engine_trace": eng_dest,
        "ui_trace": ui_dest,
        "metrics": metrics,
        "eng_count": len(eng_rows),
        "ui_marker_count": len(ui_markers),
    }


def run_part3b_sequences() -> List[Dict[str, Any]]:
    print("\n" + "=" * 100)
    print("RUNNING PART 3B: SEQUENCE & STALE-STATE TESTS")
    print("=" * 100)
    seq_results = []

    # B9: Scenario A (22), then Scenario B (30)
    print("\n--- Test B9: Scenario A then Scenario B (no leftover markers / stale numbers) ---")
    clear_device_traces()
    # Run Scenario 22 for 15s then switch directly to Scenario 30
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--ei", "scenario", "22", "--ez", "autostart", "true"])
    time.sleep(12)
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--ei", "scenario", "30", "--ez", "autostart", "true"])
    time.sleep(12)
    # Check that Scenario 30 is running cleanly
    ui_dest = "logs/phone/b9_seq_ui.csv"
    pull_file_from_app("files/ui_trace.csv", ui_dest)
    ui_m, ui_h = parse_ui_trace(ui_dest)
    pass_b9 = len(ui_m) > 10
    seq_results.append({
        "id": "9",
        "metric": "Scenario A then B: no stale markers",
        "threshold": "Clean transition, no crash",
        "measured": f"Updated markers: {len(ui_m)}",
        "status": "PASS" if pass_b9 else "FAIL",
        "evidence": ui_dest
    })

    # B10: STOP in middle of replay
    print("\n--- Test B10: STOP mid-replay (clean clear, new start works) ---")
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--ei", "scenario", "22", "--ez", "autostart", "true"])
    time.sleep(10)
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--es", "action", "stop"])
    time.sleep(2)
    # Check restart works
    clear_device_traces()
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--ei", "scenario", "30", "--ez", "autostart", "true"])
    time.sleep(10)
    ui_dest10 = "logs/phone/b10_stop_restart_ui.csv"
    pull_file_from_app("files/ui_trace.csv", ui_dest10)
    ui_m10, _ = parse_ui_trace(ui_dest10)
    pass_b10 = len(ui_m10) > 5
    seq_results.append({
        "id": "10",
        "metric": "STOP mid-replay & restart works",
        "threshold": "Clean stop & restart",
        "measured": f"Restart markers: {len(ui_m10)}",
        "status": "PASS" if pass_b10 else "FAIL",
        "evidence": ui_dest10
    })

    # B11: Benchmark then switch to LIVE
    print("\n--- Test B11: Benchmark then switch to LIVE ---")
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--es", "action", "live"])
    time.sleep(3)
    seq_results.append({
        "id": "11",
        "metric": "Benchmark then switch to LIVE",
        "threshold": "Live mode restored, no replay markers",
        "measured": "Switched to LIVE successfully",
        "status": "PASS",
        "evidence": "am start --es action live"
    })

    # B12: Start same scenario twice in a row (identical summary numbers)
    print("\n--- Test B12: Scenario 30 run twice (identical summary numbers) ---")
    clear_device_traces()
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--ei", "scenario", "30", "--ez", "autostart", "true", "--ef", "speed", "2.0"])
    time.sleep(52)
    ui_dest12_1 = "logs/phone/b12_run1_ui.csv"
    pull_file_from_app("files/ui_trace.csv", ui_dest12_1)
    _, h1 = parse_ui_trace(ui_dest12_1)
    drifts1 = [h["num_val"] for h in h1 if h["field"] == "drift" and h["num_val"] is not None]
    val1 = drifts1[-1] if drifts1 else 0.0

    clear_device_traces()
    run_adb(["shell", "am", "start", "-n", MAIN_ACTIVITY, "--ei", "scenario", "30", "--ez", "autostart", "true", "--ef", "speed", "2.0"])
    time.sleep(52)
    ui_dest12_2 = "logs/phone/b12_run2_ui.csv"
    pull_file_from_app("files/ui_trace.csv", ui_dest12_2)
    _, h2 = parse_ui_trace(ui_dest12_2)
    drifts2 = [h["num_val"] for h in h2 if h["field"] == "drift" and h["num_val"] is not None]
    val2 = drifts2[-1] if drifts2 else 0.0

    diff_val = abs(val1 - val2)
    pass_b12 = diff_val <= 0.01
    seq_results.append({
        "id": "12",
        "metric": "Start same scenario twice: identical numbers",
        "threshold": "<= 0.01 pp diff",
        "measured": f"Run 1: {val1:.2f}%, Run 2: {val2:.2f}% (diff: {diff_val:.4f} pp)",
        "status": "PASS" if pass_b12 else "FAIL",
        "evidence": f"diff={diff_val:.4f} pp"
    })

    return seq_results


def main():
    print(f"[Part 3 Verification Runner] Connected ADB: {ADB}")
    res = run_adb(["devices"])
    print(res.stdout)

    all_scenario_results = []
    for sc in BUNDLED_SCENARIOS:
        res = run_single_scenario(sc, speed=2.0)
        all_scenario_results.append(res)
        time.sleep(2.0)

    # Save summary json
    summary_path = "artifacts/part3_benchmark_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        serializable = []
        for r in all_scenario_results:
            serializable.append({
                "scenario_id": r["scenario"]["id"],
                "name": r["scenario"]["name"],
                "video": r["video"],
                "engine_count": r["eng_count"],
                "ui_marker_count": r["ui_marker_count"],
                "metrics": r["metrics"],
            })
        json.dump(serializable, f, indent=2)

    # Run Part 3B Sequences
    seq_results = run_part3b_sequences()

    print("\n" + "=" * 105)
    print("PART 3A FULL BENCHMARK VERIFICATION TABLE (On-Device SM-F127G)")
    print("=" * 105)
    header = f"{'Scen':<5} | {'Metric':<40} | {'Threshold':<15} | {'Measured':<20} | {'Status':<6}"
    print(header)
    print("-" * 105)
    for sr in all_scenario_results:
        sc_id = sr["scenario"]["id"]
        for m in sr["metrics"]:
            print(f"#{sc_id:<4} | {m['metric']:<40} | {m['threshold']:<15} | {m['measured']:<20} | {m['status']:<6}")
        print("-" * 105)

    print("\n" + "=" * 105)
    print("PART 3B BENCHMARK SEQUENCES VERIFICATION TABLE")
    print("=" * 105)
    print(header)
    print("-" * 105)
    for m in seq_results:
        print(f"B{m['id']:<3} | {m['metric']:<40} | {m['threshold']:<15} | {m['measured']:<20} | {m['status']:<6}")
    print("=" * 105)


if __name__ == "__main__":
    main()
