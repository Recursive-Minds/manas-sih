import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sdk_adb = os.path.expandvars(r"$LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe")
adb = sdk_adb if os.path.exists(sdk_adb) else "adb"

def run_scenario(scenario_id: int, duration_s: int = 35):
    print(f"\n=======================================================")
    print(f"Starting test for Scenario #{scenario_id}...")
    print(f"=======================================================")
    
    # 1. Clear old screenrecord
    subprocess.run([adb, "shell", "rm", "-f", f"/sdcard/benchmark_sc{scenario_id}.mp4"])
    
    # 2. Launch / reset app with scenario intent
    print(f"Launching app with scenario {scenario_id} (autostart=false)...")
    subprocess.run([adb, "shell", "am", "start", "-n", "com.recursiveminds.idr.ondevice/com.recursiveminds.idr.ui.MainActivity", "--ei", "scenario", str(scenario_id), "--ez", "autostart", "false"])
    time.sleep(3)
    
    # 3. Start screenrecord
    print("Starting screen recording on device...")
    rec_proc = subprocess.Popen([adb, "shell", "screenrecord", "--time-limit", "180", f"/sdcard/benchmark_sc{scenario_id}.mp4"])
    time.sleep(2)
    
    # 4. Trigger autostart via intent
    print(f"Starting scenario {scenario_id} replay @ 2.0x...")
    subprocess.run([adb, "shell", "am", "start", "-n", "com.recursiveminds.idr.ondevice/com.recursiveminds.idr.ui.MainActivity", "--ei", "scenario", str(scenario_id), "--ez", "autostart", "true", "--ef", "speed", "2.0"])
    
    # 5. Wait for scenario replay to finish
    print(f"Replaying scenario for {duration_s} seconds...")
    for s in range(duration_s):
        time.sleep(1)
        if (s + 1) % 10 == 0:
            print(f"  Progress: {s + 1}/{duration_s}s...")
            
    # 6. Stop screenrecord
    print("Stopping screenrecord...")
    subprocess.run([adb, "shell", "pkill", "-2", "screenrecord"])
    time.sleep(3)
    rec_proc.terminate()
    try:
        rec_proc.wait(timeout=5)
    except Exception:
        pass
        
    # 7. Pull video
    os.makedirs("artifacts/phone_videos", exist_ok=True)
    out_video = os.path.abspath(f"artifacts/phone_videos/benchmark_sc{scenario_id}.mp4")
    print(f"Pulling video to {out_video}...")
    subprocess.run([adb, "pull", f"/sdcard/benchmark_sc{scenario_id}.mp4", out_video])
    if os.path.exists(out_video):
        print(f"Video saved: {os.path.getsize(out_video):,} bytes")
        
    # 8. Pull and verify traces
    from scripts.check_ui_trace import pull_traces_from_device, parse_engine_trace, parse_ui_trace, compute_metrics
    eng_p, ui_p = pull_traces_from_device("logs/phone")
    eng_rows = parse_engine_trace(eng_p)
    ui_markers, ui_hud = parse_ui_trace(ui_p)
    
    results = compute_metrics(eng_rows, ui_markers, ui_hud, trace_path=ui_p, video_path=out_video)
    print("\n" + "=" * 90)
    print(f"RESULTS FOR SCENARIO #{scenario_id}:")
    print("=" * 90)
    for r in results:
        print(f"{r['id']:<3} | {r['metric']:<42} | {r['threshold']:<20} | {r['measured']:<20} | {r['status']:<6}")
    print("=" * 90 + "\n")
    return results

if __name__ == "__main__":
    sc_id = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    dur = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    run_scenario(sc_id, dur)
