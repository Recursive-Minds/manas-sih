import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    sdk_adb = os.path.expandvars(r"$LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe")
    adb = sdk_adb if os.path.exists(sdk_adb) else "adb"
    
    print(f"Using adb: {adb}")
    
    # 1. Clear old screenrecord if any
    subprocess.run([adb, "shell", "rm", "-f", "/sdcard/benchmark_demo.mp4"])
    
    # 2. Start screenrecord
    print("Starting screenrecord...")
    rec_proc = subprocess.Popen([adb, "shell", "screenrecord", "--time-limit", "180", "/sdcard/benchmark_demo.mp4"])
    time.sleep(2)
    
    # 3. Tap RUN BENCHMARK button (x=430, y=305)
    print("Tapping RUN BENCHMARK (430, 305)...")
    subprocess.run([adb, "shell", "input", "tap", "430", "305"])
    
    # 4. Wait for replay to progress (e.g. 40 seconds)
    print("Recording scenario replay for 40 seconds...")
    for s in range(40):
        time.sleep(1)
        if s % 10 == 0:
            print(f"  Recorded {s}/40 seconds...")
            
    # 5. Stop screenrecord
    print("Stopping screenrecord...")
    # Killing screenrecord process on Android via pkill
    subprocess.run([adb, "shell", "pkill", "-2", "screenrecord"])
    time.sleep(3)
    rec_proc.terminate()
    rec_proc.wait(timeout=5)
    
    # 6. Pull video
    out_video = os.path.abspath("artifacts/phone_videos/benchmark_demo.mp4")
    print(f"Pulling video to {out_video}...")
    subprocess.run([adb, "pull", "/sdcard/benchmark_demo.mp4", out_video])
    if os.path.exists(out_video):
        print(f"Video saved successfully: {os.path.getsize(out_video)} bytes")
    else:
        print("Warning: video pull failed")
        
    # 7. Pull traces
    from scripts.check_ui_trace import pull_traces_from_device, parse_engine_trace, parse_ui_trace, compute_metrics
    eng_p, ui_p = pull_traces_from_device("logs/phone")
    eng_rows = parse_engine_trace(eng_p)
    ui_markers, ui_hud = parse_ui_trace(ui_p)
    print(f"\nTrace summary:")
    print(f"  Engine trace rows: {len(eng_rows)}")
    print(f"  UI markers: {len(ui_markers)}")
    print(f"  UI hud events: {len(ui_hud)}")
    
    results = compute_metrics(eng_rows, ui_markers, ui_hud, trace_path=ui_p, video_path=out_video)
    print("\n" + "=" * 90)
    for r in results:
        print(f"{r['id']:<3} | {r['metric']:<38} | {r['threshold']:<15} | {r['measured']:<18} | {r['status']:<6}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
