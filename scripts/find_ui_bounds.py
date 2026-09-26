import xml.etree.ElementTree as ET
import subprocess
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

adb = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
subprocess.check_call([adb, "shell", "uiautomator", "dump", "/sdcard/window_dump2.xml"])
out = subprocess.check_output([adb, "exec-out", "cat", "/sdcard/window_dump2.xml"], encoding="utf-8", errors="replace")
root = ET.fromstring(out)
for n in root.iter("node"):
    text = n.get("text", "")
    res_id = n.get("resource-id", "")
    bounds = n.get("bounds", "")
    if text or "btn" in res_id:
        print(f"{res_id:50} | {text:30} | {bounds}")
