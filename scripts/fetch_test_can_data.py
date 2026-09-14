"""
Fetch testing CAN reference files (V-S3a.csv, V-S4.csv) from GitHub LFS for ground-truth evaluation and visualization.
"""

import os
import json
import urllib.request
from typing import Dict, Any, List

LFS_BATCH_URL = "https://github.com/onyekpeu/IO-VNBD.git/info/lfs/objects/batch"
BASE_RAW_URL = "https://raw.githubusercontent.com/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Uncategorised%20IOVNB%20Dataset"

TEST_V_FILES = ["V-S3a.csv", "V-S4.csv"]
TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "iovnbd_trips")


def fetch_lfs_pointer_info(v_filename: str) -> Dict[str, Any]:
    url = f"{BASE_RAW_URL}/V-Dataset/{v_filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
        lines = content.strip().split("\n")
        oid = None
        size = None
        for line in lines:
            if line.startswith("oid sha256:"):
                oid = line.split("oid sha256:")[1].strip()
            elif line.startswith("size "):
                size = int(line.split("size ")[1].strip())
        return {"filename": v_filename, "oid": oid, "size": size}


def download_lfs_objects(objects_to_fetch: List[Dict[str, Any]], target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": item["oid"], "size": item["size"]} for item in objects_to_fetch]
    }

    req = urllib.request.Request(
        LFS_BATCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        lfs_objects = data.get("objects", [])

        for item, lfs_obj in zip(objects_to_fetch, lfs_objects):
            download_href = lfs_obj.get("actions", {}).get("download", {}).get("href")
            if not download_href:
                print(f"Failed to get download URL for {item['filename']}")
                continue

            target_path = os.path.join(target_dir, item["filename"])
            if os.path.exists(target_path) and os.path.getsize(target_path) == item["size"]:
                print(f"Already downloaded: {item['filename']} ({item['size']} bytes)")
                continue

            print(f"Downloading {item['filename']} ({item['size']:,} bytes)...")
            dl_req = urllib.request.Request(download_href, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(dl_req) as dl_resp, open(target_path, "wb") as f_out:
                bytes_done = 0
                while True:
                    chunk = dl_resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    bytes_done += len(chunk)
                    print(f"  {bytes_done:,} / {item['size']:,} bytes ({bytes_done/item['size']*100:.1f}%)", end="\r")
            print(f"\nSaved to: {target_path}")


if __name__ == "__main__":
    print("Resolving Git LFS pointers for test sequences V-S3a.csv and V-S4.csv...")
    info_list = [fetch_lfs_pointer_info(f) for f in TEST_V_FILES]
    print(info_list)
    download_lfs_objects(info_list, TARGET_DIR)
    print("Test CAN files ready for ground-truth visualization!")
