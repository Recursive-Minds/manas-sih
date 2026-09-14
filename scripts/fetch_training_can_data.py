"""
Fetch training CAN reference files (V-M.csv, V-S1.csv, V-S2.csv) from GitHub LFS.
Strictly restricted to the 3 training trips (S-M, S-S1, S-S2).
Zero access to S-S3a, S-S4, or any test data.
"""

import os
import json
import urllib.request
from typing import Dict, Any, List

LFS_BATCH_URL = "https://github.com/onyekpeu/IO-VNBD.git/info/lfs/objects/batch"
BASE_RAW_URL = "https://raw.githubusercontent.com/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Uncategorised%20IOVNB%20Dataset"

TRAIN_V_FILES = ["V-S1.csv", "V-S2.csv", "V-M.csv"]
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

            out_path = os.path.join(target_dir, item["filename"])
            if os.path.exists(out_path) and os.path.getsize(out_path) == item["size"]:
                print(f"Already downloaded: {item['filename']} ({item['size'] / (1024*1024):.2f} MB)")
                continue

            print(f"Downloading {item['filename']} ({item['size'] / (1024*1024):.2f} MB)...")
            urllib.request.urlretrieve(download_href, out_path)
            print(f"Successfully saved: {out_path} ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    target_dir = os.path.abspath(TARGET_DIR)
    print(f"Fetching LFS metadata for training CAN files in {target_dir}...")
    to_fetch = []
    for f in TRAIN_V_FILES:
        out_path = os.path.join(target_dir, f)
        info = fetch_lfs_pointer_info(f)
        if os.path.exists(out_path) and os.path.getsize(out_path) == info["size"]:
            print(f"  - {f} already exists and size matches ({info['size']:,} bytes)")
        else:
            print(f"  - {f}: oid={info['oid'][:12]}..., size={info['size']:,} bytes")
            to_fetch.append(info)

    if to_fetch:
        print(f"\nDownloading {len(to_fetch)} CAN files...")
        download_lfs_objects(to_fetch, target_dir)
    else:
        print("\nAll training CAN files already present!")
