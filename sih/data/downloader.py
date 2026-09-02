"""
Utility to download benchmark sequences from IO-VNBD via Git LFS with progress tracking.
"""

from __future__ import annotations
import os
import json
import requests
from typing import Optional, Dict


# Pre-indexed SHA256 OIDs and sizes for standard IO-VNBD benchmark trips (verified via GitHub API)
IO_VNBD_INDEX: Dict[str, Dict[str, Any]] = {
    "S-S1": {
        "oid": "e79a2eea18143b825438f9a11c7d724a97c9e18329776defd4d9291d03d4aaca",
        "size": 9631499,
        "type": "smartphone",
        "driver": "Driver A",
    },
    "S-S2": {
        "oid": "8f48d68ef4ab225c8381af6fb8ec0f36fe6f6048977eeebecc74cfec50d13cb8",
        "size": 17469302,
        "type": "smartphone",
        "driver": "Driver A",
    },
    "S-M": {
        "oid": "26a1ee080da4db98e175e15d6fec631e6f8c285c13817d9d74ecc90ed280d103",
        "size": 19798721,
        "type": "smartphone",
        "driver": "Driver B",
    },
}


def download_iovnbd_trip(trip_key: str, output_dir: str = "data/raw/iovnbd_trips", force: bool = False) -> str:
    """
    Downloads an IO-VNBD trip CSV using the Git LFS batch API.
    """
    if trip_key not in IO_VNBD_INDEX:
        raise ValueError(f"Unknown trip key '{trip_key}'. Available: {list(IO_VNBD_INDEX.keys())}")

    meta = IO_VNBD_INDEX[trip_key]
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{trip_key}.csv")

    if os.path.exists(out_file) and not force:
        if os.path.getsize(out_file) == meta["size"]:
            return out_file

    lfs_url = "https://github.com/onyekpeu/IO-VNBD.git/info/lfs/objects/batch"
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": meta["oid"], "size": meta["size"]}],
    }
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
        "User-Agent": "git-lfs/3.0.0",
    }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(lfs_url, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            res_data = resp.json()

            if "actions" not in res_data["objects"][0] or "download" not in res_data["objects"][0]["actions"]:
                err_msg = res_data["objects"][0].get("error", {}).get("message", "Unknown LFS error")
                raise IOError(f"Git LFS server error for {trip_key}: {err_msg}")

            download_url = res_data["objects"][0]["actions"]["download"]["href"]

            # Stream download with tqdm visual progress bar
            from tqdm import tqdm
            temp_file = out_file + ".tmp"
            total_size = meta["size"]

            with requests.get(download_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(temp_file, "wb") as f, tqdm(
                    desc=f"Downloading {trip_key}.csv",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    dynamic_ncols=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=131072):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            if os.path.exists(out_file):
                os.remove(out_file)
            os.rename(temp_file, out_file)

            actual_size = os.path.getsize(out_file)
            if actual_size != meta["size"]:
                raise IOError(f"Downloaded file size mismatch: expected {meta['size']} bytes, got {actual_size} bytes")

            return out_file

        except Exception as e:
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)
            else:
                raise IOError(f"Failed to download {trip_key} after {max_retries} attempts: {e}") from e

    return out_file
