"""
scripts/prepare_ondevice_bundle.py
-----------------------------------
Prepares Python sources and assets for the Chaquopy ondevice Android APK build.
1. Copies pure-Python sih and server modules to android/app/src/ondevice/python/.
2. Copies exported TFLite model and normalization params to android/app/src/ondevice/assets/.
3. Generates 60s recorded smoke test dataset (60 batches x 1.0s) from S-S3a.csv into android/app/src/ondevice/assets/smoke_test_60s.json.
"""

import os
import sys
import shutil
import json
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANDROID_DIR = os.path.join(ROOT_DIR, "android", "app", "src", "ondevice")
PYTHON_TARGET_DIR = os.path.join(ANDROID_DIR, "python")
ASSETS_TARGET_DIR = os.path.join(ANDROID_DIR, "assets")


def ignore_patterns(path, names):
    ignored = set()
    for name in names:
        if name in ("__pycache__", ".pytest_cache", "checkpoints", "tests", "eval_reports", "figures", "results"):
            ignored.add(name)
        elif name.endswith(".pyc") or name.endswith(".pt") or name.endswith(".zip"):
            ignored.add(name)
    return ignored


def sync_python_sources():
    print(f"Syncing Python sources to {PYTHON_TARGET_DIR}...")
    os.makedirs(PYTHON_TARGET_DIR, exist_ok=True)

    # 1. Sync sih/
    sih_src = os.path.join(ROOT_DIR, "sih")
    sih_dst = os.path.join(PYTHON_TARGET_DIR, "sih")
    if os.path.exists(sih_dst):
        shutil.rmtree(sih_dst)
    shutil.copytree(sih_src, sih_dst, ignore=ignore_patterns)

    # 2. Sync server/
    server_src = os.path.join(ROOT_DIR, "server")
    server_dst = os.path.join(PYTHON_TARGET_DIR, "server")
    if os.path.exists(server_dst):
        shutil.rmtree(server_dst)
    shutil.copytree(server_src, server_dst, ignore=ignore_patterns)

    print("Python sources synced successfully.")


def sync_assets():
    print(f"Syncing assets to {ASSETS_TARGET_DIR}...")
    os.makedirs(ASSETS_TARGET_DIR, exist_ok=True)

    # 1. TFLite model
    tflite_src = os.path.join(ROOT_DIR, "models", "exported", "moe_velocity_model.tflite")
    tflite_dst = os.path.join(ASSETS_TARGET_DIR, "moe_velocity_model.tflite")
    shutil.copy2(tflite_src, tflite_dst)
    print(f"  - Copied TFLite model: {os.path.getsize(tflite_dst):,} bytes")

    # 2. Normalization params
    norm_src = os.path.join(ROOT_DIR, "models", "exported", "normalization_params.npz")
    norm_dst = os.path.join(ASSETS_TARGET_DIR, "normalization_params.npz")
    shutil.copy2(norm_src, norm_dst)
    print(f"  - Copied Normalization params: {os.path.getsize(norm_dst):,} bytes")


def generate_smoke_test_dataset():
    print("Generating 60s recorded smoke test dataset from S-S3a.csv...")
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from sih.data.loader import GenericDataLoader

    trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-S3a.csv")
    trip = GenericDataLoader().load_file(trip_path)

    # Slice 60 seconds (600 IMU samples @ 10 Hz) starting around index 2000
    start_imu_idx = 2000
    end_imu_idx = start_imu_idx + 600
    imu_slice = trip.imu_samples[start_imu_idx:end_imu_idx]

    t_start_ns = imu_slice[0].timestamp_ns
    t_end_ns = imu_slice[-1].timestamp_ns

    gnss_slice = [g for g in trip.gnss_samples if t_start_ns <= g.timestamp_ns <= t_end_ns]

    # Group into 60 batches (10 IMU samples per batch, ~1 GNSS fix per batch)
    batches = []
    for b_idx in range(60):
        b_imu = imu_slice[b_idx * 10 : (b_idx + 1) * 10]
        b_t_start = b_imu[0].timestamp_ns
        b_t_end = b_imu[-1].timestamp_ns
        b_gnss = [g for g in gnss_slice if b_t_start <= g.timestamp_ns <= b_t_end]

        batch = {
            "batch_index": b_idx,
            "state": "WARMING_UP" if b_idx < 30 else "BLACKOUT",
            "imu": [
                {
                    "timestamp_ns": int(s.timestamp_ns),
                    "accel": [float(s.accel[0]), float(s.accel[1]), float(s.accel[2])],
                    "gyro": [float(s.gyro[0]), float(s.gyro[1]), float(s.gyro[2])],
                }
                for s in b_imu
            ],
            "gnss": [
                {
                    "timestamp_ns": int(g.timestamp_ns),
                    "latitude_deg": float(g.latitude_deg),
                    "longitude_deg": float(g.longitude_deg),
                    "altitude_m": float(g.altitude_m),
                    "speed_mps": float(g.speed_mps) if g.speed_mps is not None else None,
                    "bearing_deg": float(g.bearing_deg) if g.bearing_deg is not None else None,
                    "accuracy_h_m": float(g.accuracy_h_m),
                    "is_valid": True,
                }
                for g in b_gnss
            ],
        }
        batches.append(batch)

    out_json = os.path.join(ASSETS_TARGET_DIR, "smoke_test_60s.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "reference_lat_deg": float(trip.reference_lat_deg),
            "reference_lon_deg": float(trip.reference_lon_deg),
            "reference_alt_m": float(trip.reference_alt_m),
            "total_batches": len(batches),
            "batches": batches,
        }, f, indent=2)

    print(f"Generated smoke test dataset: {out_json} ({os.path.getsize(out_json):,} bytes, 60 batches)")


if __name__ == "__main__":
    sync_python_sources()
    sync_assets()
    generate_smoke_test_dataset()
    print("Bundle preparation completed successfully.")
