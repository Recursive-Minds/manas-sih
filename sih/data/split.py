"""
Standardized 3-Way Dataset Split Manager for Smartphone Dead-Reckoning (SIH).

Enforces strict zero-data-leakage temporal partitions across trips:
  1. Part 1: Train (60%)       -> Multi-trip training across urban, arterial, highway.
  2. Embargo Buffer 1 (15s)   -> 150-sample discarded boundary purge.
  3. Part 2: Validation (20%)  -> Independent checkpoint selection & early stopping.
  4. Embargo Buffer 2 (15s)   -> 150-sample discarded boundary purge.
  5. Part 3: Benchmark (20%)   -> Strictly held-out test ground truth for 35 benchmark scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass(slots=True, frozen=True)
class TripPartition:
    trip_id: str
    total_samples: int
    train_range: Tuple[int, int]
    embargo_1_range: Tuple[int, int]
    val_range: Tuple[int, int]
    embargo_2_range: Tuple[int, int]
    bench_range: Tuple[int, int]

    def get_range(self, partition_name: str) -> Tuple[int, int]:
        p = partition_name.lower()
        if p in ("train", "training"):
            return self.train_range
        elif p in ("val", "validation", "eval"):
            return self.val_range
        elif p in ("bench", "benchmark", "test"):
            return self.bench_range
        elif p in ("all", "full"):
            return (0, self.total_samples)
        else:
            raise ValueError(f"Unknown partition '{partition_name}'. Expected 'train', 'val', 'bench', or 'all'.")


def compute_trip_partition(
    trip_id: str,
    total_samples: int,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    embargo_s: float = 15.0,
    sampling_rate_hz: float = 10.0,
) -> TripPartition:
    """Computes exact non-overlapping partition boundaries with temporal embargo buffers."""
    embargo_ticks = int(embargo_s * sampling_rate_hz)
    
    train_end = int(total_samples * train_ratio)
    embargo_1_end = min(total_samples, train_end + embargo_ticks)
    
    val_target_end = int(total_samples * (train_ratio + val_ratio))
    val_start = embargo_1_end
    val_end = max(val_start, val_target_end)
    
    embargo_2_end = min(total_samples, val_end + embargo_ticks)
    bench_start = embargo_2_end
    bench_end = total_samples
    
    return TripPartition(
        trip_id=trip_id,
        total_samples=total_samples,
        train_range=(0, train_end),
        embargo_1_range=(train_end, embargo_1_end),
        val_range=(val_start, val_end),
        embargo_2_range=(val_end, embargo_2_end),
        bench_range=(bench_start, bench_end),
    )


# Canonical precomputed partition registries for IO-VNBD trips
KNOWN_TRIP_LENGTHS = {
    "S-S1": 51746,
    "S-S2": 93876,
    "S-M": 105974,
}


def get_canonical_partition(trip_id: str) -> TripPartition:
    """Returns canonical 3-way partition for known dataset trips."""
    clean_id = trip_id.replace(".csv", "").strip()
    if clean_id in KNOWN_TRIP_LENGTHS:
        return compute_trip_partition(clean_id, KNOWN_TRIP_LENGTHS[clean_id])
    raise KeyError(f"Trip '{trip_id}' not in KNOWN_TRIP_LENGTHS. Use compute_trip_partition() with explicit length.")


def slice_array_by_partition(
    arr: np.ndarray,
    partition_name: str,
    trip_id: str,
) -> np.ndarray:
    """Slices a numpy array (N, ...) along axis 0 according to trip partition boundaries."""
    clean_id = trip_id.replace(".csv", "").strip()
    part = compute_trip_partition(clean_id, len(arr))
    start_idx, end_idx = part.get_range(partition_name)
    return arr[start_idx:end_idx]


def load_partition(
    trip_name: str,
    partition: str = "part3",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads a specific partition (e.g. 'train', 'val', 'part3'/'bench') for a trip.
    
    Returns calibrated IMU samples, GNSS samples, partition range, and calibrator.
    """
    import os
    from sih.data.loader import GenericDataLoader
    from sih.calibration.mount import MountCalibrator

    clean_id = trip_name.replace(".csv", "").strip()
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cand_dirs = [
            os.path.join(base_dir, "data", "raw", "iovnbd_trips"),
            os.path.join(base_dir, "data"),
        ]
        for d in cand_dirs:
            if os.path.exists(os.path.join(d, f"{clean_id}.csv")):
                data_dir = d
                break
        if data_dir is None:
            data_dir = cand_dirs[0]

    trip_path = os.path.join(data_dir, f"{clean_id}.csv")
    if not os.path.exists(trip_path):
        raise FileNotFoundError(f"Dataset file not found: {trip_path}")



    loader = GenericDataLoader()
    trip = loader.load_file(trip_path)

    part = compute_trip_partition(clean_id, len(trip.imu_samples))
    p_norm = partition.lower()
    if p_norm in ("part1", "train", "training"):
        r = part.train_range
    elif p_norm in ("part2", "val", "validation"):
        r = part.val_range
    elif p_norm in ("part3", "bench", "benchmark", "test"):
        r = part.bench_range
    else:
        r = part.get_range(partition)

    imu_sub = trip.imu_samples[r[0]:r[1]]
    t_start_ns = imu_sub[0].timestamp_ns if len(imu_sub) > 0 else 0
    t_end_ns = imu_sub[-1].timestamp_ns if len(imu_sub) > 0 else 0

    gnss_sub = [g for g in trip.gnss_samples if t_start_ns <= g.timestamp_ns <= t_end_ns]

    calibrator = MountCalibrator(min_samples=30)
    gnss_idx = 0
    n_g = len(trip.gnss_samples)
    calib_samples = []
    for imu in trip.imu_samples:
        while gnss_idx < n_g and trip.gnss_samples[gnss_idx].timestamp_ns <= imu.timestamp_ns:
            calibrator.observe_gnss(trip.gnss_samples[gnss_idx])
            gnss_idx += 1
        calib_samples.append(calibrator.update(imu))

    calib_sub = calib_samples[r[0]:r[1]]

    return {
        "trip_id": clean_id,
        "partition": partition,
        "range": r,
        "total_samples": len(trip.imu_samples),
        "imu_samples": imu_sub,
        "calibrated_samples": calib_sub,
        "gnss_samples": gnss_sub,
        "calibrator": calibrator,
        "full_trip": trip,
        "full_calib_samples": calib_samples,
    }

