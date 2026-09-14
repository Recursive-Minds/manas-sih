"""
CAN-Supervised Multi-Scale Dataset Generator for Dual-Expert Bayesian MoE.

Pairs smartphone IMU features with time-synchronized 10 Hz vehicle CAN wheel speed targets.
Strictly enforces:
1. Zero Data Leakage: Partitions strictly into 60% Train, 15s Embargo, 20% Val. Part 3 (Test) is NEVER touched.
2. Unseen Test Protection: Only trips S-M, S-S1, and S-S2 are used. S-S3a and S-S4 are strictly barred.
3. Sub-Second Cross-Correlation Synchronization: Time offsets (+0.10s S-S1, +8.60s S-S2, +1.70s S-M) applied.
4. SO(3) 3D Rotational Augmentation: Invariant to mount orientation (Rule 9).
"""

from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from sih.data.split import compute_trip_partition


CAN_TIME_OFFSETS = {
    "S-S1": 1,    # +0.10 s (1 tick at 10 Hz)
    "S-S2": 86,   # +8.60 s (86 ticks at 10 Hz)
    "S-M": 17,    # +1.70 s (17 ticks at 10 Hz)
}

ALLOWED_TRAIN_TRIPS = {"S-M", "S-S1", "S-S2"}


class MultiScaleCANMoEDataset(Dataset):
    """
    Dual-Scale Dataset providing short (2.0s) and long (6.0s) IMU windows
    supervised by time-synchronized 10 Hz vehicle CAN wheel speeds.
    """
    def __init__(
        self,
        trips: List[Any],
        short_len: int = 20,
        long_len: int = 60,
        stride: int = 2,
        in_channels: int = 12,
        use_calibrated: bool = True,
        is_train: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        rot_aug_deg: float = 15.0,
        partition: str = "train",
        data_dir: str = "data/raw/iovnbd_trips",
        cache_dir: str = "data/cache",
    ) -> None:
        super().__init__()
        self.short_len = short_len
        self.long_len = long_len
        self.stride = stride
        self.in_channels = in_channels
        self.is_train = is_train
        self.rot_aug_rad = float(np.radians(rot_aug_deg))
        self.partition = partition

        self.samples = []
        all_features = []

        for trip in trips:
            tid = trip.trip_id
            # STRICT ZERO-LEAKAGE GUARD: Only S-M, S-S1, S-S2 are permitted in training/validation
            if tid not in ALLOWED_TRAIN_TRIPS:
                continue

            # 1. Load cached phone IMU features
            cache_path = os.path.join(cache_dir, f"{tid}_features_{in_channels}ch.npz")
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Feature cache missing for {tid}: {cache_path}")

            cached = np.load(cache_path)
            feats = cached["feats"]
            f_accel = cached["f_accel"]
            f_gyro = cached["f_gyro"]

            # 2. Load and synchronize 10 Hz vehicle CAN wheel speed
            v_path = os.path.join(data_dir, f"V-{tid[2:]}.csv")
            if not os.path.exists(v_path):
                raise FileNotFoundError(f"CAN reference file missing for {tid}: {v_path}")

            v_df = pd.read_csv(v_path, encoding="latin-1")
            v_cols = {c.strip(): c for c in v_df.columns}
            v_col = v_cols.get("Velocity (km/hr)", v_cols.get("Indicated Vehicle Speed (km/hr)"))
            raw_v_mps = (v_df[v_col].fillna(0).to_numpy() / 3.6).astype(np.float32)

            # Apply cross-correlation temporal offset
            lag = CAN_TIME_OFFSETS.get(tid, 0)
            can_speeds = np.zeros_like(raw_v_mps)
            if lag > 0:
                can_speeds[:-lag] = raw_v_mps[lag:]
                can_speeds[-lag:] = raw_v_mps[-1]
            elif lag < 0:
                can_speeds[-lag:] = raw_v_mps[:lag]
                can_speeds[:-lag] = raw_v_mps[0]
            else:
                can_speeds = raw_v_mps

            can_speeds[can_speeds < 0.2] = 0.0

            # Ensure lengths match
            min_len = min(len(feats), len(can_speeds))
            feats = feats[:min_len]
            f_accel = f_accel[:min_len]
            f_gyro = f_gyro[:min_len]
            can_speeds = can_speeds[:min_len]

            # 3. Strictly apply sequence-level partition (60% Train, 15s Embargo, 20% Val)
            part = compute_trip_partition(tid, min_len)
            p_start, p_end = part.get_range(partition)
            feats_part = feats[p_start:p_end]
            f_accel_part = f_accel[p_start:p_end]
            f_gyro_part = f_gyro[p_start:p_end]
            can_speeds_part = can_speeds[p_start:p_end]

            all_features.append(feats_part)
            N = len(feats_part)

            # 4. Generate sliding windows with physical ground truth targets
            for i in range(0, N - long_len + 1, stride):
                w_long = feats_part[i : i + long_len]
                w_short = feats_part[i + long_len - short_len : i + long_len]
                target_v = float(can_speeds_part[i + long_len - 1])
                a_lat = float(f_accel_part[i + long_len - 1, 1])
                w_yaw = float(f_gyro_part[i + long_len - 1, 2])

                # Physical motion regime classification
                if target_v < 0.3:
                    motion_label = 0  # Stationary / Stopped
                elif abs(w_yaw) > 0.08 or abs(a_lat) > 1.2:
                    motion_label = 2  # Dynamic Turn / Cornering
                else:
                    motion_label = 1  # Cruising

                self.samples.append((w_short, w_long, target_v, a_lat, w_yaw, motion_label))

        # 5. Normalization parameters computed strictly on training data
        if len(all_features) > 0:
            concat_all = np.vstack(all_features)
            if mean is None or std is None:
                self.mean = np.mean(concat_all, axis=0, keepdims=True).astype(np.float32)
                self.std = (np.std(concat_all, axis=0, keepdims=True) + 1e-6).astype(np.float32)
            else:
                self.mean = mean.astype(np.float32)
                self.std = std.astype(np.float32)
        else:
            self.mean = np.zeros((1, in_channels), dtype=np.float32)
            self.std = np.ones((1, in_channels), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        w_short, w_long, target_v, a_lat, w_yaw, motion_label = self.samples[idx]

        w_s = w_short.copy()
        w_l = w_long.copy()

        if self.is_train:
            # Rule 9: Online 3D SO(3) random rotation perturbation (+/- 15 deg)
            angles = np.random.uniform(-self.rot_aug_rad, self.rot_aug_rad, size=3).astype(np.float32)
            rot = R.from_euler("xyz", angles).as_matrix().astype(np.float32)

            w_s[:, :3] = w_s[:, :3] @ rot.T
            w_s[:, 3:6] = w_s[:, 3:6] @ rot.T
            w_l[:, :3] = w_l[:, :3] @ rot.T
            w_l[:, 3:6] = w_l[:, 3:6] @ rot.T

            # Sensor noise jitter
            w_s[:, :3] += np.random.normal(0, 0.02, size=w_s[:, :3].shape).astype(np.float32)
            w_s[:, 3:6] += np.random.normal(0, 0.005, size=w_s[:, 3:6].shape).astype(np.float32)
            w_l[:, :3] += np.random.normal(0, 0.02, size=w_l[:, :3].shape).astype(np.float32)
            w_l[:, 3:6] += np.random.normal(0, 0.005, size=w_l[:, 3:6].shape).astype(np.float32)

            # Recompute norms
            w_s[:, 6:7] = np.linalg.norm(w_s[:, :3], axis=1, keepdims=True)
            w_s[:, 7:8] = np.linalg.norm(w_s[:, 3:6], axis=1, keepdims=True)
            w_l[:, 6:7] = np.linalg.norm(w_l[:, :3], axis=1, keepdims=True)
            w_l[:, 7:8] = np.linalg.norm(w_l[:, 3:6], axis=1, keepdims=True)

        feats_s_norm = (w_s - self.mean) / self.std
        feats_l_norm = (w_l - self.mean) / self.std

        x_short = torch.tensor(feats_s_norm.T, dtype=torch.float32)
        x_long = torch.tensor(feats_l_norm.T, dtype=torch.float32)
        target_v_tensor = torch.tensor([target_v], dtype=torch.float32)
        alat_tensor = torch.tensor([a_lat], dtype=torch.float32)
        wyaw_tensor = torch.tensor([w_yaw], dtype=torch.float32)
        motion_label_tensor = torch.tensor(motion_label, dtype=torch.long)

        return x_short, x_long, target_v_tensor, alat_tensor, wyaw_tensor, motion_label_tensor
