"""
PyTorch Dataset and Sliding Window Generator for IMU-to-Velocity Modeling.

Features:
- Fast vectorized sliding window generation with zero memory fragmentation.
- Online 3D SO(3) random rotation data augmentation for mounting-angle invariance.
- Realistic sensor noise jitter.
- Enforces strict trip-level partitioning with zero row-level data leakage.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from sih.data.loader import TripSequence


from sih.data.split import compute_trip_partition


class IMUVelocityDataset(Dataset):
    """
    Sliding-window dataset transforming continuous TripSequences into (Window, GroundTruthSpeed) samples.
    
    Inputs:
      - IMU Window: Tensor of shape (in_channels, window_size) -> [ax, ay, az, gx, gy, gz, ||a||, ||w||]
    Targets:
      - Forward Speed: float (m/s)
    """
    def __init__(
        self,
        trips: List[TripSequence],
        window_size: int = 100,        # 100 samples = 10 seconds at 10Hz
        step_size: int = 2,            # Stride between windows (0.2s)
        in_channels: int = 8,          # 6 IMU + 2 magnitudes
        is_train: bool = True,
        use_calibrated: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        partition: str = "all",
    ) -> None:
        self.window_size = window_size
        self.step_size = step_size
        self.in_channels = in_channels
        self.is_train = is_train
        self.partition = partition
        
        all_windows = []
        all_targets = []

        for trip in trips:
            if len(trip.imu_samples) < window_size or len(trip.gnss_samples) == 0:
                continue

            imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.int64)
            if use_calibrated:
                from sih.calibration.mount import MountCalibrator
                calib = MountCalibrator(window_size=100)
                for g in trip.gnss_samples:
                    calib.observe_gnss(g)
                calib_samples = [calib.update(s) for s in trip.imu_samples]
                accels = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
                gyros = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
            else:
                accels = np.array([s.accel for s in trip.imu_samples], dtype=np.float32)
                gyros = np.array([s.gyro for s in trip.imu_samples], dtype=np.float32)

            # GNSS speed profile
            gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
            gnss_speeds = np.array([
                g.speed_mps if g.speed_mps is not None else 0.0
                for g in trip.gnss_samples
            ], dtype=np.float32)

            interp_speeds = np.interp(imu_ts, gnss_ts, gnss_speeds).astype(np.float32)
            interp_speeds[interp_speeds < 0.2] = 0.0

            raw_6 = np.hstack([accels, gyros])  # (N, 6)
            if partition != "all":
                part = compute_trip_partition(trip.trip_id, len(raw_6))
                p_start, p_end = part.get_range(partition)
                raw_6 = raw_6[p_start:p_end]
                interp_speeds = interp_speeds[p_start:p_end]

            n_samples = len(raw_6)
            num_windows = (n_samples - window_size) // step_size + 1
            if num_windows <= 0:
                continue

            # Fast numpy stride sliding window view
            sub_windows = np.lib.stride_tricks.sliding_window_view(
                raw_6, window_shape=(window_size, 6)
            )[::step_size, 0, :, :]
            sub_windows = np.transpose(sub_windows, (0, 2, 1))  # (num_windows, 6, window_size)

            sub_targets = interp_speeds[window_size - 1 :: step_size][:num_windows]

            all_windows.append(sub_windows)
            all_targets.append(sub_targets)

        if len(all_windows) > 0:
            self.windows = np.concatenate(all_windows, axis=0)  # (N, 6, window_size)
            self.targets = np.concatenate(all_targets, axis=0)  # (N,)
        else:
            self.windows = np.zeros((0, 6, window_size), dtype=np.float32)
            self.targets = np.zeros((0,), dtype=np.float32)

        # Compute or apply normalization stats across channels
        if mean is None or std is None:
            if len(self.windows) > 0:
                norms_a = np.linalg.norm(self.windows[:, :3, :], axis=1, keepdims=True)
                norms_w = np.linalg.norm(self.windows[:, 3:6, :], axis=1, keepdims=True)
                all_8 = np.concatenate([self.windows, norms_a, norms_w], axis=1)  # (N, 8, window_size)
                self.mean = np.mean(all_8, axis=(0, 2), keepdims=True)[0]  # (8, 1)
                self.std = np.std(all_8, axis=(0, 2), keepdims=True)[0]    # (8, 1)
                self.std[self.std < 1e-5] = 1.0  # Guard division by zero
            else:
                self.mean = np.zeros((self.in_channels, 1), dtype=np.float32)
                self.std = np.ones((self.in_channels, 1), dtype=np.float32)
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.windows[idx].copy()  # (6, window_size)

        if self.is_train:
            # 1. Random 3D SO(3) Orientation Perturbation
            angles = np.random.uniform(-0.25, 0.25, size=3).astype(np.float32)  # +/- 15 degrees
            rot_mat = R.from_euler("xyz", angles).as_matrix().astype(np.float32)
            w[:3, :] = rot_mat @ w[:3, :]
            w[3:6, :] = rot_mat @ w[3:6, :]

            # 2. Add realistic IMU sensor noise jitter
            w[:3, :] += np.random.normal(0, 0.02, size=w[:3, :].shape).astype(np.float32)
            w[3:6, :] += np.random.normal(0, 0.005, size=w[3:6, :].shape).astype(np.float32)

        if self.in_channels == 8:
            norm_a = np.linalg.norm(w[:3, :], axis=0, keepdims=True)
            norm_w = np.linalg.norm(w[3:6, :], axis=0, keepdims=True)
            w_all = np.concatenate([w, norm_a, norm_w], axis=0)  # (8, window_size)
        else:
            w_all = w

        w_norm = (w_all - self.mean) / self.std
        x = torch.from_numpy(w_norm).float()
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


class MultiScaleMoEDataset(Dataset):
    """
    Multi-Scale Dual-Window Dataset for Bayesian Mixture-of-Experts (ResNet-1D + TCN-Attention).

    Extracts:
      - x_short: (in_channels, short_len) micro-window (e.g. 20 samples = 2s) for ResNet-1D
      - x_long:  (in_channels, long_len) macro-window (e.g. 60 samples = 6s) for TCN-Attention
      - target_v: Scalar ground truth forward speed (m/s)
      - a_lat: Lateral acceleration for centripetal consistency
      - w_yaw: Yaw rate for centripetal consistency
      - motion_label: 0 (Stationary), 1 (Cruising), 2 (Cornering)
    """

    def __init__(
        self,
        trips: List[TripSequence],
        short_len: int = 20,
        long_len: int = 60,
        stride: int = 2,
        in_channels: int = 12,
        use_calibrated: bool = True,
        is_train: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        rot_aug_deg: float = 15.0,
        partition: str = "all",
    ) -> None:
        self.short_len = short_len
        self.long_len = long_len
        self.stride = stride
        self.in_channels = in_channels
        self.is_train = is_train
        self.rot_aug_rad = float(np.radians(rot_aug_deg))
        self.partition = partition

        self.samples = []
        all_features = []
        import os

        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)

        for trip in trips:
            if len(trip.imu_samples) < long_len or len(trip.gnss_samples) == 0:
                continue

            cache_path = os.path.join(cache_dir, f"{trip.trip_id}_features_{in_channels}ch.npz")
            if os.path.exists(cache_path):
                cached = np.load(cache_path)
                feats = cached["feats"]
                interp_speeds = cached["interp_speeds"]
                f_accel = cached["f_accel"]
                f_gyro = cached["f_gyro"]
            else:
                from sih.data.spectral import DualBandSpectralExtractor
                from sih.data.vibration import VibrationConditioner
                cond = VibrationConditioner(sampling_rate=10.0)
                spec = DualBandSpectralExtractor(sampling_rate=10.0)

                imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples], dtype=np.int64)

                if use_calibrated:
                    from sih.calibration.mount import MountCalibrator
                    calib = MountCalibrator(window_size=100)
                    for g in trip.gnss_samples:
                        calib.observe_gnss(g)
                    calib_samples = [calib.update(s) for s in trip.imu_samples]
                    accels = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
                    gyros = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
                else:
                    accels = np.array([s.accel for s in trip.imu_samples], dtype=np.float32)
                    gyros = np.array([s.gyro for s in trip.imu_samples], dtype=np.float32)

                f_accel, f_gyro = cond.filter_imu_sequence(accels, gyros)

                gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples], dtype=np.int64)
                gnss_speeds = np.array([
                    g.speed_mps if g.speed_mps is not None else 0.0
                    for g in trip.gnss_samples
                ], dtype=np.float32)
                interp_speeds = np.interp(imu_ts, gnss_ts, gnss_speeds).astype(np.float32)
                interp_speeds[interp_speeds < 0.2] = 0.0

                raw_6 = np.hstack([f_accel, f_gyro])
                norm_a = np.linalg.norm(f_accel, axis=1, keepdims=True)
                norm_w = np.linalg.norm(f_gyro, axis=1, keepdims=True)

                if in_channels == 12:
                    spec_feats = spec.extract_sequence_features(raw_6, window_len=long_len, stride=5)
                    feats = np.hstack([raw_6, norm_a, norm_w, spec_feats])
                else:
                    feats = np.hstack([raw_6, norm_a, norm_w])

                np.savez_compressed(cache_path, feats=feats, interp_speeds=interp_speeds, f_accel=f_accel, f_gyro=f_gyro)

            if partition != "all":
                part = compute_trip_partition(trip.trip_id, len(feats))
                p_start, p_end = part.get_range(partition)
                feats = feats[p_start:p_end]
                interp_speeds = interp_speeds[p_start:p_end]
                f_accel = f_accel[p_start:p_end]
                f_gyro = f_gyro[p_start:p_end]

            all_features.append(feats)
            N = len(feats)

            for i in range(0, N - long_len + 1, stride):
                w_long = feats[i : i + long_len]
                w_short = feats[i + long_len - short_len : i + long_len]
                target_v = float(interp_speeds[i + long_len - 1])
                a_lat = float(f_accel[i + long_len - 1, 1])
                w_yaw = float(f_gyro[i + long_len - 1, 2])

                # Motion regime classification
                if target_v < 0.3:
                    motion_label = 0
                elif abs(w_yaw) > 0.08 or abs(a_lat) > 1.2:
                    motion_label = 2
                else:
                    motion_label = 1

                self.samples.append((w_short, w_long, target_v, a_lat, w_yaw, motion_label))

        if len(all_features) > 0:
            concat_all = np.vstack(all_features)
            if mean is None or std is None:
                self.mean = np.mean(concat_all, axis=0, keepdims=True)  # (1, C)
                self.std = np.std(concat_all, axis=0, keepdims=True) + 1e-6
            else:
                self.mean = mean
                self.std = std
        else:
            self.mean = np.zeros((1, in_channels), dtype=np.float32)
            self.std = np.ones((1, in_channels), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        w_short, w_long, target_v, a_lat, w_yaw, motion_label = self.samples[idx]

        w_s = w_short.copy()  # (short_len, C)
        w_l = w_long.copy()   # (long_len, C)

        if self.is_train:
            # Online 3D SO(3) random rotation perturbation (+/- 15 deg)
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

        x_short = torch.tensor(feats_s_norm.T, dtype=torch.float32)  # (C, short_len)
        x_long = torch.tensor(feats_l_norm.T, dtype=torch.float32)    # (C, long_len)
        target_v_tensor = torch.tensor([target_v], dtype=torch.float32)
        alat_tensor = torch.tensor([a_lat], dtype=torch.float32)
        wyaw_tensor = torch.tensor([w_yaw], dtype=torch.float32)
        motion_label_tensor = torch.tensor(motion_label, dtype=torch.long)

        return x_short, x_long, target_v_tensor, alat_tensor, wyaw_tensor, motion_label_tensor

