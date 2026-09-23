"""
T6 - Contiguous sequence sampler for the interval loss.

Leak-free by construction (same guards as MultiScaleCANMoEDataset):
  * only S-M, S-S1, S-S2 (ALLOWED_TRAIN_TRIPS); S-S3a / S-S4 are refused
  * each sequence (including its 6 s feature context) lies fully inside ONE
    partition range from compute_trip_partition (train = Part 1, val = Part 2);
    Part 3 (benchmark) is never touched
  * normalisation = the checkpoint's train-set mean/std (no refit)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from sih.models.can_dataset import ALLOWED_TRAIN_TRIPS

LONG_LEN = 60
SHORT_LEN = 20


class IntervalSequenceSampler:
    def __init__(self, arrays: Dict[str, Dict[str, np.ndarray]], mean: np.ndarray, std: np.ndarray,
                 cal_s: float = 20.0, rate_hz: float = 10.0, rot_aug_deg: float = 15.0):
        """
        arrays[trip_id] = {"feats": (N, C), "v": (N,), "a_lat": (N,), "w_yaw": (N,)}
        already restricted to ONE partition range.
        """
        for tid in arrays:
            if tid not in ALLOWED_TRAIN_TRIPS:
                raise ValueError(f"Trip {tid} is not allowed for training/validation")
        self.arrays = arrays
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1)
        self.cal_n = int(round(cal_s * rate_hz))
        self.rate_hz = rate_hz
        self.rot_aug_rad = float(np.radians(rot_aug_deg))

    @classmethod
    def from_trips(cls, trips: List, partition: str, mean, std, data_dir: str = "data/raw/iovnbd_trips",
                   cache_dir: str = "data/cache", in_channels: int = 12, **kw) -> "IntervalSequenceSampler":
        from sih.features.streaming import load_or_compute_causal_features
        from sih.data.can_sync import load_synchronized_can_speed
        from sih.data.split import compute_trip_partition

        arrays = {}
        for trip in trips:
            tid = trip.trip_id
            if tid not in ALLOWED_TRAIN_TRIPS:
                continue
            feats, f_acc, f_gyr = load_or_compute_causal_features(trip, in_channels=in_channels, cache_dir=cache_dir)
            can = load_synchronized_can_speed(tid, data_dir, strict=True)
            n = min(len(feats), len(can))
            part = compute_trip_partition(tid, n)
            a, b = part.get_range(partition)
            arrays[tid] = {
                "feats": feats[a:b].astype(np.float32), "v": can[a:b].astype(np.float32),
                "a_lat": f_acc[a:b, 1].astype(np.float32), "w_yaw": f_gyr[a:b, 2].astype(np.float32),
            }
        return cls(arrays, mean, std, **kw)

    def n_steps(self, H_s: float) -> int:
        return self.cal_n + int(round(H_s * self.rate_hz))

    def valid_starts(self, tid: str, L: int) -> np.ndarray:
        N = len(self.arrays[tid]["v"])
        lo, hi = LONG_LEN - 1, N - L          # first step needs 59 samples of context in-range
        return np.arange(lo, hi) if hi > lo else np.zeros(0, dtype=int)

    def sample(self, B: int, H_s: float, rng: np.random.Generator, augment: bool,
               min_blackout_dist_m: float = 20.0) -> Optional[Dict[str, torch.Tensor]]:
        L = self.n_steps(H_s)
        tids = [t for t in self.arrays if len(self.valid_starts(t, L)) > 0]
        if not tids:
            return None
        weights = np.array([len(self.valid_starts(t, L)) for t in tids], dtype=np.float64)
        weights /= weights.sum()
        seqs = []
        tries = 0
        while len(seqs) < B and tries < 50 * B:
            tries += 1
            tid = tids[rng.choice(len(tids), p=weights)]
            starts = self.valid_starts(tid, L)
            s = int(starts[rng.integers(len(starts))])
            v = self.arrays[tid]["v"][s:s + L]
            if float(np.sum(v[self.cal_n:])) / self.rate_hz < min_blackout_dist_m:
                continue                                   # benchmark also drops < 20 m blackouts
            seqs.append((tid, s))
        if not seqs:
            return None
        return self._build(seqs, L, rng, augment)

    def fixed_set(self, H_s: float, n_per_trip: int, seed: int = 0) -> List[Tuple[str, int]]:
        """Deterministic, evenly spaced validation sequences."""
        L = self.n_steps(H_s)
        out = []
        for tid in sorted(self.arrays):
            st = self.valid_starts(tid, L)
            if len(st) == 0:
                continue
            for s in np.linspace(st[0], st[-1], n_per_trip).astype(int):
                v = self.arrays[tid]["v"][s:s + L]
                if float(np.sum(v[self.cal_n:])) / self.rate_hz >= 20.0:
                    out.append((tid, int(s)))
        return out

    def build(self, seqs: List[Tuple[str, int]], H_s: float) -> Dict[str, torch.Tensor]:
        return self._build(seqs, self.n_steps(H_s), np.random.default_rng(0), augment=False)

    def _build(self, seqs, L, rng, augment) -> Dict[str, torch.Tensor]:
        xs, vs, al, wy = [], [], [], []
        for tid, s in seqs:
            arr = self.arrays[tid]
            f = arr["feats"][s - (LONG_LEN - 1): s + L].copy()        # (L + 59, C)
            if augment:
                ang = rng.uniform(-self.rot_aug_rad, self.rot_aug_rad, size=3)
                rot = R.from_euler("xyz", ang).as_matrix().astype(np.float32)
                f[:, :3] = f[:, :3] @ rot.T
                f[:, 3:6] = f[:, 3:6] @ rot.T
                f[:, :3] += rng.normal(0, 0.02, size=f[:, :3].shape).astype(np.float32)
                f[:, 3:6] += rng.normal(0, 0.005, size=f[:, 3:6].shape).astype(np.float32)
                f[:, 6:7] = np.linalg.norm(f[:, :3], axis=1, keepdims=True)
                f[:, 7:8] = np.linalg.norm(f[:, 3:6], axis=1, keepdims=True)
            xs.append((f - self.mean) / self.std)
            vs.append(arr["v"][s:s + L])
            al.append(arr["a_lat"][s:s + L])
            wy.append(arr["w_yaw"][s:s + L])
        x = torch.from_numpy(np.stack(xs).astype(np.float32))          # (B, L+59, C)
        v = torch.from_numpy(np.stack(vs).astype(np.float32))           # (B, L)
        a_lat = torch.from_numpy(np.stack(al).astype(np.float32))
        w_yaw = torch.from_numpy(np.stack(wy).astype(np.float32))
        label = torch.ones_like(v, dtype=torch.long)
        label[(w_yaw.abs() > 0.08) | (a_lat.abs() > 1.2)] = 2
        label[v < 0.3] = 0
        return {"x": x, "v": v, "a_lat": a_lat, "w_yaw": w_yaw, "label": label, "L": L}


def windows_from_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: (B, L+59, C) -> x_short (B*L, C, 20), x_long (B*L, C, 60), same layout as inference.py."""
    B, T, C = x.shape
    w = x.unfold(1, LONG_LEN, 1)                  # (B, L, C, 60)
    x_long = w.reshape(-1, C, LONG_LEN).contiguous()
    x_short = x_long[:, :, -SHORT_LEN:].contiguous()
    return x_short, x_long
