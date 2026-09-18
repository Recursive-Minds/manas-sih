"""
Vehicle CAN Bus Synchronization and Integrity Management.

Enforces authoritative clock offsets between vehicle CAN wheel speeds and GNSS timebase.
Strictly forbids CAN supervision on compromised datasets (e.g. S-S4 split-clock logging failure).
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "can_sync.json"))


def load_can_sync_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the CAN synchronization configuration registry."""
    path = config_path or _CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"CAN synchronization config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_can_offset_seconds(trip_id: str, config_path: Optional[str] = None) -> float:
    """Returns validated temporal offset in seconds for the given trip."""
    clean_id = trip_id.replace(".csv", "").strip()
    cfg = load_can_sync_config(config_path)
    
    if clean_id in cfg.get("excluded_trips", {}):
        raise ValueError(
            f"Trip {clean_id} is permanently excluded from CAN supervision: "
            f"{cfg['excluded_trips'][clean_id]['reason']}"
        )
    
    offsets = cfg.get("offsets_seconds", {})
    if clean_id not in offsets:
        raise KeyError(f"No CAN clock offset registered for trip '{clean_id}'.")
    return float(offsets[clean_id])


def get_can_offset_ticks(trip_id: str, hz: float = 10.0, config_path: Optional[str] = None) -> int:
    """Returns validated sample lag (ticks) for the given trip at specified frequency."""
    offset_s = get_can_offset_seconds(trip_id, config_path=config_path)
    return int(round(offset_s * hz))


def is_can_supervised_allowed(trip_id: str, config_path: Optional[str] = None) -> bool:
    """Checks whether a trip is permissible for CAN-supervised training or evaluation."""
    clean_id = trip_id.replace(".csv", "").strip()
    cfg = load_can_sync_config(config_path)
    return clean_id not in cfg.get("excluded_trips", {})


def load_synchronized_can_speed(
    trip_id: str,
    data_dir: str,
    config_path: Optional[str] = None,
    strict: bool = False,
) -> Optional[np.ndarray]:
    """
    Loads and synchronizes 10 Hz vehicle CAN wheel speed for a trip.
    
    If the trip is excluded from CAN supervision (such as S-S4), raises ValueError
    if strict=True, or returns None if strict=False, enforcing that CAN ground
    truth cannot be used for corrupted sequences.
    """
    clean_id = trip_id.replace(".csv", "").strip()
    
    if not is_can_supervised_allowed(clean_id, config_path=config_path):
        if strict:
            raise ValueError(
                f"CAN ground truth for {clean_id} is PERMANENTLY FORBIDDEN due to "
                f"split-clock logging failure. Use GPS Doppler ground truth instead."
            )
        return None

    v_path = os.path.join(data_dir, f"V-{clean_id[2:]}.csv")
    if not os.path.exists(v_path):
        if strict:
            raise FileNotFoundError(f"CAN reference file missing for {clean_id}: {v_path}")
        return None

    v_df = pd.read_csv(v_path, encoding="latin-1")
    v_cols = {c.strip(): c for c in v_df.columns}
    v_col = v_cols.get("Velocity (km/hr)", v_cols.get("Indicated Vehicle Speed (km/hr)"))
    if not v_col:
        return None

    raw_v_mps = (v_df[v_col].fillna(0).to_numpy() / 3.6).astype(np.float32)
    lag = get_can_offset_ticks(clean_id, hz=10.0, config_path=config_path)

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
    return can_speeds
