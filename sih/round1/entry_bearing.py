"""
R2 - one entry-bearing rule for both paths (benchmark and live server).

The live adapter (EngineAdapterStageB.set_blackout) replaces the last sample of the
synthesized 1 Hz pre-blackout window with the entry fix's Doppler bearing/speed; the
batch benchmark did not. That single difference was the last batch-vs-live mismatch
(a 3 deg initial heading offset on S-S3a #22).

Flag `entry_doppler_bearing` decides it for BOTH paths:
  True  -> both use the entry Doppler bearing
  False -> both use the geometric (displacement) bearing
With every round-1 flag OFF the live adapter keeps its legacy behaviour (override on).
"""

from __future__ import annotations

from typing import Any, List

from sih.core.contracts import GNSSSample


def apply_entry_doppler(pre_gnss_window: List[Any], g_entry: Any) -> List[Any]:
    """Same replacement the live adapter does (server/engine_adapter.py set_blackout)."""
    if g_entry is None or g_entry.bearing_deg is None or not pre_gnss_window:
        return pre_gnss_window
    last = pre_gnss_window[-1]
    out = list(pre_gnss_window)
    out[-1] = GNSSSample(
        timestamp_ns=last.timestamp_ns,
        latitude_deg=last.latitude_deg,
        longitude_deg=last.longitude_deg,
        altitude_m=last.altitude_m,
        speed_mps=g_entry.speed_mps if g_entry.speed_mps is not None else last.speed_mps,
        bearing_deg=g_entry.bearing_deg,
        accuracy_h_m=last.accuracy_h_m,
        is_valid=last.is_valid,
    )
    return out


def live_override_enabled() -> bool:
    from sih.round1.config import get_active_config
    cfg = get_active_config()
    if cfg.is_all_off():
        return True            # legacy live behaviour when round 1 is switched off
    return bool(cfg.entry_doppler_bearing)
