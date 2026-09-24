"""
Round-1 feature flags (T3, T4, T5, T7, T8, T9, T10).

Every flag defaults to OFF. With all flags OFF the engine must run bit-identically
to the pre-round-1 baseline (checked by scripts/round1_eval.py --assert-parity).

Activation order of precedence:
  1. set_active_config(cfg) called by an orchestration script (round1_eval.py)
  2. environment variable SIH_ROUND1_CONFIG=<path to json> ("off" forces the old behaviour)
  3. config/round1/production.json, if present (the promoted production profile)
  4. defaults (all OFF)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class StopDetectorParams:          # T3
    enabled: bool = False
    v_enter_mps: float = 1.2       # mean scaled AI speed (1 s) below this may enter STOPPED
    v_exit_mps: float = 2.5        # sustained AI speed above this exits STOPPED
    t_enter_s: float = 1.0         # conditions must hold this long to enter
    t_exit_s: float = 0.5          # exit evidence must hold this long
    gyro_quiet_rad_s: float = 0.03 # mean |gyro| over 1 s
    accel_std_default: float = 0.35  # vibration std gate (m/s^2) if no idle period in history
    accel_std_mult: float = 1.6    # learned gate = mult * median idle std
    launch_accel_mps2: float = 0.35  # a_x above stop baseline (0.5 s mean) = launch


@dataclass
class GyroScaleParams:             # T4
    enabled: bool = False
    window_s: float = 5.0          # heading-change comparison window
    min_turn_deg: float = 20.0     # only windows with a real turn
    min_speed_mps: float = 4.0     # GNSS bearing is unreliable below this
    min_windows: int = 3
    prior_windows: float = 6.0     # shrinkage toward 1.0 (pseudo-count)
    clip_lo: float = 0.92
    clip_hi: float = 1.08


@dataclass
class SpeedModeParams:             # T5
    mode: str = "ai"               # "ai" | "hold_entry" | "entry_offset_decay"
    decay_tau_s: float = 20.0      # entry_offset_decay time constant
    entry_ai_window: int = 20      # samples (2 s) used for the AI speed at entry


@dataclass
class OnlineSpeedCalibParams:      # T7
    enabled: bool = False
    window_s: float = 10.0         # GNSS-distance vs AI-distance comparison window
    min_window_dist_m: float = 20.0
    band_edges_mps: tuple = (0.0, 5.0, 10.0, 15.0, 22.0, 60.0)
    prior_s: float = 30.0          # shrinkage pseudo-seconds toward factor 1.0
    clip_lo: float = 0.85
    clip_hi: float = 1.15


@dataclass
class JunctionAnchorParams:        # T8
    enabled: bool = False
    turn_on_rad_s: float = 0.12    # |yaw rate| that opens a turn
    turn_off_rad_s: float = 0.05   # |yaw rate| that (after t_quiet_s) closes it
    t_quiet_s: float = 1.0
    min_turn_deg: float = 50.0
    max_turn_deg: float = 140.0
    max_turn_duration_s: float = 15.0
    max_speed_mps: float = 16.0    # junction turns happen slowly
    line_tol_deg: float = 25.0     # road line vs pre/post-turn heading
    base_radius_m: float = 25.0    # search radius = base + frac * distance since entry
    radius_frac: float = 0.15
    max_radius_m: float = 80.0
    ambiguity_ratio: float = 1.8   # 2nd-best corner must be this much farther
    gain: float = 0.7              # fraction of along-track correction applied
    max_correction_m: float = 40.0


@dataclass
class ScaleLevelParams:            # T10
    # Pre-blackout speed scale (GNSS / AI). Baseline clips it to [0.85, 1.25] (1.35 on
    # Highway); the ablation showed ~45 % of scenarios sitting on a clip bound.
    enabled: bool = False
    lo: float = 0.85
    hi: float = 1.25
    hi_highway: float = 1.35
    source: str = "entry"          # "entry" (baseline 15 s ratio) | "history" (180 s GNSS-distance ratio) | "blend"
    w_history: float = 0.5         # blend weight of the history ratio


@dataclass
class SpeedScaleFixParams:         # T9
    # Blackout speed reaching the EKF is  v_ai * engine.speed_scale * ekf._speed_scale.
    # Both factors are learned from the SAME pre-blackout GNSS/AI ratio, so the
    # correction is applied twice ("both" = baseline behaviour).
    # "engine": keep engine.speed_scale, reset ekf._speed_scale to 1.0 at blackout start
    # "ekf":    keep ekf._speed_scale,   set engine.speed_scale to 1.0 at blackout start
    source: str = "both"


@dataclass
class Round1Config:
    name: str = "baseline_off"
    velocity_checkpoint: str = ""  # "" = canonical model; "a.pt" or "a.pt,b.pt,c.pt" (mean ensemble)
    diagnostics: bool = False      # hooks active but behaviour unchanged (extra result keys only)
    history_s: float = 180.0       # pre-blackout history used by T3/T4/T7 learners
    scale_fix: SpeedScaleFixParams = field(default_factory=SpeedScaleFixParams)
    scale_level: ScaleLevelParams = field(default_factory=ScaleLevelParams)
    stop: StopDetectorParams = field(default_factory=StopDetectorParams)
    gyro_scale: GyroScaleParams = field(default_factory=GyroScaleParams)
    speed_mode: SpeedModeParams = field(default_factory=SpeedModeParams)
    online_calib: OnlineSpeedCalibParams = field(default_factory=OnlineSpeedCalibParams)
    junction: JunctionAnchorParams = field(default_factory=JunctionAnchorParams)

    def is_all_off(self) -> bool:
        return not (
            self.diagnostics
            or self.scale_fix.source != "both"
            or self.scale_level.enabled
            or self.stop.enabled
            or self.gyro_scale.enabled
            or self.speed_mode.mode != "ai"
            or self.online_calib.enabled
            or self.junction.enabled
        )

    def needs_history(self) -> bool:
        return (self.stop.enabled or self.gyro_scale.enabled or self.online_calib.enabled
                or (self.scale_level.enabled and self.scale_level.source != "entry"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Round1Config":
        cfg = cls()
        sections = {
            "stop": cfg.stop, "gyro_scale": cfg.gyro_scale, "speed_mode": cfg.speed_mode,
            "online_calib": cfg.online_calib, "junction": cfg.junction, "scale_fix": cfg.scale_fix,
            "scale_level": cfg.scale_level,
        }
        for k, v in d.items():
            if k in sections and isinstance(v, dict):
                sec = sections[k]
                for sk, sv in v.items():
                    if not hasattr(sec, sk):
                        raise KeyError(f"Unknown round1 key {k}.{sk}")
                    if sk == "band_edges_mps":
                        sv = tuple(float(x) for x in sv)
                    setattr(sec, sk, sv)
            elif hasattr(cfg, k) and k not in sections:
                setattr(cfg, k, v)
            else:
                raise KeyError(f"Unknown round1 key {k}")
        if cfg.speed_mode.mode not in ("ai", "hold_entry", "entry_offset_decay"):
            raise ValueError(f"Unknown speed_mode.mode {cfg.speed_mode.mode}")
        if cfg.scale_level.source not in ("entry", "history", "blend"):
            raise ValueError(f"Unknown scale_level.source {cfg.scale_level.source}")
        if cfg.scale_fix.source not in ("both", "engine", "ekf"):
            raise ValueError(f"Unknown scale_fix.source {cfg.scale_fix.source}")
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "Round1Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


_ACTIVE: Optional[Round1Config] = None


def set_active_config(cfg: Optional[Round1Config]) -> None:
    global _ACTIVE
    _ACTIVE = cfg


PRODUCTION_PROFILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "round1", "production.json")


def get_active_config() -> Round1Config:
    """
    1. set_active_config(cfg)                       (orchestration scripts, tests)
    2. env SIH_ROUND1_CONFIG=<json path> | "off"    (manual override; "off" = old behaviour)
    3. config/round1/production.json if it exists   (the promoted production profile)
    4. all OFF
    """
    if _ACTIVE is not None:
        return _ACTIVE
    env = os.environ.get("SIH_ROUND1_CONFIG", "").strip()
    if env.lower() == "off":
        return Round1Config()
    if env:
        return Round1Config.from_json(env)
    if os.path.exists(PRODUCTION_PROFILE):
        return Round1Config.from_json(PRODUCTION_PROFILE)
    return Round1Config()
