"""
Round-1 hooks called from SteppableDeadReckoningEngine.

The engine holds `self.r1 = Round1EngineHooks.create(domain)`, which returns None
when every flag is OFF, so the baseline path executes exactly the old code.

Order inside engine.step():
    (start_blackout end) r1.on_blackout_start(...)         # T9 scale fix, T3/T5 bookkeeping
    cal      = r1.pre_step_cal(cal)                      # T4 gyro scale (blackout only)
    v_ai_cal = v_pred * speed_scale
    v_ai_cal = r1.adjust_ai_speed(v_pred, v_ai_cal, t)   # T7 band factor, T5 speed mode
    v, stat  = speed_obs_pure.update(cal, v_ai_cal)
    v, stat  = r1.post_observer(obs, v, stat, cal, v_ai_cal, t)   # T3 sticky stop
    ... governor / EKF predict / matcher (unchanged) ...
    r1.post_step(engine, cal, v_map_fwd, t)              # T8 junction anchoring
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np

from sih.round1.config import Round1Config, get_active_config
from sih.round1.gyro_scale import estimate_gyro_scale
from sih.round1.history import PreBlackoutHistory, slice_history_tail
from sih.round1.junction_anchor import TurnJunctionAnchor
from sih.round1.online_speed_calib import BandSpeedCalibrator
from sih.round1.stop_detector import StopDetector


class Round1EngineHooks:
    def __init__(self, cfg: Round1Config, domain: str):
        self.cfg = cfg
        self.domain = domain
        self.history: Optional[PreBlackoutHistory] = None
        self.gyro_scale = 1.0
        self.gyro_info: Dict[str, Any] = {}
        self.band = BandSpeedCalibrator(cfg.online_calib) if cfg.online_calib.enabled else None
        self.stop = StopDetector(cfg.stop) if cfg.stop.enabled else None
        self.junction = TurnJunctionAnchor(cfg.junction) if cfg.junction.enabled else None
        self.v_entry = 0.0
        self.v_ai_entry = 0.0
        self.t_entry_ns = 0
        self._last_t: Optional[int] = None
        self.history_ratio: Optional[float] = None

    @classmethod
    def create(cls, domain: str, cfg: Optional[Round1Config] = None) -> Optional["Round1EngineHooks"]:
        cfg = cfg if cfg is not None else get_active_config()
        if cfg.is_all_off():
            return None
        return cls(cfg, domain)

    @property
    def needs_history(self) -> bool:
        return self.cfg.needs_history()

    # ---- before blackout ------------------------------------------------------------
    def set_history(self, hist: PreBlackoutHistory) -> None:
        self.history = hist
        if self.cfg.gyro_scale.enabled:
            self.gyro_info = estimate_gyro_scale(hist, self.cfg.gyro_scale)
            self.gyro_scale = float(self.gyro_info["scale"])
        if self.band is not None:
            self.band.fit(hist)
        if self.cfg.scale_level.enabled and self.cfg.scale_level.source != "entry":
            win = self.cfg.scale_level.window_s
            lvl = self.band if (self.band is not None and win <= 0) else BandSpeedCalibrator(self.cfg.online_calib)
            if lvl is not self.band:
                lvl.fit(slice_history_tail(hist, win))
            r = lvl.info.get("r_all") if lvl.fitted else None
            self.history_ratio = float(r) if r is not None else None
        if self.stop is not None:
            self.stop.learn_idle_level(hist.gnss_ts_ns, hist.gnss_speed, hist.imu_ts_ns, hist.accel)

    def on_blackout_start(self, engine: Any, recent_ai_speeds: List[float], t_entry_ns: int) -> None:
        # T10 scale level / clip bounds (needs engine.speed_scale_raw from edit E9)
        sl = self.cfg.scale_level
        self.scale_raw = getattr(engine, "speed_scale_raw", None)
        if sl.enabled:
            raw = self.scale_raw
            if sl.source == "history" and self.history_ratio is not None:
                raw = self.history_ratio
            elif sl.source == "blend" and self.history_ratio is not None and raw is not None:
                raw = (1.0 - sl.w_history) * raw + sl.w_history * self.history_ratio
            elif sl.source == "blend" and raw is None:
                raw = self.history_ratio
            if raw is not None:
                hi = sl.hi_highway if self.domain == "Highway" else sl.hi
                engine.speed_scale = float(np.clip(raw, sl.lo, hi))
        # T9 diagnostics + fix for the double speed-scale
        self.scale_engine = float(engine.speed_scale)
        self.scale_ekf = float(engine.ekf_map._speed_scale)
        src = self.cfg.scale_fix.source
        if src == "engine":
            engine.ekf_pure._speed_scale = 1.0
            engine.ekf_map._speed_scale = 1.0
        elif src == "ekf":
            engine.speed_scale = 1.0
        self.v_entry = float(engine.v_entry)
        n = self.cfg.speed_mode.entry_ai_window
        tail = recent_ai_speeds[-n:] if recent_ai_speeds else []
        self.v_ai_entry = float(np.mean(tail)) * float(engine.speed_scale) if len(tail) else self.v_entry
        self.t_entry_ns = int(t_entry_ns)
        self._last_t = int(t_entry_ns)
        if self.stop is not None:
            self.stop.reset(self.v_entry)

    # ---- per step -------------------------------------------------------------------
    def _dt(self, t_curr: int) -> float:
        if self._last_t is None:
            self._last_t = t_curr
            return 0.1
        dt = (t_curr - self._last_t) * 1e-9
        self._last_t = t_curr
        return dt if 0.0 < dt <= 1.0 else 0.1

    def pre_step_cal(self, cal: Any) -> Any:
        if self.gyro_scale == 1.0:
            return cal
        g = np.array(cal.gyro_vehicle, dtype=np.float64).copy()
        g[2] *= self.gyro_scale
        return dataclasses.replace(cal, gyro_vehicle=g)

    def adjust_ai_speed(self, v_pred: float, v_ai_cal: float, t_curr: int) -> float:
        if self.band is not None:
            v_ai_cal = v_ai_cal * self.band.factor(v_ai_cal)
        mode = self.cfg.speed_mode.mode
        if mode == "hold_entry":
            v_ai_cal = self.v_entry
        elif mode == "entry_offset_decay":
            t_rel = (t_curr - self.t_entry_ns) * 1e-9
            offset = (self.v_entry - self.v_ai_entry) * np.exp(-max(0.0, t_rel) / self.cfg.speed_mode.decay_tau_s)
            v_ai_cal = max(0.0, v_ai_cal + offset)
        return float(v_ai_cal)

    def post_observer(self, obs: Any, v: float, is_stat: bool, cal: Any, v_ai_cal: float, t_curr: int):
        dt = self._dt(t_curr)
        self._step_dt = dt
        if self.stop is None:
            return v, is_stat
        stopped = self.stop.update(v_ai_cal, cal.accel_vehicle, cal.gyro_vehicle, dt)
        if stopped:
            obs._v_est = 0.0
            return 0.0, True
        return v, is_stat

    def post_step(self, engine: Any, cal: Any, v_map_fwd: float, t_curr: int) -> None:
        if self.junction is None:
            return
        ekf = engine.ekf_map
        w_z = float(cal.gyro_vehicle[2]) - float(ekf._bg[2])
        self.junction.update(w_z, v_map_fwd, getattr(self, "_step_dt", 0.1), ekf, engine.road_network)

    def summary(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {"r1_config": self.cfg.name, "r1_gyro_scale": self.gyro_scale,
                             "r1_scale_engine": round(getattr(self, "scale_engine", float("nan")), 4),
                             "r1_scale_ekf": round(getattr(self, "scale_ekf", float("nan")), 4),
                             "r1_scale_raw": float("nan") if getattr(self, "scale_raw", None) is None else round(self.scale_raw, 4),
                             "r1_scale_history": float("nan") if self.history_ratio is None else round(self.history_ratio, 4),
                             "r1_scale_effective": round(getattr(self, "scale_engine", float("nan"))
                                                         * getattr(self, "scale_ekf", float("nan")), 4)}
        if self.gyro_info:
            s["r1_gyro_windows"] = self.gyro_info.get("n_windows", 0)
        if self.band is not None:
            s["r1_band_info"] = self.band.info
        if self.stop is not None:
            s["r1_stops"] = self.stop.n_stops
            s["r1_stopped_s"] = round(self.stop.stopped_time_s, 1)
            s["r1_idle_std_gate"] = round(self.stop.accel_std_gate, 3)
        if self.junction is not None:
            ev = self.junction.events
            s["r1_turns"] = len(ev)
            s["r1_anchors"] = int(sum(1 for e in ev if e.get("applied")))
            s["r1_anchor_shift_m"] = [round(e.get("delta_along_m", 0.0), 1) for e in ev if e.get("applied")]
        return s
