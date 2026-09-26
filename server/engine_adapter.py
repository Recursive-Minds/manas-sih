"""
server/engine_adapter.py
------------------------
Stage A Engine Adapter:
Wraps sih/mobile/causal_stream.py (MobileDeadReckoningStream).

Responsibilities:
1. Resampling & Anti-Aliasing:
   - Causal 2nd-order low-pass filter (cutoff 4.0 Hz) and 10 Hz decimation for
     variable/high-rate phone IMU streams (e.g. 50-100 Hz).
   - Direct pass-through if stream is already 10.0 Hz nominal.
2. Mount Reuse Guard:
   - Evaluates convergence after 30 accel samples.
   - Compares current phone gravity unit vector with saved alignment's vertical axis.
   - If angular difference < 5.0 degrees: restores saved alignment and sets _yaw_locked = True ("Mount: reused").
   - Else (>= 5.0 degrees): discards saved alignment, calls calibrator.reset() ("mount changed - drive turns").
   - Reports live mount status: "Mount: reused", "Mount: calibrating n/8", "Mount: locked", or "mount changed - drive turns".
3. Strict No-Future-Leak Firewall:
   - In WARMING_UP state: forwards GNSS to stream.
   - In BLACKOUT state: drops GNSS completely; feeds only IMU to stream.
"""

from __future__ import annotations
import os
import sys
import math
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
if os.environ.get("SIH_FORCE_SCIPY_SHIM", "0") == "1":
    from sih.core.scipy_shim import butter, sosfilt, sosfilt_zi
else:
    try:
        from scipy.signal import butter, sosfilt, sosfilt_zi
    except ImportError:
        from sih.core.scipy_shim import butter, sosfilt, sosfilt_zi

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.core.contracts import IMUSample, GNSSSample, FusedPosition, VelocityEstimate, CalibratedSample
from sih.calibration.mount import MountCalibrator, MountAlignment
from sih.mobile.causal_stream import MobileDeadReckoningStream
from sih.map.network import RoadNetwork
from sih.fusion.es_ekf import ErrorStateEKF
from sih.features.streaming import StreamingFeatureExtractor
from sih.engine.speed_observer import KinematicSpeedObserver
from sih.map.governor import RoadKinematicsGovernor
from sih.map.matcher import HMMMapMatcher
from sih.data.geo import geodetic_to_enu, enu_to_geodetic
from sih.engine.dead_reckoning_engine import DeadReckoningEngine, SteppableDeadReckoningEngine
from sih.fusion.speed_smoother import CausalSpeedSmoother


class CausalAntiAliasFilter:
    """
    Causal 2nd-order Butterworth low-pass filter (cutoff 4.0 Hz) for 3D signals.
    Maintains internal filter state across streaming samples.
    """
    def __init__(self, cutoff_hz: float = 4.0, default_fs: float = 50.0) -> None:
        self.cutoff_hz = cutoff_hz
        self.default_fs = default_fs
        self.sos = butter(2, cutoff_hz, btype='lowpass', fs=default_fs, output='sos')
        # zi state for 3 channels: shape (n_sections, 3, 2)
        self.zi = np.zeros((self.sos.shape[0], 3, 2), dtype=np.float64)
        self.initialized = False

    def reset(self) -> None:
        self.zi = np.zeros((self.sos.shape[0], 3, 2), dtype=np.float64)
        self.initialized = False

    def filter_sample(self, val_3d: np.ndarray) -> np.ndarray:
        """Filters a single 3D vector [x, y, z] causally."""
        val_3d = np.asarray(val_3d, dtype=np.float64)
        if not self.initialized:
            # Initialize filter steady-state with first sample to prevent step artifact
            for ch in range(3):
                zi_ch = sosfilt_zi(self.sos) * val_3d[ch]
                self.zi[:, ch, :] = zi_ch
            self.initialized = True

        out = np.zeros(3, dtype=np.float64)
        for ch in range(3):
            # val_3d[ch:ch+1] has shape (1,)
            filtered, self.zi[:, ch, :] = sosfilt(self.sos, val_3d[ch:ch+1], zi=self.zi[:, ch, :])
            out[ch] = filtered[0]
        return out


class EngineAdapterStageA:
    """
    Stage A streaming dead-reckoning engine adapter.
    """
    def __init__(
        self,
        reference_lat_deg: float = 0.0,
        reference_lon_deg: float = 0.0,
        reference_alt_m: float = 0.0,
        road_network: Optional[RoadNetwork] = None,
        saved_alignment: Optional[MountAlignment] = None,
        enable_map_matching: bool = True,
        decimate_gnss_for_seeding: bool = True,
        decimate_gnss_for_mount: bool = False,
        gnss_decimate_interval_s: float = 9.0,
        decimate_gnss: Optional[bool] = None,
    ) -> None:
        if decimate_gnss is not None:
            decimate_gnss_for_seeding = decimate_gnss
            decimate_gnss_for_mount = decimate_gnss

        self.ref_lat = reference_lat_deg
        self.ref_lon = reference_lon_deg
        self.ref_alt = reference_alt_m
        self.road_network = road_network
        self.saved_alignment = saved_alignment
        self.enable_map_matching = enable_map_matching
        self.decimate_gnss_for_seeding = decimate_gnss_for_seeding
        self.decimate_gnss_for_mount = decimate_gnss_for_mount
        self.gnss_decimate_interval_s = gnss_decimate_interval_s
        self.last_mount_gnss_ns: Optional[int] = None
        self.last_seeding_gnss_ns: Optional[int] = None
        self.moving_gnss_fixes_count: int = 0

        self.stream = MobileDeadReckoningStream(
            reference_lat_deg=self.ref_lat,
            reference_lon_deg=self.ref_lon,
            reference_alt_m=self.ref_alt,
            road_network=self.road_network,
            enable_map_matching=self.enable_map_matching,
        )

        # Filters for high-rate IMU decimation
        self.accel_filter = CausalAntiAliasFilter(cutoff_hz=4.0, default_fs=50.0)
        self.gyro_filter = CausalAntiAliasFilter(cutoff_hz=4.0, default_fs=50.0)

        # Decimation state (targets 10.0 Hz / 100 ms)
        self.last_emitted_imu_ns: Optional[int] = None
        self.target_interval_ns: int = int(0.10 * 1e9)  # 100 ms

        # Mount guard state
        self.initial_accels: List[np.ndarray] = []
        self.mount_guard_evaluated: bool = False
        self.mount_reused: bool = False
        self.mount_changed: bool = False

        # State tracking: "WARMING_UP" vs "BLACKOUT"
        self.state: str = "WARMING_UP"
        self.latest_fused_position: Optional[FusedPosition] = None

    def reset(self) -> None:
        self.stream = MobileDeadReckoningStream(
            reference_lat_deg=self.ref_lat,
            reference_lon_deg=self.ref_lon,
            reference_alt_m=self.ref_alt,
            road_network=self.road_network,
            enable_map_matching=self.enable_map_matching,
        )
        self.accel_filter.reset()
        self.gyro_filter.reset()
        self.last_emitted_imu_ns = None
        self.initial_accels.clear()
        self.mount_guard_evaluated = False
        self.mount_reused = False
        self.mount_changed = False
        self.last_mount_gnss_ns = None
        self.last_seeding_gnss_ns = None
        self.moving_gnss_fixes_count = 0
        self.state = "WARMING_UP"
        self.latest_fused_position = None

    def set_blackout(self, active: bool = True) -> None:
        """Enforces the strict blackout state."""
        self.state = "BLACKOUT" if active else "WARMING_UP"

    def get_mount_status(self) -> str:
        """
        Returns the formatted mount status:
        - "Mount: reused"
        - "mount changed - drive turns"
        - "Mount: locked"
        - "Mount: calibrating n/8"
        """
        if self.mount_reused:
            return "Mount: reused"
        if self.mount_changed:
            return "mount changed - drive turns"
        if self.stream.calibrator._yaw_locked:
            return "Mount: locked"
        n_turns = min(len(self.stream.calibrator._turn_events), 8)
        return f"Mount: calibrating {n_turns}/8"

    def get_warmup_status(self) -> Dict[str, Any]:
        """
        Evaluates explicit ready conditions for the warm-up panel.
        """
        gravity_converged = len(self.stream.calibrator._accel_buf) >= 30
        mount_locked = bool(self.mount_reused or self.stream.calibrator._yaw_locked)
        buffer_warm = bool(self.stream.feature_extractor.is_warm)
        alpha_learned = bool(self.moving_gnss_fixes_count >= 3)
        n_turns = min(len(self.stream.calibrator._turn_events), 8)
        turns_str = "reused" if self.mount_reused else f"{n_turns}/8"

        # Mount is ready once initial SO(3) leveling settles (30 accel samples).
        # Dynamic turns (0/8) continuously refine the horizontal yaw axis, but do not hard-block dead-reckoning start.
        mount_ready = bool(mount_locked or self.stream.calibrator.is_calibrated or gravity_converged)
        is_ready = bool(gravity_converged and buffer_warm and mount_ready)

        return {
            "is_ready": is_ready,
            "gravity_converged": gravity_converged,
            "mount_locked": mount_ready,
            "mount_status": self.get_mount_status(),
            "turn_events": n_turns,
            "turn_events_target": 8,
            "turns_display": f"turns: {turns_str}",
            "buffer_warm": buffer_warm,
            "alpha_learned": alpha_learned,
        }

    def _evaluate_mount_guard(self) -> None:
        """
        Compares converged gravity vector with saved alignment:
        If angle < 5.0 deg -> reuse saved alignment and lock yaw.
        Else -> discard saved alignment and require turns.
        """
        if len(self.initial_accels) < 30 or self.mount_guard_evaluated:
            return

        self.mount_guard_evaluated = True
        if self.saved_alignment is None or not self.saved_alignment.is_calibrated:
            return

        g_curr = np.mean(self.initial_accels[:30], axis=0)
        g_curr_norm = g_curr / max(np.linalg.norm(g_curr), 1e-4)

        # saved_alignment.vertical_axis_phone is the unit gravity vector in phone body frame
        g_saved = self.saved_alignment.vertical_axis_phone
        g_saved_norm = g_saved / max(np.linalg.norm(g_saved), 1e-4)

        cos_angle = float(np.clip(np.dot(g_curr_norm, g_saved_norm), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))

        if angle_deg < 5.0:
            # Re-use saved alignment safely
            self.stream.calibrator._alignment = self.saved_alignment
            self.stream.calibrator._yaw_locked = True
            self.stream.alignment = self.saved_alignment
            self.mount_reused = True
            self.mount_changed = False
        else:
            # Mount changed physically!
            self.saved_alignment = None
            self.stream.calibrator.reset()
            self.mount_reused = False
            self.mount_changed = True

    def on_gnss(self, gnss: GNSSSample) -> None:
        """
        STRICT NO-LEAK FIREWALL:
        GNSS observations are ingested ONLY during WARMING_UP.
        In BLACKOUT state, GNSS is dropped immediately.

        Two independent decimation switches:
        1. decimate_gnss_for_mount (default False / full 1 Hz):
           Controls feed to MountCalibrator.observe_gnss. At 1 Hz, 8 turn events
           are accumulated in ~2-3 turns.
        2. decimate_gnss_for_seeding (default True / 9s):
           Controls feed to heading seeding & EKF updates to match IO-VNBD training rate.
        """
        if self.state == "BLACKOUT":
            return  # Firewall: zero future leakage

        # Track moving fixes for alpha learned precondition
        if gnss.is_valid and (gnss.speed_mps or 0.0) >= 2.0:
            self.moving_gnss_fixes_count += 1

        # 1. Update reference coordinates on first valid fix
        if not self.stream.has_ref_coords and (gnss.latitude_deg != 0.0 or gnss.longitude_deg != 0.0):
            self.stream.ref_lat = gnss.latitude_deg
            self.stream.ref_lon = gnss.longitude_deg
            self.stream.ref_alt = gnss.altitude_m or 0.0
            self.stream.has_ref_coords = True

        # 2. Feed MountCalibrator (turn event detection)
        feed_mount = True
        if self.decimate_gnss_for_mount and self.last_mount_gnss_ns is not None:
            dt_m = (gnss.timestamp_ns - self.last_mount_gnss_ns) * 1e-9
            if dt_m < (self.gnss_decimate_interval_s - 0.05):
                feed_mount = False

        if feed_mount:
            self.stream.calibrator.observe_gnss(gnss)
            self.last_mount_gnss_ns = gnss.timestamp_ns

        # 3. Feed Heading Seeding and EKF
        feed_seeding = True
        if self.decimate_gnss_for_seeding and self.last_seeding_gnss_ns is not None:
            dt_s = (gnss.timestamp_ns - self.last_seeding_gnss_ns) * 1e-9
            if dt_s < (self.gnss_decimate_interval_s - 0.05):
                feed_seeding = False

        if feed_seeding:
            self.stream.last_gnss_sample = gnss
            self.stream.is_gnss_healthy = gnss.is_valid
            if gnss.is_valid:
                self.stream.consecutive_outage_samples = 0
                self.stream.ekf.update_gnss(gnss)
            self.last_seeding_gnss_ns = gnss.timestamp_ns

    def on_imu(self, imu: IMUSample) -> Optional[FusedPosition]:
        """
        Ingests IMU sample, applies anti-aliasing low-pass filter,
        decimates to 10.0 Hz, evaluates mount reuse guard, and propagates dead reckoning.
        """
        # Collect for mount guard leveling check
        if len(self.initial_accels) < 30:
            self.initial_accels.append(imu.accel.copy())
            if len(self.initial_accels) == 30:
                self._evaluate_mount_guard()

        # Check if rate decimation is needed
        # If samples arrive >= 15 Hz (dt < 70 ms), apply lowpass filter and decimate to 10 Hz
        if self.last_emitted_imu_ns is not None:
            dt_since_last_emit = imu.timestamp_ns - self.last_emitted_imu_ns
            # If not yet time for a 10 Hz sample, filter and hold
            if dt_since_last_emit < (self.target_interval_ns - int(0.015 * 1e9)):
                self.accel_filter.filter_sample(imu.accel)
                self.gyro_filter.filter_sample(imu.gyro)
                return None

        # Filtered sample
        f_accel = self.accel_filter.filter_sample(imu.accel)
        f_gyro = self.gyro_filter.filter_sample(imu.gyro)
        self.last_emitted_imu_ns = imu.timestamp_ns

        # Execute single causal dead reckoning step
        fused = self.stream.on_imu_sample(
            ax=float(f_accel[0]),
            ay=float(f_accel[1]),
            az=float(f_accel[2]),
            gx=float(f_gyro[0]),
            gy=float(f_gyro[1]),
            gz=float(f_gyro[2]),
            timestamp_ns=imu.timestamp_ns,
        )
        self.latest_fused_position = fused
        return fused


class EngineAdapterStageB:
    """
    Stage B production-parity streaming dead-reckoning engine adapter.
    Thin causal streaming wrapper around sih.engine.dead_reckoning_engine.SteppableDeadReckoningEngine.
    Enforces sample-by-sample causal dead reckoning with zero copied or re-implemented algorithmic logic:
    1. MountCalibrator (gravity leveling + turn detection + reuse guard)
    2. StreamingFeatureExtractor (12-ch causal features) & AI model forward inference
    3. Resampling & anti-aliasing for high-rate phone IMU streams (>15 Hz)
    4. SteppableDeadReckoningEngine session execution:
       - Heading seeding & 1Hz historical GNSS synthesis
       - Dynamic speed scale factor (alpha) learning
       - KinematicSpeedObserver & RoadKinematicsGovernor
       - 15-state ErrorStateEKF & HMMMapMatcher
    5. Strict No-Leak Firewall during blackout
    """
    def __init__(
        self,
        reference_lat_deg: float = 0.0,
        reference_lon_deg: float = 0.0,
        reference_alt_m: float = 0.0,
        road_network: Optional[RoadNetwork] = None,
        saved_alignment: Optional[MountAlignment] = None,
        model: Optional[Any] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        device: Optional[Any] = None,
        domain: str = "Highway",
        enable_map_matching: bool = True,
        enable_speed_scale: bool = True,
        turn_threshold_rad_s: float = np.radians(1.5),
        cooldown_duration_s: float = 0.5,
        max_gyro_bias_rad_s: float = np.radians(0.1),
        smoothing_factor: float = 0.35,
        decimate_gnss_for_seeding: bool = True,
        decimate_gnss_for_mount: bool = False,
        gnss_decimate_interval_s: float = 9.0,
        decimate_gnss: Optional[bool] = None,
        lock_saved_alignment: bool = False,
        use_speed_smoother: bool = True,
        predictor: Optional[Any] = None,
    ) -> None:
        if decimate_gnss is not None:
            decimate_gnss_for_seeding = decimate_gnss
            decimate_gnss_for_mount = decimate_gnss

        self.ref_lat = reference_lat_deg
        self.ref_lon = reference_lon_deg
        self.ref_alt = reference_alt_m
        self.has_ref_coords = (reference_lat_deg != 0.0 or reference_lon_deg != 0.0)
        self.road_network = road_network if enable_map_matching else None
        self.saved_alignment = saved_alignment
        self.lock_saved_alignment = lock_saved_alignment
        self.domain = domain
        self.enable_map_matching = enable_map_matching and (road_network is not None)
        self.enable_speed_scale = enable_speed_scale

        self.turn_threshold_rad_s = turn_threshold_rad_s
        self.cooldown_duration_s = cooldown_duration_s
        self.max_gyro_bias_rad_s = max_gyro_bias_rad_s
        self.smoothing_factor = smoothing_factor

        self.decimate_gnss_for_seeding = decimate_gnss_for_seeding
        self.decimate_gnss_for_mount = decimate_gnss_for_mount
        self.gnss_decimate_interval_s = gnss_decimate_interval_s
        self.last_mount_gnss_ns: Optional[int] = None
        self.last_seeding_gnss_ns: Optional[int] = None
        self.moving_gnss_fixes_count: int = 0
        self.last_moving_gnss: Optional[GNSSSample] = None
        self.last_imu_ts_ns: Optional[int] = None

        # AI model & predictor setup
        self.device = device
        self.model = model
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.predictor = None

        if predictor is not None:
            if isinstance(predictor, str):
                from sih.models.predictor import create_predictor
                self.predictor = create_predictor(predictor, model=model, device=device)
            else:
                self.predictor = predictor
        elif self.model is not None:
            from sih.models.predictor import TorchVelocityPredictor
            self.predictor = TorchVelocityPredictor(self.model, device=self.device)
        else:
            from sih.models.inference import load_ai_model
            from sih.models.predictor import TorchVelocityPredictor
            try:
                import torch
                dev = self.device or torch.device("cpu")
                self.model, self.norm_mean, self.norm_std, _ = load_ai_model(dev)
                self.device = dev
                self.predictor = TorchVelocityPredictor(self.model, device=self.device)
            except Exception as e:
                print(f"[EngineAdapterStageB] Could not load AI model: {e}")

        # If normalization params still not loaded, attempt to load from exported sidecar
        if (self.norm_mean is None or self.norm_std is None):
            norm_sidecar = os.path.join(ROOT_DIR, "models", "exported", "normalization_params.npz")
            if os.path.exists(norm_sidecar):
                try:
                    npz = np.load(norm_sidecar)
                    if self.norm_mean is None:
                        self.norm_mean = npz["mean"].reshape(-1, 1)
                    if self.norm_std is None:
                        self.norm_std = npz["std"].reshape(-1, 1)
                except Exception:
                    pass

        # Decimation & anti-alias filter
        self.accel_filter = CausalAntiAliasFilter(cutoff_hz=4.0, default_fs=50.0)
        self.gyro_filter = CausalAntiAliasFilter(cutoff_hz=4.0, default_fs=50.0)
        self.last_emitted_imu_ns: Optional[int] = None
        self.target_interval_ns: int = int(0.10 * 1e9)

        # State tracking
        self.state: str = "WARMING_UP"
        self.blackout_started: bool = False
        self.blackout_start_ns: Optional[int] = None
        self.latest_fused_position: Optional[FusedPosition] = None

        # Buffers for pre-blackout conditioning
        self.initial_accels: List[np.ndarray] = []
        self.mount_guard_evaluated: bool = False
        self.mount_reused: bool = False
        self.mount_changed: bool = False

        self.recent_gnss_window: List[GNSSSample] = []
        self.recent_imu_calib: List[CalibratedSample] = []
        self.recent_ai_speeds: List[float] = []
        self.recent_ai_ts: List[int] = []

        # Feature extraction & speed smoothing
        self.feature_extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
        self.feature_buf: List[np.ndarray] = []
        self.use_speed_smoother = use_speed_smoother
        self.speed_smoother = CausalSpeedSmoother(a_max_mps2=3.5, a_min_mps2=-5.0, tau_s=0.25) if use_speed_smoother else None

        # Mount calibration
        self.calibrator = MountCalibrator(min_samples=30)
        if saved_alignment is not None and saved_alignment.is_calibrated:
            self.calibrator._alignment = saved_alignment
            self.calibrator._yaw_locked = True
            self.mount_reused = True

        # Wrap SteppableDeadReckoningEngine directly from sih/engine
        self.session = SteppableDeadReckoningEngine(
            reference_lat_deg=self.ref_lat,
            reference_lon_deg=self.ref_lon,
            reference_alt_m=self.ref_alt,
            road_network=self.road_network,
            domain=self.domain,
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            smoothing_factor=self.smoothing_factor,
            enable_speed_scale=self.enable_speed_scale,
        )

        # Per-batch timing split tracking (ms)
        self.batch_features_ms: float = 0.0
        self.batch_model_ms: float = 0.0
        self.batch_ekf_map_ms: float = 0.0

    def reset_batch_timing(self) -> None:
        self.batch_features_ms = 0.0
        self.batch_model_ms = 0.0
        self.batch_ekf_map_ms = 0.0

    def get_batch_timing(self) -> Dict[str, float]:
        return {
            "features_ms": float(self.batch_features_ms),
            "model_ms": float(self.batch_model_ms),
            "ekf_map_ms": float(self.batch_ekf_map_ms),
        }

    @property
    def ekf(self) -> ErrorStateEKF:
        """Returns the active ErrorStateEKF filter (map-matched if enabled, else pure)."""
        return self.session.ekf_map if self.session.matcher is not None else self.session.ekf_pure

    @property
    def speed_scale(self) -> float:
        return self.session.speed_scale

    @property
    def v_entry(self) -> float:
        return self.session.v_entry

    @property
    def entry_segment(self):
        return self.session.entry_segment

    @property
    def matcher(self):
        return self.session.matcher

    @property
    def governor(self):
        return self.session.governor

    def reset(self) -> None:
        self.state = "WARMING_UP"
        self.blackout_started = False
        self.blackout_start_ns = None
        self.latest_fused_position = None
        self.initial_accels.clear()
        self.mount_guard_evaluated = False
        self.mount_reused = False
        self.mount_changed = False
        self.recent_gnss_window.clear()
        self.recent_imu_calib.clear()
        self.recent_ai_speeds.clear()
        self.recent_ai_ts.clear()
        self.feature_extractor.reset()
        self.feature_buf.clear()
        self.speed_smoother.reset()
        self.accel_filter.reset()
        self.gyro_filter.reset()
        self.last_emitted_imu_ns = None
        self.calibrator.reset()
        if self.saved_alignment is not None and self.saved_alignment.is_calibrated:
            self.calibrator._alignment = self.saved_alignment
            self.calibrator._yaw_locked = True
            self.mount_reused = True

        self.last_mount_gnss_ns = None
        self.last_seeding_gnss_ns = None
        self.moving_gnss_fixes_count = 0
        self.last_moving_gnss = None
        self.last_imu_ts_ns = None

        self.session = SteppableDeadReckoningEngine(
            reference_lat_deg=self.ref_lat,
            reference_lon_deg=self.ref_lon,
            reference_alt_m=self.ref_alt,
            road_network=self.road_network,
            domain=self.domain,
            turn_threshold_rad_s=self.turn_threshold_rad_s,
            cooldown_duration_s=self.cooldown_duration_s,
            max_gyro_bias_rad_s=self.max_gyro_bias_rad_s,
            smoothing_factor=self.smoothing_factor,
            enable_speed_scale=self.enable_speed_scale,
        )

    def get_mount_status(self) -> str:
        if self.lock_saved_alignment and self.saved_alignment is not None:
            return "Mount: reused"
        if self.mount_reused:
            return "Mount: reused"
        if self.mount_changed:
            return "mount changed - drive turns"
        if self.calibrator._yaw_locked:
            return "Mount: locked"
        n_turns = min(len(self.calibrator._turn_events), 8)
        return f"Mount: calibrating {n_turns}/8"

    def get_warmup_status(self) -> Dict[str, Any]:
        if self.lock_saved_alignment and self.saved_alignment is not None:
            return {
                "is_ready": True,
                "gravity_converged": True,
                "mount_locked": True,
                "mount_status": "Mount: reused",
                "turn_events": 8,
                "turn_events_target": 8,
                "turns_display": "turns: reused",
                "buffer_warm": True,
                "alpha_learned": True,
            }
        gravity_converged = len(self.calibrator._accel_buf) >= 30
        mount_locked = bool(self.mount_reused or self.calibrator._yaw_locked)
        buffer_warm = bool(self.feature_extractor.is_warm)
        alpha_learned = bool(self.moving_gnss_fixes_count >= 3)
        n_turns = min(len(self.calibrator._turn_events), 8)
        turns_str = "reused" if self.mount_reused else f"{n_turns}/8"
        mount_ready = bool(mount_locked or self.calibrator.is_calibrated or gravity_converged)
        is_ready = bool(gravity_converged and buffer_warm and mount_ready)
        return {
            "is_ready": is_ready,
            "gravity_converged": gravity_converged,
            "mount_locked": mount_ready,
            "mount_status": self.get_mount_status(),
            "turn_events": n_turns,
            "turn_events_target": 8,
            "turns_display": f"turns: {turns_str}",
            "buffer_warm": buffer_warm,
            "alpha_learned": alpha_learned,
        }

    def _evaluate_mount_guard(self) -> None:
        if self.lock_saved_alignment:
            return
        if len(self.initial_accels) < 30 or self.mount_guard_evaluated:
            return
        self.mount_guard_evaluated = True
        if self.saved_alignment is None or not self.saved_alignment.is_calibrated:
            return
        g_curr = np.mean(self.initial_accels[:30], axis=0)
        g_curr_norm = g_curr / max(np.linalg.norm(g_curr), 1e-4)
        g_saved = self.saved_alignment.vertical_axis_phone
        g_saved_norm = g_saved / max(np.linalg.norm(g_saved), 1e-4)
        cos_angle = float(np.clip(np.dot(g_curr_norm, g_saved_norm), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg < 5.0:
            self.calibrator._alignment = self.saved_alignment
            self.calibrator._yaw_locked = True
            self.mount_reused = True
            self.mount_changed = False
        else:
            self.saved_alignment = None
            self.calibrator.reset()
            self.mount_reused = False
            self.mount_changed = True

    def on_gnss(self, gnss: GNSSSample) -> None:
        """Strict No-Leak Firewall: dropped immediately if in BLACKOUT."""
        if self.state == "BLACKOUT":
            return

        if gnss.is_valid and (gnss.speed_mps or 0.0) >= 2.0:
            self.moving_gnss_fixes_count += 1
            self.last_moving_gnss = gnss

        # 1. Update reference coordinates
        if not self.has_ref_coords and (gnss.latitude_deg != 0.0 or gnss.longitude_deg != 0.0):
            self.ref_lat = gnss.latitude_deg
            self.ref_lon = gnss.longitude_deg
            self.ref_alt = gnss.altitude_m or 0.0
            self.has_ref_coords = True
            self.session.ref_lat = self.ref_lat
            self.session.ref_lon = self.ref_lon
            self.session.ref_alt = self.ref_alt
            if self.session.matcher is not None:
                self.session.matcher.ref_lat = self.ref_lat
                self.session.matcher.ref_lon = self.ref_lon

        # 2. Store in historical window
        self.recent_gnss_window.append(gnss)

        # 3. Feed MountCalibrator
        feed_mount = True
        if self.decimate_gnss_for_mount and self.last_mount_gnss_ns is not None:
            dt_m = (gnss.timestamp_ns - self.last_mount_gnss_ns) * 1e-9
            if dt_m < (self.gnss_decimate_interval_s - 0.05):
                feed_mount = False
        if feed_mount:
            self.calibrator.observe_gnss(gnss)
            self.last_mount_gnss_ns = gnss.timestamp_ns

        # 4. Feed EKF during warmup
        feed_seeding = True
        if self.decimate_gnss_for_seeding and self.last_seeding_gnss_ns is not None:
            dt_s = (gnss.timestamp_ns - self.last_seeding_gnss_ns) * 1e-9
            if dt_s < (self.gnss_decimate_interval_s - 0.05):
                feed_seeding = False
        if feed_seeding and gnss.is_valid:
            if not self.session.ekf_pure._initialised or (self.session.ekf_pure._ref[0] == 0.0 and self.session.ekf_pure._ref[1] == 0.0):
                self.session.init_from_gnss(gnss)
            else:
                self.session.update_gnss_warmup(gnss)
            self.last_seeding_gnss_ns = gnss.timestamp_ns

    def set_blackout(self, active: bool = True, entry_gnss: Optional[GNSSSample] = None) -> None:
        """
        Transitions to BLACKOUT and triggers heading seeding, alpha estimation,
        entry road segment acquisition, and speed observer anchor via SteppableDeadReckoningEngine.
        """
        if not active:
            self.state = "WARMING_UP"
            self.blackout_started = False
            return

        self.state = "BLACKOUT"
        self.blackout_started = True

        g_ref = entry_gnss
        if g_ref is None and self.recent_gnss_window:
            valid_g = [g for g in self.recent_gnss_window if g.is_valid]
            if valid_g:
                g_ref = valid_g[-1]

        t_entry_ns = g_ref.timestamp_ns if g_ref is not None else (self.last_imu_ts_ns or 0)
        self.blackout_start_ns = t_entry_ns

        if g_ref is not None and self.has_ref_coords:
            entry_pos_enu = geodetic_to_enu(
                g_ref.latitude_deg, g_ref.longitude_deg, 0.0,
                self.ref_lat, self.ref_lon, 0.0
            )[:2]
        else:
            entry_pos_enu = self.session.ekf_map._p[:2].copy()

        valid_hist_gnss = [g for g in self.recent_gnss_window if g.is_valid and g.timestamp_ns <= t_entry_ns]
        pre_gnss_window = SteppableDeadReckoningEngine.synthesize_1hz_gnss_window(
            valid_hist_gnss, t_entry_ns, self.ref_lat, self.ref_lon
        )
        from sih.round1.entry_bearing import live_override_enabled  # [ROUND1] R2
        if g_ref is not None and g_ref.bearing_deg is not None and pre_gnss_window and live_override_enabled():
            # GNSSSample is a frozen dataclass: replace the last element with preserved Doppler bearing
            last_g = pre_gnss_window[-1]
            pre_gnss_window[-1] = GNSSSample(
                timestamp_ns=last_g.timestamp_ns,
                latitude_deg=last_g.latitude_deg,
                longitude_deg=last_g.longitude_deg,
                altitude_m=last_g.altitude_m,
                speed_mps=g_ref.speed_mps if g_ref.speed_mps is not None else last_g.speed_mps,
                bearing_deg=g_ref.bearing_deg,
                accuracy_h_m=last_g.accuracy_h_m,
                is_valid=last_g.is_valid,
            )

        cal_entry = self.recent_imu_calib[-1] if self.recent_imu_calib else CalibratedSample(
            timestamp_ns=t_entry_ns,
            accel_vehicle=np.array([0.0, 0.0, 9.81], dtype=np.float64),
            gyro_vehicle=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_body_to_vehicle=np.eye(3, dtype=np.float64),
            gravity_vehicle=np.array([0.0, 0.0, 9.81], dtype=np.float64),
            is_calibrated=True,
        )

        # [ROUND1] live pre-blackout history (same data the benchmark uses) for T7 / T10 learners
        if self.session.r1 is not None and self.session.r1.needs_history:
            from sih.round1.history import build_history_from_buffers
            self.session.r1.set_history(build_history_from_buffers(
                self.recent_imu_calib, self.recent_ai_speeds, valid_hist_gnss,
                self.ref_lat, self.ref_lon, t_entry_ns, self.session.r1.cfg.history_s))

        self.session.start_blackout(
            entry_pos_enu=entry_pos_enu,
            pre_gnss_window=pre_gnss_window,
            recent_ai_speeds=self.recent_ai_speeds,
            recent_imu_calib=self.recent_imu_calib,
            cal_entry=cal_entry,
            t_entry_ns=t_entry_ns,
        )

    def _infer_speed(self, cal: CalibratedSample, override_speed: Optional[float] = None) -> float:
        if override_speed is not None:
            return float(override_speed)

        # Causal feature extraction
        t_feat0 = time.perf_counter()
        f = self.feature_extractor.push(cal)
        self.feature_buf.append(f)
        if len(self.feature_buf) > 60:
            self.feature_buf.pop(0)

        if self.model is None and self.predictor is None or len(self.feature_buf) < 1:
            self.batch_features_ms += (time.perf_counter() - t_feat0) * 1000.0
            return max(0.0, float(np.linalg.norm(self.session.ekf_map._v)))

        # Build short (20) and long (60) feature windows with pad repetition
        if len(self.feature_buf) < 60:
            pad = [self.feature_buf[0]] * (60 - len(self.feature_buf))
            w_l = np.array(pad + self.feature_buf, dtype=np.float32).T
        else:
            w_l = np.array(self.feature_buf[-60:], dtype=np.float32).T

        if self.norm_mean is not None and self.norm_std is not None:
            norm_w_l = (w_l - self.norm_mean.reshape(-1, 1)) / (self.norm_std.reshape(-1, 1) + 1e-6)
        else:
            norm_w_l = w_l

        norm_w_s = norm_w_l[:, -20:].copy()
        self.batch_features_ms += (time.perf_counter() - t_feat0) * 1000.0

        t_mod0 = time.perf_counter()
        if self.predictor is not None:
            v_raw, _ = self.predictor.predict_window(norm_w_s, norm_w_l)
        elif self.model is not None:
            import torch
            dev = self.device or torch.device("cpu")
            ts_s = torch.from_numpy(norm_w_s[None, ...]).to(dev)
            ts_l = torch.from_numpy(norm_w_l[None, ...]).to(dev)
            with torch.no_grad():
                vf, _, _ = self.model(ts_s, ts_l)
            v_raw = float(vf.item())
        else:
            self.batch_model_ms += (time.perf_counter() - t_mod0) * 1000.0
            return max(0.0, float(np.linalg.norm(self.session.ekf_map._v)))

        self.batch_model_ms += (time.perf_counter() - t_mod0) * 1000.0

        if self.use_speed_smoother and self.speed_smoother is not None:
            return self.speed_smoother.update(v_raw, dt_s=0.1)
        return v_raw

    def prime_features(
        self,
        imu_samples: List[IMUSample],
        calib_samples: Optional[List[CalibratedSample]] = None,
    ) -> None:
        """
        Pre-rolls the feature extractor and AI speed model over historical IMU samples
        from trip / app start up to the streaming connection point.
        Runs ONLY feature extraction + model forward (+ smoother if enabled).
        Appends to recent_ai_speeds / recent_imu_calib exactly as on_imu does,
        without touching the EKF session, GNSS buffers, or filter state machines.
        """
        if not imu_samples:
            return

        N = len(imu_samples)
        c_list: List[CalibratedSample] = []
        for i, imu in enumerate(imu_samples):
            if calib_samples is not None and i < len(calib_samples):
                c = calib_samples[i]
            else:
                c = self.calibrator.update(imu)
            c_list.append(c)
            self.recent_imu_calib.append(c)
            self.recent_ai_ts.append(imu.timestamp_ns)

        if self.model is None:
            for c in c_list:
                f = self.feature_extractor.push(c)
                self.feature_buf.append(f)
                if len(self.feature_buf) > 60:
                    self.feature_buf.pop(0)
                self.recent_ai_speeds.append(0.0)
            return

        feats = []
        for c in c_list:
            f = self.feature_extractor.push(c)
            self.feature_buf.append(f)
            if len(self.feature_buf) > 60:
                self.feature_buf.pop(0)
            feats.append(f)

        feats_arr = np.array(feats, dtype=np.float32)
        norm_mean = self.norm_mean if self.norm_mean is not None else np.zeros(12, dtype=np.float32)
        norm_std = self.norm_std if self.norm_std is not None else np.ones(12, dtype=np.float32)
        norm_feats = (feats_arr.T - norm_mean) / (norm_std + 1e-6)
        short_len, long_len = 20, 60
        pad_l = np.repeat(norm_feats[:, 0:1], long_len - 1, axis=1)
        padded_feats = np.hstack([pad_l, norm_feats]).astype(np.float32)

        from numpy.lib.stride_tricks import sliding_window_view
        windows_l = sliding_window_view(padded_feats, window_shape=long_len, axis=1)
        windows_l = np.ascontiguousarray(windows_l.transpose(1, 0, 2)).astype(np.float32)
        windows_s = np.ascontiguousarray(windows_l[:, :, -short_len:]).astype(np.float32)

        if self.predictor is not None:
            preds = self.predictor.predict_batch(windows_s, windows_l)
        else:
            import torch
            dev = self.device or torch.device("cpu")
            batch_size = 2048
            preds = []
            with torch.no_grad():
                for b in range(0, N, batch_size):
                    b_s = torch.from_numpy(windows_s[b : b + batch_size]).to(dev)
                    b_l = torch.from_numpy(windows_l[b : b + batch_size]).to(dev)
                    vf, _, _ = self.model(b_s, b_l)
                    preds.extend(vf.squeeze(-1).float().cpu().numpy().flatten())

        for v_raw in preds:
            v_val = float(v_raw)
            if self.use_speed_smoother and self.speed_smoother is not None:
                v_val = self.speed_smoother.update(v_val, dt_s=0.1)
            self.recent_ai_speeds.append(v_val)

    def on_imu(
        self,
        imu: IMUSample,
        override_speed: Optional[float] = None,
        pre_calibrated: Optional[CalibratedSample] = None,
    ) -> Optional[FusedPosition]:
        self.last_imu_ts_ns = imu.timestamp_ns

        # Collect for mount guard leveling check
        if len(self.initial_accels) < 30:
            self.initial_accels.append(imu.accel.copy())
            if len(self.initial_accels) == 30:
                self._evaluate_mount_guard()

        # Resampling & anti-aliasing: filter & decimate only for high-rate phone streams (> 15 Hz)
        if pre_calibrated is None:
            is_high_rate = False
            if self.last_emitted_imu_ns is not None:
                dt_since_last_emit = imu.timestamp_ns - self.last_emitted_imu_ns
                if 0 <= dt_since_last_emit < int(0.065 * 1e9):
                    is_high_rate = True
                    self.accel_filter.filter_sample(imu.accel)
                    self.gyro_filter.filter_sample(imu.gyro)
                    return None
                elif dt_since_last_emit < 0 or dt_since_last_emit > int(1.0 * 1e9):
                    # Stream timestamp discontinuity or new source: reset filters
                    self.accel_filter.reset()
                    self.gyro_filter.reset()

            if is_high_rate:
                f_accel = self.accel_filter.filter_sample(imu.accel)
                f_gyro = self.gyro_filter.filter_sample(imu.gyro)
            else:
                f_accel = imu.accel
                f_gyro = imu.gyro

            self.last_emitted_imu_ns = imu.timestamp_ns
            imu_clean = IMUSample(timestamp_ns=imu.timestamp_ns, accel=f_accel, gyro=f_gyro)
            cal = self.calibrator.update(imu_clean)
        else:
            cal = pre_calibrated

        self.recent_imu_calib.append(cal)

        # Speed inference
        v_raw = self._infer_speed(cal, override_speed=override_speed)
        self.recent_ai_speeds.append(v_raw)
        self.recent_ai_ts.append(imu.timestamp_ns)

        # Predict warmup
        t_ekf0 = time.perf_counter()
        if self.state == "WARMING_UP":
            if self.has_ref_coords and self.session.ekf_pure._initialised and (self.session.ekf_pure._ref[0] != 0.0 or self.session.ekf_pure._ref[1] != 0.0):
                self.session.predict_warmup(cal, v_raw, imu.timestamp_ns)
            self.batch_ekf_map_ms += (time.perf_counter() - t_ekf0) * 1000.0
            return None

        # BLACKOUT propagation
        step_res = self.session.step(cal, v_raw, imu.timestamp_ns)
        self.batch_ekf_map_ms += (time.perf_counter() - t_ekf0) * 1000.0

        pos_enu = self.session.ekf_map._p
        if step_res.matched_pos is not None and step_res.matched_pos.is_matched:
            lat_out = step_res.matched_pos.latitude_deg
            lon_out = step_res.matched_pos.longitude_deg
        else:
            lat_out, lon_out, _ = enu_to_geodetic(
                pos_enu[0], pos_enu[1], pos_enu[2],
                self.ref_lat, self.ref_lon, self.ref_alt
            )

        self.latest_fused_position = FusedPosition(
            timestamp_ns=imu.timestamp_ns,
            latitude_deg=lat_out,
            longitude_deg=lon_out,
            altitude_m=float(pos_enu[2]),
            position_enu_m=pos_enu.copy(),
            velocity_enu_mps=self.session.ekf_map._v.copy(),
            heading_rad=self.session.ekf_map._heading_rad,
            covariance=self.session.ekf_map._P.copy(),
            mode="INS_ONLY_BLACKOUT",
        )
        return self.latest_fused_position

