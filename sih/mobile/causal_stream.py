"""
Mobile Edge-Native Causal Dead Reckoning Streaming Engine.

Provides an incremental, zero-future-lookahead event-driven interface designed for
direct embedding into an Android Foreground Service (via PyTorch Mobile, ONNX Runtime,
or JNI C++ bindings).
"""

from __future__ import annotations
import os
import math
import collections
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from sih.core.contracts import IMUSample, GNSSSample, FusedPosition, VelocityEstimate
from sih.calibration.mount import MountCalibrator, MountAlignment
from sih.fusion.es_ekf import ErrorStateEKF
from sih.fusion.speed_smoother import CausalSpeedSmoother
from sih.map.network import RoadNetwork
from sih.map.matcher import HMMMapMatcher
from sih.data.geo import geodetic_to_enu, enu_to_geodetic


class MobileDeadReckoningStream:
    """
    Stateful, streaming dead-reckoning engine for smartphone navigation.
    Receives asynchronous sensor callbacks (on_imu_sample, on_gnss_sample) and emits
    real-time fused positions without any historical re-computation.
    """

    def __init__(
        self,
        reference_lat_deg: float = 0.0,
        reference_lon_deg: float = 0.0,
        reference_alt_m: float = 0.0,
        road_network: Optional[RoadNetwork] = None,
        torch_model_path: Optional[str] = None,
        onnx_model_path: Optional[str] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        enable_map_matching: bool = True,
    ):
        self.ref_lat = reference_lat_deg
        self.ref_lon = reference_lon_deg
        self.ref_alt = reference_alt_m
        self.has_ref_coords = (reference_lat_deg != 0.0 or reference_lon_deg != 0.0)

        self.calibrator = MountCalibrator()
        self.alignment: Optional[MountAlignment] = None

        self.ekf = ErrorStateEKF(
            turn_threshold_rad_s=np.radians(1.5),
            cooldown_duration_s=0.5,
            max_gyro_bias_rad_s=np.radians(0.1),
            initial_speed_scale=1.00,
        )
        self.speed_smoother = CausalSpeedSmoother(
            a_max_mps2=3.5,
            a_min_mps2=-5.0,
            tau_s=0.25,
        )

        self.road_net = road_network
        self.enable_map_matching = enable_map_matching and (road_network is not None)
        self.matcher: Optional[HMMMapMatcher] = None
        if self.enable_map_matching and self.road_net is not None:
            self.matcher = HMMMapMatcher(
                road_network=self.road_net,
                reference_lat_deg=self.ref_lat,
                reference_lon_deg=self.ref_lon,
                smoothing_factor=0.35,
            )

        from sih.features.streaming import StreamingFeatureExtractor
        self.feature_extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)

        # Ring buffer for 12-channel rolling features (60 samples @ 10 Hz)
        self.feature_buffer = collections.deque(maxlen=60)
        self.short_len = 20
        self.long_len = 60
        self.in_channels = 12

        # Normalization parameters (auto-load exported parameters if available)
        default_norm_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "exported", "normalization_params.npz")
        default_norm_path = os.path.abspath(default_norm_path)
        if norm_mean is None and norm_std is None and os.path.exists(default_norm_path):
            try:
                npz = np.load(default_norm_path)
                norm_mean = npz["mean"].reshape(12, 1).astype(np.float32)
                norm_std = npz["std"].reshape(12, 1).astype(np.float32)
            except Exception as e:
                print(f"[MobileStream] Normalization auto-load warning: {e}")

        self.norm_mean = norm_mean.reshape(12, 1).astype(np.float32) if norm_mean is not None else np.zeros((12, 1), dtype=np.float32)
        self.norm_std = norm_std.reshape(12, 1).astype(np.float32) if norm_std is not None else np.ones((12, 1), dtype=np.float32)

        # TorchScript mobile model (auto-load exported model if available)
        self.torch_model = None
        if torch_model_path is None and onnx_model_path is None:
            default_ts_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "exported", "moe_velocity_model.torchscript.pt")
            default_ts_path = os.path.abspath(default_ts_path)
            if os.path.exists(default_ts_path):
                torch_model_path = default_ts_path

        if torch_model_path is not None:
            try:
                import torch
                self.torch_model = torch.jit.load(torch_model_path, map_location="cpu")
                self.torch_model.eval()
            except Exception as e:
                print(f"[MobileStream] TorchScript model load warning: {e}")

        # Optional ONNX session
        self.onnx_session = None
        if onnx_model_path is not None:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(onnx_model_path, sess_options=opts)
            except Exception as e:
                print(f"[MobileStream] ONNX Runtime session init warning: {e}")

        # Operational state
        self.last_imu_ts_ns: Optional[int] = None
        self.last_gnss_sample: Optional[GNSSSample] = None
        self.is_gnss_healthy: bool = False
        self.consecutive_outage_samples: int = 0
        self.current_speed_estimate: float = 0.0

    def on_gnss_sample(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float = 0.0,
        speed_mps: Optional[float] = None,
        bearing_deg: Optional[float] = None,
        accuracy_h_m: float = 5.0,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Ingests a 1 Hz satellite observation from Android FusedLocationProviderClient.
        """
        if timestamp_ns is None:
            import time
            timestamp_ns = int(time.time() * 1e9)

        if not self.has_ref_coords:
            self.ref_lat = lat_deg
            self.ref_lon = lon_deg
            self.ref_alt = alt_m
            self.has_ref_coords = True

        gnss = GNSSSample(
            timestamp_ns=timestamp_ns,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            altitude_m=alt_m,
            speed_mps=speed_mps,
            bearing_deg=bearing_deg,
            accuracy_h_m=accuracy_h_m,
            is_valid=(accuracy_h_m <= 25.0 and speed_mps is not None),
        )

        self.last_gnss_sample = gnss
        self.is_gnss_healthy = gnss.is_valid

        # Accumulate GNSS in calibrator for course-over-ground turn alignment
        self.calibrator.observe_gnss(gnss)

        # Reset outage sample counter upon receiving healthy fix
        if self.is_gnss_healthy:
            self.consecutive_outage_samples = 0
            # Condition EKF
            self.ekf.update_gnss(gnss)

    def on_imu_sample(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        timestamp_ns: int,
    ) -> FusedPosition:
        """
        Ingests a single 10 Hz / 50 Hz / 100 Hz IMU sample from SensorEventListener.
        Executes incremental calibration, AI velocity inference, kinematic smoothing,
        and Kalman propagation.

        Returns
        -------
        FusedPosition
            Continuous, sub-lane vehicle coordinate in local ENU and WGS84 coordinates.
        """
        imu = IMUSample(
            timestamp_ns=timestamp_ns,
            accel=np.array([ax, ay, az], dtype=np.float64),
            gyro=np.array([gx, gy, gz], dtype=np.float64),
        )

        dt = 0.1
        if self.last_imu_ts_ns is not None:
            dt = max(0.001, (timestamp_ns - self.last_imu_ts_ns) * 1e-9)
        self.last_imu_ts_ns = timestamp_ns

        # 1. Mount Calibration
        cal = self.calibrator.calibrate(imu)
        if self.alignment is None and self.calibrator.is_calibrated:
            self.alignment = self.calibrator.alignment

        # 2. Extract 12-channel causal kinematic and spectral features
        feat_12 = self.feature_extractor.push(cal)
        self.feature_buffer.append(feat_12)

        # 3. Forward Speed Estimation
        v_raw = self._estimate_forward_speed(cal)

        # 4. Causal Kinematic Slew-Rate and Low-Pass Smoothing
        v_smooth = self.speed_smoother.step(v_raw, dt_s=dt)
        self.current_speed_estimate = v_smooth

        # 5. EKF Kinematic Propagation
        m_state = "STATIONARY" if v_smooth < 0.2 else "DRIVING"
        vel = VelocityEstimate(
            timestamp_ns=timestamp_ns,
            forward_speed_mps=v_smooth,
            speed_variance=0.3,
            motion_state=m_state,
        )
        fused = self.ekf.predict(cal, vel)

        # 6. Topological Map Matching (if available and driving)
        matched = None
        if self.enable_map_matching and self.matcher is not None and v_smooth > 1.0:
            matched = self.matcher.match(fused, ekf=self.ekf, domain="Highway", v_fwd=v_smooth)

        # Update consecutive outage count if no recent healthy GNSS fix
        self.consecutive_outage_samples += 1

        # Use snapped road coordinates if map matching succeeded
        pos_enu = fused.position_enu_m
        if matched is not None and matched.is_matched and matched.matched_position_enu_m is not None:
            pos_enu = np.array([matched.matched_position_enu_m[0], matched.matched_position_enu_m[1], fused.position_enu_m[2]], dtype=np.float64)

        # Transform to WGS-84 lat/lon if reference frame is available
        lat_out, lon_out = self.ref_lat, self.ref_lon
        if self.has_ref_coords:
            lat_out, lon_out, _ = enu_to_geodetic(
                pos_enu[0],
                pos_enu[1],
                pos_enu[2],
                self.ref_lat,
                self.ref_lon,
                self.ref_alt,
            )

        mode = "INS_ONLY_BLACKOUT" if (self.consecutive_outage_samples > 15) else "GNSS_AIDED"
        return FusedPosition(
            timestamp_ns=timestamp_ns,
            latitude_deg=lat_out,
            longitude_deg=lon_out,
            altitude_m=float(pos_enu[2]),
            position_enu_m=pos_enu,
            velocity_enu_mps=fused.velocity_enu_mps,
            heading_rad=fused.heading_rad,
            covariance=fused.covariance,
            mode=mode,
            gnss_outage_duration_s=float(self.consecutive_outage_samples * dt),
        )

    def _estimate_forward_speed(self, cal: Any) -> float:
        """
        Executes real-time forward velocity inference via TorchScript or ONNX Runtime session if loaded,
        falling back to centripetal kinematic fusion.
        """
        if (self.torch_model is not None or self.onnx_session is not None) and len(self.feature_buffer) >= self.long_len:
            buf_arr = np.array(self.feature_buffer, dtype=np.float32).T  # (12, 60)
            norm_buf = (buf_arr - self.norm_mean) / (self.norm_std + 1e-6)
            x_long_np = np.expand_dims(norm_buf, axis=0).astype(np.float32)  # (1, 12, 60)
            x_short_np = x_long_np[:, :, -self.short_len:].copy()  # (1, 12, 20)

            if self.torch_model is not None:
                try:
                    import torch
                    with torch.no_grad():
                        t_short = torch.from_numpy(x_short_np)
                        t_long = torch.from_numpy(x_long_np)
                        v_out, _ = self.torch_model(t_short, t_long)
                        v_pred = float(v_out.squeeze().item())
                    return max(0.0, v_pred)
                except Exception as e:
                    print(f"[MobileStream] TorchScript inference error: {e}")
            elif self.onnx_session is not None:
                try:
                    ort_inputs = {
                        self.onnx_session.get_inputs()[0].name: x_short_np,
                        self.onnx_session.get_inputs()[1].name: x_long_np,
                    }
                    outputs = self.onnx_session.run(None, ort_inputs)
                    v_pred = float(outputs[0][0])
                    return max(0.0, v_pred)
                except Exception as e:
                    print(f"[MobileStream] ONNX inference error: {e}")

        # Kinematic fallback: centripetal relation or last known speed
        if self.is_gnss_healthy and self.last_gnss_sample and self.last_gnss_sample.speed_mps is not None:
            return float(self.last_gnss_sample.speed_mps)

        # Rest detector fallback
        if len(self.feature_buffer) >= 10:
            recent_acc = np.array([f[0:3] for f in list(self.feature_buffer)[-10:]])
            acc_var = float(np.mean(np.var(recent_acc, axis=0)))
            if acc_var < 0.04:
                return 0.0

        return max(0.0, self.current_speed_estimate)

    def get_current_state(self) -> Dict[str, Any]:
        """
        Returns snapshot telemetry for navigation UI and debug logs.
        """
        p_enu = self.ekf._p
        v_enu = self.ekf._v
        hdg_deg = float(np.degrees(self.ekf._heading_rad)) % 360.0
        return {
            "east_m": float(p_enu[0]),
            "north_m": float(p_enu[1]),
            "altitude_m": float(p_enu[2]),
            "speed_mps": float(self.current_speed_estimate),
            "heading_deg": hdg_deg,
            "is_dead_reckoning": (self.consecutive_outage_samples > 15),
            "outage_seconds": self.consecutive_outage_samples * 0.1,
            "mount_calibrated": (self.alignment is not None),
        }
