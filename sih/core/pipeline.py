"""
Central Pipeline Assembly Point for Smartphone Intelligent Dead Reckoning.

Data Flow Contract:
IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np

from sih.core.contracts import (
    IMUSample,
    GNSSSample,
    CalibratedSample,
    VelocityEstimate,
    FusedPosition,
    MatchedPosition,
)
from sih.core.interfaces import (
    ICalibration,
    IVelocityEstimator,
    IFusionFilter,
    IMapMatcher,
    IGNSSHandoffPolicy,
)
from sih.core.config import PipelineConfig


# -------------------------------------------------------------------------
# Default Pass-Through Implementations (Stubs for Stage Testing)
# -------------------------------------------------------------------------

class PassThroughCalibration(ICalibration):
    """Assumes phone is already aligned with vehicle frame (Identity rotation)."""
    def __init__(self, **params) -> None:
        self._is_aligned = True

    def reset(self) -> None:
        self._is_aligned = True

    def update(self, imu: IMUSample) -> CalibratedSample:
        return CalibratedSample(
            timestamp_ns=imu.timestamp_ns,
            accel_vehicle=imu.accel.copy(),
            gyro_vehicle=imu.gyro.copy(),
            rotation_body_to_vehicle=np.eye(3, dtype=np.float64),
            gravity_vehicle=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            is_calibrated=self._is_aligned,
        )

    def notify_mount_change(self) -> None:
        pass

    def is_aligned(self) -> bool:
        return self._is_aligned


class PassThroughVelocityEstimator(IVelocityEstimator):
    """Pass-through velocity estimator."""
    def __init__(self, **params) -> None:
        pass

    def reset(self) -> None:
        pass

    def estimate(self, sample: CalibratedSample) -> VelocityEstimate:
        return VelocityEstimate(
            timestamp_ns=sample.timestamp_ns,
            forward_speed_mps=0.0,
            speed_variance=1.0,
            motion_state="UNKNOWN",
            lateral_speed_mps=0.0,
            vertical_speed_mps=0.0,
        )


class PassThroughMapMatcher(IMapMatcher):
    """Direct pass-through without road snapping."""
    def __init__(self, **params) -> None:
        pass

    def reset(self) -> None:
        pass

    def match(self, position: FusedPosition) -> MatchedPosition:
        bearing = float(np.degrees(position.heading_rad)) % 360.0
        return MatchedPosition(
            timestamp_ns=position.timestamp_ns,
            latitude_deg=position.latitude_deg,
            longitude_deg=position.longitude_deg,
            bearing_deg=bearing,
            road_segment_id=None,
            distance_to_road_m=0.0,
            confidence=1.0,
            is_matched=False,
        )


class ThresholdGNSSHandoffPolicy(IGNSSHandoffPolicy):
    """Simple threshold-based GNSS quality gating."""
    def __init__(self, max_accuracy_h_m: float = 25.0, blackout_timeout_s: float = 3.0, **params) -> None:
        self.max_accuracy_h_m = max_accuracy_h_m
        self.blackout_timeout_s = blackout_timeout_s
        self._mode = "INITIALIZING"
        self._last_valid_gnss_ts_ns: Optional[int] = None

    def reset(self) -> None:
        self._mode = "INITIALIZING"
        self._last_valid_gnss_ts_ns = None

    def evaluate_gnss(self, gnss: GNSSSample, current_state: Optional[FusedPosition]) -> bool:
        is_acceptable = gnss.is_valid and (gnss.accuracy_h_m <= self.max_accuracy_h_m)
        if is_acceptable:
            self._mode = "GNSS_AIDED"
            self._last_valid_gnss_ts_ns = gnss.timestamp_ns
            return True
        else:
            if self._last_valid_gnss_ts_ns is not None:
                elapsed_s = (gnss.timestamp_ns - self._last_valid_gnss_ts_ns) * 1e-9
                if elapsed_s > self.blackout_timeout_s:
                    self._mode = "INS_ONLY_BLACKOUT"
                else:
                    self._mode = "DEGRADED"
            return False

    def get_mode(self) -> str:
        return self._mode


# -------------------------------------------------------------------------
# Factory Registries (Single Assembly Point)
# -------------------------------------------------------------------------

CALIBRATION_FACTORIES: Dict[str, Callable[..., ICalibration]] = {
    "pass_through": lambda **params: PassThroughCalibration(**params),
}

VELOCITY_FACTORIES: Dict[str, Callable[..., IVelocityEstimator]] = {
    "pass_through": lambda **params: PassThroughVelocityEstimator(**params),
}

FUSION_FACTORIES: Dict[str, Callable[..., IFusionFilter]] = {}

from sih.map.matcher import HMMMapMatcher

MAP_MATCHER_FACTORIES: Dict[str, Callable[..., IMapMatcher]] = {
    "pass_through": lambda **params: PassThroughMapMatcher(**params),
    "hmm_matcher": lambda **params: HMMMapMatcher(**params),
}

HANDOFF_FACTORIES: Dict[str, Callable[..., IGNSSHandoffPolicy]] = {
    "threshold_policy": lambda **params: ThresholdGNSSHandoffPolicy(**params),
}


def register_calibration(name: str, factory: Callable[..., ICalibration]) -> None:
    CALIBRATION_FACTORIES[name] = factory


def register_velocity_estimator(name: str, factory: Callable[..., IVelocityEstimator]) -> None:
    VELOCITY_FACTORIES[name] = factory


def register_fusion_filter(name: str, factory: Callable[..., IFusionFilter]) -> None:
    FUSION_FACTORIES[name] = factory


def register_map_matcher(name: str, factory: Callable[..., IMapMatcher]) -> None:
    MAP_MATCHER_FACTORIES[name] = factory


def register_handoff_policy(name: str, factory: Callable[..., IGNSSHandoffPolicy]) -> None:
    HANDOFF_FACTORIES[name] = factory


# -------------------------------------------------------------------------
# Central Pipeline
# -------------------------------------------------------------------------

class IDRPipeline:
    """
    Core IDR Pipeline orchestrating all stages.
    """
    def __init__(
        self,
        calibration: ICalibration,
        velocity_estimator: IVelocityEstimator,
        fusion_filter: IFusionFilter,
        map_matcher: IMapMatcher,
        handoff_policy: IGNSSHandoffPolicy,
    ) -> None:
        self.calibration = calibration
        self.velocity_estimator = velocity_estimator
        self.fusion_filter = fusion_filter
        self.map_matcher = map_matcher
        self.handoff_policy = handoff_policy

    def reset(self, initial_gnss: Optional[GNSSSample] = None) -> None:
        self.calibration.reset()
        self.velocity_estimator.reset()
        self.fusion_filter.reset(initial_gnss)
        self.map_matcher.reset()
        self.handoff_policy.reset()

    def process_imu(self, imu: IMUSample) -> Tuple[CalibratedSample, VelocityEstimate, FusedPosition, MatchedPosition]:
        """
        Process a single high-frequency IMU sample through the full pipeline.
        Data flow:
        IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition
        """
        # 1. Calibration Stage
        calib_sample = self.calibration.update(imu)

        # 2. Velocity Estimation Stage
        vel_estimate = self.velocity_estimator.estimate(calib_sample)

        # 3. Fusion Prediction Stage (INS mechanization / propagation)
        fused_pos = self.fusion_filter.predict(calib_sample, vel_estimate)

        # 4. Map Matching Stage
        matched_pos = self.map_matcher.match(fused_pos)

        return calib_sample, vel_estimate, fused_pos, matched_pos

    def process_gnss(self, gnss: GNSSSample) -> Optional[FusedPosition]:
        """
        Process a GNSS measurement update when received.
        """
        current_state = None
        try:
            current_state = self.fusion_filter.get_state()
        except Exception:
            pass

        # Feed calibration stage
        if hasattr(self.calibration, "observe_gnss"):
            self.calibration.observe_gnss(gnss)

        # Evaluate quality / blackout status
        is_trusted = self.handoff_policy.evaluate_gnss(gnss, current_state)
        if is_trusted:
            return self.fusion_filter.update_gnss(gnss)
        return current_state


def assemble_pipeline(config: PipelineConfig) -> IDRPipeline:
    """
    Single assembly point: builds and connects concrete stages based on PipelineConfig.
    """
    calib_algo = config.calibration.algorithm
    if calib_algo not in CALIBRATION_FACTORIES:
        raise ValueError(f"Unknown calibration algorithm: {calib_algo}. Available: {list(CALIBRATION_FACTORIES.keys())}")
    calibration = CALIBRATION_FACTORIES[calib_algo](**config.calibration.params)

    vel_algo = config.velocity.algorithm
    if vel_algo not in VELOCITY_FACTORIES:
        raise ValueError(f"Unknown velocity algorithm: {vel_algo}. Available: {list(VELOCITY_FACTORIES.keys())}")
    velocity_estimator = VELOCITY_FACTORIES[vel_algo](**config.velocity.params)

    fusion_algo = config.fusion.algorithm
    if fusion_algo not in FUSION_FACTORIES:
        raise ValueError(f"Unknown fusion algorithm: {fusion_algo}. Available: {list(FUSION_FACTORIES.keys())}")
    fusion_filter = FUSION_FACTORIES[fusion_algo](**config.fusion.params)

    map_algo = config.map_matching.algorithm
    if map_algo not in MAP_MATCHER_FACTORIES:
        raise ValueError(f"Unknown map matcher algorithm: {map_algo}. Available: {list(MAP_MATCHER_FACTORIES.keys())}")
    map_matcher = MAP_MATCHER_FACTORIES[map_algo](**config.map_matching.params)

    handoff_algo = config.handoff.algorithm
    if handoff_algo not in HANDOFF_FACTORIES:
        raise ValueError(f"Unknown handoff algorithm: {handoff_algo}. Available: {list(HANDOFF_FACTORIES.keys())}")
    handoff_policy = HANDOFF_FACTORIES[handoff_algo](**config.handoff.params)

    return IDRPipeline(
        calibration=calibration,
        velocity_estimator=velocity_estimator,
        fusion_filter=fusion_filter,
        map_matcher=map_matcher,
        handoff_policy=handoff_policy,
    )
