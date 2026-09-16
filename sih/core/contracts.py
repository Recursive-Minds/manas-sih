"""
Core immutable data contracts for the Smartphone Intelligent Dead Reckoning (IDR) Pipeline.

Pipeline Data Flow Contract:
IMUSample -> CalibratedSample -> VelocityEstimate -> FusedPosition -> MatchedPosition
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(slots=True, frozen=True)
class IMUSample:
    """
    Raw sensor reading directly from smartphone or external IMU.
    Coordinate frame: Sensor Body Frame (x, y, z).
    """
    timestamp_ns: int
    accel: np.ndarray  # Shape (3,), [ax, ay, az] in m/s^2 (includes gravity)
    gyro: np.ndarray   # Shape (3,), [wx, wy, wz] in rad/s
    mag: Optional[np.ndarray] = None  # Shape (3,), [mx, my, mz] in µT
    temperature_c: Optional[float] = None

    def __post_init__(self) -> None:
        if self.accel.shape != (3,):
            raise ValueError(f"accel must be a 3D vector, got shape {self.accel.shape}")
        if self.gyro.shape != (3,):
            raise ValueError(f"gyro must be a 3D vector, got shape {self.gyro.shape}")
        if self.mag is not None and self.mag.shape != (3,):
            raise ValueError(f"mag must be a 3D vector, got shape {self.mag.shape}")


@dataclass(slots=True, frozen=True)
class GNSSSample:
    """
    Standardized GNSS / GPS measurement.
    """
    timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    speed_mps: Optional[float] = None
    bearing_deg: Optional[float] = None
    accuracy_h_m: float = 5.0
    accuracy_v_m: Optional[float] = None
    is_valid: bool = True


@dataclass(slots=True, frozen=True)
class CalibratedSample:
    """
    IMU reading transformed into the vehicle body coordinate frame:
    - X: Forward along vehicle driving axis
    - Y: Right (lateral)
    - Z: Down (aligned with gravity at rest)
    """
    timestamp_ns: int
    accel_vehicle: np.ndarray            # Shape (3,), [forward, right, down] in m/s^2
    gyro_vehicle: np.ndarray             # Shape (3,), [roll_rate, pitch_rate, yaw_rate] in rad/s
    rotation_body_to_vehicle: np.ndarray # Shape (3, 3) rotation matrix
    gravity_vehicle: np.ndarray          # Shape (3,), estimated gravity direction in vehicle frame
    is_calibrated: bool                  # True if mount alignment has converged

    def __post_init__(self) -> None:
        if self.accel_vehicle.shape != (3,):
            raise ValueError(f"accel_vehicle must be shape (3,), got {self.accel_vehicle.shape}")
        if self.gyro_vehicle.shape != (3,):
            raise ValueError(f"gyro_vehicle must be shape (3,), got {self.gyro_vehicle.shape}")
        if self.rotation_body_to_vehicle.shape != (3, 3):
            raise ValueError(f"rotation matrix must be shape (3, 3), got {self.rotation_body_to_vehicle.shape}")


@dataclass(slots=True, frozen=True)
class VelocityEstimate:
    """
    Output of the AI/classical velocity estimation stage.
    """
    timestamp_ns: int
    forward_speed_mps: float
    speed_variance: float               # Estimated uncertainty for adaptive-noise fusion
    motion_state: str                   # e.g., 'STATIONARY', 'DRIVING', 'TURNING', 'IDLE_VIBRATION'
    lateral_speed_mps: float = 0.0      # Expected ~0 under non-holonomic constraints
    vertical_speed_mps: float = 0.0     # Expected ~0 under non-holonomic constraints


@dataclass(slots=True, frozen=True)
class FusedPosition:
    """
    Output of the INS + GNSS Sensor Fusion Filter.
    Position in both WGS-84 (Lat/Lon) and Local Tangent Plane (ENU).
    """
    timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    position_enu_m: np.ndarray      # Shape (3,), [East, North, Up] in meters
    velocity_enu_mps: np.ndarray    # Shape (3,), [v_East, v_North, v_Up] in m/s
    heading_rad: float              # Vehicle yaw angle (azimuth from True North, clockwise) in radians
    covariance: np.ndarray          # State covariance matrix (e.g. 6x6 or 9x9)
    mode: str                       # 'GNSS_AIDED', 'INS_ONLY_BLACKOUT', 'DEGRADED', 'INITIALIZING'
    gnss_outage_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.position_enu_m.shape != (3,):
            raise ValueError(f"position_enu_m must be shape (3,), got {self.position_enu_m.shape}")
        if self.velocity_enu_mps.shape != (3,):
            raise ValueError(f"velocity_enu_mps must be shape (3,), got {self.velocity_enu_mps.shape}")

    @property
    def east_m(self) -> float:
        return float(self.position_enu_m[0])

    @property
    def north_m(self) -> float:
        return float(self.position_enu_m[1])

    @property
    def up_m(self) -> float:
        return float(self.position_enu_m[2])

    @property
    def is_dead_reckoning(self) -> bool:
        return self.mode in ("INS_ONLY_BLACKOUT", "DEGRADED")


@dataclass(slots=True, frozen=True)
class MatchedPosition:
    """
    Final output bounded to OpenStreetMap road network geometry.
    """
    timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    bearing_deg: float
    road_segment_id: Optional[str] = None
    distance_to_road_m: float = 0.0
    confidence: float = 1.0         # 0.0 (unmatched) to 1.0 (high confidence on-road)
    is_matched: bool = False
