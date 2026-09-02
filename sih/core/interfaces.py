"""
Abstract interfaces for each stage in the IDR pipeline.

Every stage is strictly decoupled behind an interface:
- ICalibration: Phone Body Frame -> Vehicle Forward/Right/Down Frame
- IVelocityEstimator: CalibratedSample -> Forward Velocity + Uncertainty
- IFusionFilter: CalibratedSample + VelocityEstimate + GNSSSample -> FusedPosition
- IMapMatcher: FusedPosition -> MatchedPosition on Road Geometry
- IGNSSHandoffPolicy: Oversees GNSS quality & outage state transitions
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from sih.core.contracts import (
    IMUSample,
    GNSSSample,
    CalibratedSample,
    VelocityEstimate,
    FusedPosition,
    MatchedPosition,
)


class ICalibration(ABC):
    """
    Interface for dynamic mount-angle auto-detection and frame alignment.
    """
    @abstractmethod
    def reset(self) -> None:
        """Reset calibration state to initial uncalibrated state."""
        pass

    @abstractmethod
    def update(self, imu: IMUSample) -> CalibratedSample:
        """Process an IMU reading and transform into vehicle body frame."""
        pass

    @abstractmethod
    def notify_mount_change(self) -> None:
        """Notify that the device was bumped or remounted, triggering re-alignment."""
        pass

    @abstractmethod
    def is_aligned(self) -> bool:
        """Returns True if the mount orientation estimate is converged."""
        pass


class IVelocityEstimator(ABC):
    """
    Interface for forward velocity estimation and vibration filtering.
    """
    @abstractmethod
    def reset(self) -> None:
        """Reset internal estimator/filter state."""
        pass

    @abstractmethod
    def estimate(self, sample: CalibratedSample) -> VelocityEstimate:
        """Estimate forward speed, motion state, and uncertainty."""
        pass


class IFusionFilter(ABC):
    """
    Interface for GNSS + INS state estimation (e.g. EKF, UKF, Factor Graph).
    """
    @abstractmethod
    def reset(self, initial_gnss: Optional[GNSSSample] = None) -> None:
        """Reset filter state and covariance."""
        pass

    @abstractmethod
    def predict(self, sample: CalibratedSample, vel: Optional[VelocityEstimate] = None) -> FusedPosition:
        """Time propagation / INS mechanization step at IMU rate."""
        pass

    @abstractmethod
    def update_gnss(self, gnss: GNSSSample) -> FusedPosition:
        """Measurement update step when a valid GNSS fix is available."""
        pass

    @abstractmethod
    def get_state(self) -> FusedPosition:
        """Return the current filter state estimate."""
        pass


class IMapMatcher(ABC):
    """
    Interface for road network map matching with kinematic constraints.
    """
    @abstractmethod
    def reset(self) -> None:
        """Reset map matcher trajectory history."""
        pass

    @abstractmethod
    def match(self, position: FusedPosition) -> MatchedPosition:
        """Project continuous position onto road network edges."""
        pass


class IGNSSHandoffPolicy(ABC):
    """
    Interface for detecting GNSS degradation / blackout and controlling smooth handoffs.
    """
    @abstractmethod
    def reset(self) -> None:
        """Reset state machine to initial state."""
        pass

    @abstractmethod
    def evaluate_gnss(self, gnss: GNSSSample, current_state: Optional[FusedPosition]) -> bool:
        """Returns True if GNSS fix should be trusted for measurement update."""
        pass

    @abstractmethod
    def get_mode(self) -> str:
        """Returns current operational mode ('GNSS_AIDED', 'INS_ONLY_BLACKOUT', etc.)."""
        pass
