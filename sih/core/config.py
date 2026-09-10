"""
Configuration dataclasses for the IDR pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CalibrationConfig:
    algorithm: str = "pass_through"  # e.g., 'pass_through', 'gravity_pca', 'adaptive_quaternion'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VelocityEstimatorConfig:
    algorithm: str = "pass_through"  # e.g., 'pass_through', 'integral', 'ronin_resnet', 'tcn_velocity'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionFilterConfig:
    algorithm: str = "naive_dead_reckoning"  # e.g., 'naive_dead_reckoning', 'ekf_ai_imu', 'ukf'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapMatcherConfig:
    algorithm: str = "pass_through"  # e.g., 'pass_through', 'fmm_osm', 'mappymatch'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GNSSHandoffConfig:
    algorithm: str = "threshold_policy"  # e.g., 'threshold_policy', 'kalman_residual_gate'
    params: Dict[str, Any] = field(default_factory=dict)


HandoffStageConfig = GNSSHandoffConfig


@dataclass
class PipelineConfig:
    """
    Central pipeline configuration.
    Defines which concrete implementations are plugged into each stage.
    """
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    velocity: VelocityEstimatorConfig = field(default_factory=VelocityEstimatorConfig)
    fusion: FusionFilterConfig = field(default_factory=FusionFilterConfig)
    map_matching: MapMatcherConfig = field(default_factory=MapMatcherConfig)
    handoff: GNSSHandoffConfig = field(default_factory=GNSSHandoffConfig)
