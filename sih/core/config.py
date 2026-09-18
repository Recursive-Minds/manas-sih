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


import json
import os
from typing import Optional


def get_pipeline_config_path() -> str:
    """Returns absolute path to canonical frozen configuration JSON."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "config", "pipeline_config.json")


def load_frozen_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads runtime configuration from config/pipeline_config.json.
    Guarantees consistent parameters across benchmark, live streaming, and unit tests.
    """
    path = config_path or get_pipeline_config_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {
        "pipeline_version": "4.0.0-fallback",
        "defaults": {"map_source": "osm", "enable_route_matching": False},
        "map_matcher": {
            "sigma_dist_m": 4.0,
            "sigma_heading_deg": 30.0,
            "hard_gate_dist_multiplier": 2.5,
            "hard_gate_heading_multiplier": 2.0,
            "ambiguity_min_ratio": 1.5,
            "hysteresis_steps": 2,
            "smoothing_factor": 0.35,
        },
        "entry_segment_acquisition": {
            "search_radii_m": [35.0, 75.0, 150.0],
            "heading_gate_deg": 45.0,
        },
        "route_matcher": {
            "enabled": False,
            "confidence_ratio_threshold": 1.80,
            "max_acceptable_cost": 8.0,
            "min_dist_ratio": 0.70,
            "max_dist_ratio": 1.40,
            "max_routes": 200,
            "sliding_window_s": 5.0,
            "turn_threshold_deg": 30.0,
        },
        "osm_geometry": {
            "douglas_peucker_tol_m": 2.0,
            "max_tile_size_deg": 0.25,
            "bbox_pad_m": 500.0,
            "cache_dir": "data/maps/cache",
        },
        "speed_scaling": {
            "min_scale": 0.85,
            "max_scale_highway": 1.35,
            "max_scale_non_highway": 1.25,
        },
        "kinematics_governor": {
            "a_lat_max_highway": 2.2,
            "a_lat_max_other": 3.5,
            "speed_limit_mps": 33.3,
        },
    }
