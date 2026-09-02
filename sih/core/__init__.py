"""
IDR Core Package: Smartphone Intelligent Dead Reckoning with GNSS Fusion.
"""

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

from sih.core.config import (
    PipelineConfig,
    CalibrationConfig,
    VelocityEstimatorConfig,
    FusionFilterConfig,
    MapMatcherConfig,
    GNSSHandoffConfig,
)

from sih.core.pipeline import (
    IDRPipeline,
    assemble_pipeline,
    register_calibration,
    register_velocity_estimator,
    register_fusion_filter,
    register_map_matcher,
    register_handoff_policy,
)

__all__ = [
    "IMUSample",
    "GNSSSample",
    "CalibratedSample",
    "VelocityEstimate",
    "FusedPosition",
    "MatchedPosition",
    "ICalibration",
    "IVelocityEstimator",
    "IFusionFilter",
    "IMapMatcher",
    "IGNSSHandoffPolicy",
    "PipelineConfig",
    "CalibrationConfig",
    "VelocityEstimatorConfig",
    "FusionFilterConfig",
    "MapMatcherConfig",
    "GNSSHandoffConfig",
    "IDRPipeline",
    "assemble_pipeline",
    "register_calibration",
    "register_velocity_estimator",
    "register_fusion_filter",
    "register_map_matcher",
    "register_handoff_policy",
]
