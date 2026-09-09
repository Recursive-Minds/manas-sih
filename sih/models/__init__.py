"""Models module for AI velocity estimation, spectral features, and Bayesian MoE fusion."""

from sih.models.dataset import IMUVelocityDataset, MultiScaleMoEDataset
from sih.models.tcn_attention import TCNAttentionVelocityModel, gaussian_nll_loss
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.moe_fusion import BayesianMoEFusion, numpy_bayesian_fusion
from sih.models.losses import (
    balanced_velocity_loss,
    phase55_balanced_loss,
    l_dynamics_variance_alignment,
    l_centripetal,
    l_drift_windowed,
    l_jerk_hinge,
)

from sih.models.export_onnx import export_moe_to_onnx

__all__ = [
    "IMUVelocityDataset",
    "MultiScaleMoEDataset",
    "TCNAttentionVelocityModel",
    "ResNet1DSpeedEstimator",
    "BayesianMoEFusion",
    "numpy_bayesian_fusion",
    "gaussian_nll_loss",
    "balanced_velocity_loss",
    "phase55_balanced_loss",
    "l_dynamics_variance_alignment",
    "l_centripetal",
    "l_drift_windowed",
    "l_jerk_hinge",
    "export_moe_to_onnx",
]
