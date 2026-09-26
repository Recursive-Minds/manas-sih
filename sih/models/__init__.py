"""Models module for AI velocity estimation, spectral features, and Bayesian MoE fusion."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sih.models.dataset import IMUVelocityDataset, MultiScaleMoEDataset

from sih.models.predictor import (
    VelocityPredictor,
    TorchVelocityPredictor,
    ONNXVelocityPredictor,
    TFLiteVelocityPredictor,
    JavaBridgeVelocityPredictor,
    create_predictor,
)

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

if _HAVE_TORCH:
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
    from sih.models.inference import load_ai_model, predict_velocities

__all__ = [
    "VelocityPredictor",
    "TorchVelocityPredictor",
    "ONNXVelocityPredictor",
    "TFLiteVelocityPredictor",
    "JavaBridgeVelocityPredictor",
    "create_predictor",
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
    "load_ai_model",
    "predict_velocities",
]


def __getattr__(name: str) -> Any:
    if name in ("IMUVelocityDataset", "MultiScaleMoEDataset"):
        from sih.models.dataset import IMUVelocityDataset, MultiScaleMoEDataset
        return IMUVelocityDataset if name == "IMUVelocityDataset" else MultiScaleMoEDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

