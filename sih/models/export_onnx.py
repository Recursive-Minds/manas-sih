"""Export Bayesian MoE Fusion Model to standard ONNX format for Edge Runtime."""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.moe_fusion import BayesianMoEFusion


class MoEONNXWrapper(nn.Module):
    """Wrapper that returns clean tensor tuple (v_fused, var_fused) for ONNX runtime."""

    def __init__(self, moe_model: BayesianMoEFusion):
        super().__init__()
        self.moe_model = moe_model

    def forward(self, x_short: torch.Tensor, x_long: torch.Tensor):
        v_fused, var_fused, _ = self.moe_model(x_short, x_long)
        return v_fused, var_fused


def export_moe_to_onnx(
    checkpoint_path: str = "models/checkpoints/best_moe_velocity_model.pt",
    output_onnx_path: str = "models/checkpoints/moe_fusion.onnx",
    short_len: int = 20,
    long_len: int = 60,
    channels: int = 12,
):
    print("=" * 75)
    print("EXPORTING BAYESIAN MoE MODEL TO ONNX")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Destination: {output_onnx_path}")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    in_channels = ckpt.get("in_channels", channels)

    print(f"Instantiating model (channels={in_channels}, short_len={short_len}, long_len={long_len})...")
    expert_resnet = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
    expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
    moe_model = BayesianMoEFusion(expert_resnet=expert_resnet, expert_tcn=expert_tcn)

    if "expert_resnet_state_dict" in ckpt and "expert_tcn_state_dict" in ckpt:
        expert_resnet.load_state_dict(ckpt["expert_resnet_state_dict"])
        expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
    elif "model_state_dict" in ckpt:
        moe_model.load_state_dict(ckpt["model_state_dict"])

    moe_model.eval()
    wrapper = MoEONNXWrapper(moe_model)
    wrapper.eval()

    dummy_short = torch.randn(1, in_channels, short_len, dtype=torch.float32)
    dummy_long = torch.randn(1, in_channels, long_len, dtype=torch.float32)

    dynamic_axes = {
        "x_short": {0: "batch_size"},
        "x_long": {0: "batch_size"},
        "v_fused": {0: "batch_size"},
        "var_fused": {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    print("Exporting ONNX graph with opset_version=14...")
    torch.onnx.export(
        wrapper,
        (dummy_short, dummy_long),
        output_onnx_path,
        input_names=["x_short", "x_long"],
        output_names=["v_fused", "var_fused"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Export successful! ONNX model saved to: {output_onnx_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    ckpt = "models/checkpoints/best_moe_velocity_model.pt"
    out = "models/checkpoints/moe_fusion.onnx"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    export_moe_to_onnx(checkpoint_path=ckpt, output_onnx_path=out)
