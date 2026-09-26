"""
Mobile Edge Model Exporter: ONNX & TorchScript Mobile.

Exports the champion 10 Hz CAN-supervised Bayesian MoE forward velocity model
for on-device edge deployment (Android / iOS / Embedded Linux).
Performs numerical parity verification and CPU latency benchmarking.
"""

import os
import sys
import time
from typing import Optional
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.moe_fusion import BayesianMoEFusion


class MoEEdgeWrapper(torch.nn.Module):
    """Clean wrapper returning forward velocity and variance tensors for mobile runtimes."""

    def __init__(self, moe_model: BayesianMoEFusion):
        super().__init__()
        self.moe_model = moe_model

    def forward(self, x_short: torch.Tensor, x_long: torch.Tensor):
        v_fused, var_fused, _ = self.moe_model(x_short, x_long)
        return v_fused, var_fused


def export_edge_models(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "models/exported",
):
    if checkpoint_path is None:
        from sih.round1.model_select import resolve_velocity_checkpoint
        resolved = resolve_velocity_checkpoint(ROOT_DIR)
        checkpoint_path = resolved.split(",")[0] if resolved else os.path.join(ROOT_DIR, "models", "checkpoints", "best_moe_velocity_model.pt")

    print("=" * 75)
    print("SMARTPHONE INTELLIGENT DEAD RECKONING - MOBILE EDGE MODEL EXPORTER")
    print(f"Source Checkpoint : {checkpoint_path}")
    print(f"Export Directory  : {output_dir}")
    print("=" * 75)

    ckpt_path = os.path.join(ROOT_DIR, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
    out_dir = os.path.join(ROOT_DIR, output_dir) if not os.path.isabs(output_dir) else output_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    # 1. Load Model Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_channels = ckpt.get("in_channels", 12)
    expert_resnet = ResNet1DSpeedEstimator(in_channels=in_channels, base_channels=64)
    expert_tcn = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4)
    expert_resnet.load_state_dict(ckpt["expert_resnet_state_dict"])
    expert_tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
    moe_model = BayesianMoEFusion(expert_resnet=expert_resnet, expert_tcn=expert_tcn)
    moe_model.eval()
    wrapper = MoEEdgeWrapper(moe_model)
    wrapper.eval()

    # Save normalization statistics alongside exported models
    norm_mean = ckpt.get("norm_mean", ckpt.get("mean", np.zeros((in_channels, 1), dtype=np.float32)))
    norm_std = ckpt.get("norm_std", ckpt.get("std", np.ones((in_channels, 1), dtype=np.float32)))
    norm_path = os.path.join(out_dir, "normalization_params.npz")
    np.savez_compressed(norm_path, mean=norm_mean, std=norm_std, in_channels=in_channels)
    print(f"[Export] Saved normalization parameters to: {norm_path}")

    # Dummy inputs for tracing (batch=1, channels=12, short=20, long=60)
    dummy_short = torch.randn(1, in_channels, 20, dtype=torch.float32)
    dummy_long = torch.randn(1, in_channels, 60, dtype=torch.float32)

    # 2. Export TorchScript Mobile (Optimized for Android / iOS PyTorch Mobile)
    ts_path = os.path.join(out_dir, "moe_velocity_model.torchscript.pt")
    traced_model = torch.jit.trace(wrapper, (dummy_short, dummy_long), check_trace=False, strict=False)
    traced_model.save(ts_path)
    ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
    print(f"[Export] TorchScript export successful ({ts_size_mb:.2f} MB)")

    # 3. Export ONNX (for ONNX Runtime Mobile / NNAPI)
    onnx_path = os.path.join(out_dir, "moe_velocity_model.onnx")
    onnx_exported = False
    try:
        import onnx
        print(f"[Export] Exporting ONNX graph to: {onnx_path} ...")
        dynamic_axes = {
            "x_short": {0: "batch_size"},
            "x_long": {0: "batch_size"},
            "v_fused": {0: "batch_size"},
            "var_fused": {0: "batch_size"},
        }
        torch.onnx.export(
            wrapper,
            (dummy_short, dummy_long),
            onnx_path,
            input_names=["x_short", "x_long"],
            output_names=["v_fused", "var_fused"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[Export] ONNX export successful ({onnx_size_mb:.2f} MB)")
        onnx_exported = True
    except ImportError:
        print("[Export] Note: 'onnx' package not found in current environment.")
        print("[Export] Standard TorchScript Mobile export completed successfully.")
        print("[Export] To export ONNX on deployment host: 'pip install onnx onnxruntime'")

    # 4. Export TFLite (FP32)
    tflite_path = os.path.join(out_dir, "moe_velocity_model.tflite")
    tflite_exported = False
    if onnx_exported:
        try:
            print(f"[Export] Exporting TFLite FP32 model to: {tflite_path} ...")
            # Export static ONNX first for optimal TFLite lowering
            static_onnx_path = os.path.join(out_dir, "_moe_static_temp.onnx")
            torch.onnx.export(
                wrapper,
                (dummy_short, dummy_long),
                static_onnx_path,
                input_names=["x_short", "x_long"],
                output_names=["v_fused", "var_fused"],
                opset_version=14,
                do_constant_folding=True,
            )
            import subprocess
            cmd = [
                sys.executable, "-m", "onnx2tf",
                "-i", static_onnx_path,
                "-o", os.path.join(out_dir, "_tflite_temp"),
                "-tb", "flatbuffer_direct",
                "-k", "x_short", "x_long",
                "-n",
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_tflite = os.path.join(out_dir, "_tflite_temp", "_moe_static_temp_float32.tflite")
            if os.path.exists(temp_tflite):
                import shutil
                shutil.copyfile(temp_tflite, tflite_path)
                tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
                print(f"[Export] TFLite export successful ({tflite_size_mb:.2f} MB)")
                tflite_exported = True
                # Cleanup temp files
                shutil.rmtree(os.path.join(out_dir, "_tflite_temp"), ignore_errors=True)
                if os.path.exists(static_onnx_path):
                    os.remove(static_onnx_path)
        except Exception as e:
            print(f"[Export] TFLite export skipped or failed: {e}")

    # 5. Numerical Equivalence & Latency Benchmark across 1000 Feature Windows
    print("\n" + "-" * 75)
    print("VERIFYING NUMERICAL EQUIVALENCE AND LATENCY (1000 Windows)")
    print("-" * 75)

    num_samples = 1000
    # Try extracting real feature windows from S-S3a trip
    real_trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-S3a.csv")
    windows_s_np = None
    windows_l_np = None
    if os.path.exists(real_trip_path):
        try:
            from sih.data.loader import GenericDataLoader
            from sih.calibration.mount import calibrate_stream
            from sih.features.streaming import StreamingFeatureExtractor
            from numpy.lib.stride_tricks import sliding_window_view
            loader = GenericDataLoader()
            trip = loader.load_file(real_trip_path)
            calibs = calibrate_stream(trip, min_samples=30)
            extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
            feats = extractor.batch_extract(calibs[: num_samples + 200])
            norm_feats = (feats.T - norm_mean.reshape(-1, 1)) / (norm_std.reshape(-1, 1) + 1e-6)
            short_len, long_len = 20, 60
            pad_l = np.repeat(norm_feats[:, 0:1], long_len - 1, axis=1)
            padded = np.hstack([pad_l, norm_feats]).astype(np.float32)
            w_l = sliding_window_view(padded, window_shape=long_len, axis=1)
            windows_l_np = np.ascontiguousarray(w_l.transpose(1, 0, 2)).astype(np.float32)[:num_samples]
            windows_s_np = np.ascontiguousarray(windows_l_np[:, :, -short_len:]).astype(np.float32)
            print(f"[Parity] Extracted {num_samples} real feature windows from S-S3a.")
        except Exception as e:
            print(f"[Parity] Note: falling back to synthetic test windows ({e})")

    if windows_s_np is None:
        windows_s_np = np.random.randn(num_samples, in_channels, 20).astype(np.float32)
        windows_l_np = np.random.randn(num_samples, in_channels, 60).astype(np.float32)

    # PyTorch CPU evaluation
    torch.set_num_threads(1)
    with torch.no_grad():
        v_pt_batch, _ = wrapper(torch.from_numpy(windows_s_np), torch.from_numpy(windows_l_np))
        v_pt = v_pt_batch.squeeze(-1).numpy()

    # Latency: PyTorch single-step
    t0 = time.perf_counter()
    for i in range(100):
        with torch.no_grad():
            _ = wrapper(torch.from_numpy(windows_s_np[i : i + 1]), torch.from_numpy(windows_l_np[i : i + 1]))
    t1 = time.perf_counter()
    lat_pt_ms = ((t1 - t0) / 100) * 1000.0

    # ONNX evaluation
    max_diff_onnx = None
    lat_onnx_ms = None
    if onnx_exported:
        from sih.models.predictor import ONNXVelocityPredictor
        p_onnx = ONNXVelocityPredictor(onnx_path, intra_op_threads=1)
        v_onnx = p_onnx.predict_batch(windows_s_np, windows_l_np)
        max_diff_onnx = float(np.max(np.abs(v_pt - v_onnx)))

        t0 = time.perf_counter()
        for i in range(200):
            _ = p_onnx.predict_window(windows_s_np[i : i + 1], windows_l_np[i : i + 1])
        t1 = time.perf_counter()
        lat_onnx_ms = ((t1 - t0) / 200) * 1000.0

    # TFLite evaluation
    max_diff_tflite = None
    lat_tflite_ms = None
    if tflite_exported:
        from sih.models.predictor import TFLiteVelocityPredictor
        p_tflite = TFLiteVelocityPredictor(tflite_path, num_threads=1)
        v_tflite = p_tflite.predict_batch(windows_s_np, windows_l_np)
        max_diff_tflite = float(np.max(np.abs(v_pt - v_tflite)))

        t0 = time.perf_counter()
        for i in range(200):
            _ = p_tflite.predict_window(windows_s_np[i : i + 1], windows_l_np[i : i + 1])
        t1 = time.perf_counter()
        lat_tflite_ms = ((t1 - t0) / 200) * 1000.0

    print(f"PyTorch CPU Latency : {lat_pt_ms:.2f} ms/step ({1000.0/lat_pt_ms:.1f} Hz)")
    if max_diff_onnx is not None:
        print(f"ONNX CPU Parity     : Max Diff = {max_diff_onnx:.6e} m/s | Latency = {lat_onnx_ms:.2f} ms/step ({1000.0/lat_onnx_ms:.1f} Hz)")
    if max_diff_tflite is not None:
        print(f"TFLite CPU Parity   : Max Diff = {max_diff_tflite:.6e} m/s | Latency = {lat_tflite_ms:.2f} ms/step ({1000.0/lat_tflite_ms:.1f} Hz)")
    print("=" * 75)

    return {
        "ts_path": ts_path,
        "onnx_path": onnx_path if onnx_exported else None,
        "tflite_path": tflite_path if tflite_exported else None,
        "ts_size_mb": ts_size_mb,
        "onnx_size_mb": onnx_size_mb if onnx_exported else None,
        "tflite_size_mb": tflite_size_mb if tflite_exported else None,
        "max_diff_onnx": max_diff_onnx,
        "max_diff_tflite": max_diff_tflite,
        "lat_pt_ms": lat_pt_ms,
        "lat_onnx_ms": lat_onnx_ms,
        "lat_tflite_ms": lat_tflite_ms,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mobile Edge Model Exporter: ONNX & TorchScript")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model checkpoint (default: production model)")
    parser.add_argument("--output-dir", "-o", type=str, default="models/exported", help="Path to output directory")
    args, unknown = parser.parse_known_args()
    ckpt = args.checkpoint
    out_d = args.output_dir
    if ckpt is None and unknown:
        ckpt = unknown[0]
        if len(unknown) > 1:
            out_d = unknown[1]
    export_edge_models(checkpoint_path=ckpt, output_dir=out_d)
