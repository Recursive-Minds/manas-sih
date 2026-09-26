"""
tests/test_app_exported_parity.py
---------------------------------
Verifies numerical parity and execution of exported ONNX and TFLite models
against PyTorch eager baseline and through EngineAdapterStageB.
"""

import os
import pytest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX_PATH = os.path.join(ROOT_DIR, "models", "exported", "moe_velocity_model.onnx")
TFLITE_PATH = os.path.join(ROOT_DIR, "models", "exported", "moe_velocity_model.tflite")
NORM_PATH = os.path.join(ROOT_DIR, "models", "exported", "normalization_params.npz")


def test_exported_artifacts_exist():
    assert os.path.exists(ONNX_PATH), f"Missing exported ONNX: {ONNX_PATH}"
    assert os.path.exists(TFLITE_PATH), f"Missing exported TFLite: {TFLITE_PATH}"
    assert os.path.exists(NORM_PATH), f"Missing normalization params: {NORM_PATH}"


def test_onnx_vs_torch_parity():
    import torch
    from sih.models.inference import load_ai_model
    from sih.models.predictor import ONNXVelocityPredictor, TorchVelocityPredictor

    device = torch.device("cpu")
    model, _, _, _ = load_ai_model(device, root_dir=ROOT_DIR)

    p_torch = TorchVelocityPredictor(model, device=device)
    p_onnx = ONNXVelocityPredictor(ONNX_PATH, intra_op_threads=1)

    np.random.seed(42)
    x_s = np.random.randn(20, 12, 20).astype(np.float32)
    x_l = np.random.randn(20, 12, 60).astype(np.float32)

    v_pt = p_torch.predict_batch(x_s, x_l)
    v_ox = p_onnx.predict_batch(x_s, x_l)

    max_diff = float(np.max(np.abs(v_pt - v_ox)))
    assert max_diff < 1.0e-5, f"ONNX vs Torch batch diff {max_diff:.3e} exceeds 1e-5 m/s"


def test_tflite_vs_torch_parity():
    import torch
    from sih.models.inference import load_ai_model
    from sih.models.predictor import TFLiteVelocityPredictor, TorchVelocityPredictor

    device = torch.device("cpu")
    model, _, _, _ = load_ai_model(device, root_dir=ROOT_DIR)

    p_torch = TorchVelocityPredictor(model, device=device)
    p_tf = TFLiteVelocityPredictor(TFLITE_PATH, num_threads=1)

    np.random.seed(42)
    x_s = np.random.randn(20, 12, 20).astype(np.float32)
    x_l = np.random.randn(20, 12, 60).astype(np.float32)

    v_pt = p_torch.predict_batch(x_s, x_l)
    v_tf = p_tf.predict_batch(x_s, x_l)

    max_diff = float(np.max(np.abs(v_pt - v_tf)))
    assert max_diff < 1.0e-5, f"TFLite vs Torch batch diff {max_diff:.3e} exceeds 1e-5 m/s"


def test_engine_adapter_with_exported_predictors():
    from server.engine_adapter import EngineAdapterStageB
    from sih.core.contracts import IMUSample

    # Test ONNX predictor initialization & streaming
    adapter_onnx = EngineAdapterStageB(predictor="onnx")
    assert adapter_onnx.predictor is not None

    # Test TFLite predictor initialization & streaming
    adapter_tf = EngineAdapterStageB(predictor="tflite")
    assert adapter_tf.predictor is not None

    # Feed 15 IMU samples into both
    for i in range(15):
        ts = int((1000.0 + i * 0.1) * 1e9)
        imu = IMUSample(
            timestamp_ns=ts,
            accel=np.array([0.1, 0.0, 9.81], dtype=np.float64),
            gyro=np.array([0.0, 0.0, 0.01], dtype=np.float64),
        )
        adapter_onnx.on_imu(imu)
        adapter_tf.on_imu(imu)

    assert len(adapter_onnx.recent_ai_speeds) == 15
    assert len(adapter_tf.recent_ai_speeds) == 15
    # Speeds should match closely between ONNX and TFLite
    speed_diff = abs(adapter_onnx.recent_ai_speeds[-1] - adapter_tf.recent_ai_speeds[-1])
    assert speed_diff < 1.0e-4, f"Speed difference {speed_diff} between ONNX and TFLite"
