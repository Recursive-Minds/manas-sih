"""
sih/models/predictor.py
-----------------------
Modular runtime predictor interface for velocity inference.
Decouples speed estimation runtime (PyTorch CPU/CUDA, ONNX Runtime, LiteRT/TFLite)
from the streaming dead reckoning engine adapter and offline evaluation harnesses.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Any
import numpy as np


class VelocityPredictor(ABC):
    """Abstract interface for causal velocity inference on normalized feature windows."""

    @abstractmethod
    def predict_window(self, x_short: np.ndarray, x_long: np.ndarray) -> Tuple[float, float]:
        """
        Runs causal inference on a single feature window.

        Args:
            x_short: ndarray of shape (1, 12, 20) or (12, 20), normalized short window.
            x_long: ndarray of shape (1, 12, 60) or (12, 60), normalized long window.

        Returns:
            Tuple of (v_fused_mps, var_fused).
        """
        pass

    @abstractmethod
    def predict_batch(self, x_short: np.ndarray, x_long: np.ndarray) -> np.ndarray:
        """
        Runs inference across a batch of feature windows.

        Args:
            x_short: ndarray of shape (N, 12, 20), normalized short windows.
            x_long: ndarray of shape (N, 12, 60), normalized long windows.

        Returns:
            1D ndarray of length N containing predicted speeds (m/s).
        """
        pass


class TorchVelocityPredictor(VelocityPredictor):
    """PyTorch execution backend (CPU or CUDA)."""

    def __init__(self, model: Any, device: Optional[Any] = None) -> None:
        import torch
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device) if hasattr(model, "to") else model
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict_window(self, x_short: np.ndarray, x_long: np.ndarray) -> Tuple[float, float]:
        import torch
        if x_short.ndim == 2:
            x_short = x_short[None, ...]
        if x_long.ndim == 2:
            x_long = x_long[None, ...]
        ts_s = torch.from_numpy(x_short).to(self.device)
        ts_l = torch.from_numpy(x_long).to(self.device)
        with torch.no_grad():
            vf, var_f, _ = self.model(ts_s, ts_l)
        return float(vf.item()), float(var_f.item())

    def predict_batch(self, x_short: np.ndarray, x_long: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        import torch
        N = len(x_short)
        preds = []
        with torch.no_grad():
            for b in range(0, N, batch_size):
                b_s = torch.from_numpy(x_short[b : b + batch_size]).to(self.device)
                b_l = torch.from_numpy(x_long[b : b + batch_size]).to(self.device)
                vf, _, _ = self.model(b_s, b_l)
                preds.extend(vf.squeeze(-1).float().cpu().numpy().flatten())
        return np.array(preds, dtype=np.float32)


class ONNXVelocityPredictor(VelocityPredictor):
    """ONNX Runtime execution backend (optimized single-thread CPU)."""

    def __init__(self, model_path_or_session: Union[str, Any], intra_op_threads: int = 1) -> None:
        import onnxruntime as ort
        if isinstance(model_path_or_session, str):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = intra_op_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path_or_session, opts)
        else:
            self.session = model_path_or_session

    def predict_window(self, x_short: np.ndarray, x_long: np.ndarray) -> Tuple[float, float]:
        if x_short.ndim == 2:
            x_short = x_short[None, ...]
        if x_long.ndim == 2:
            x_long = x_long[None, ...]
        outputs = self.session.run(None, {"x_short": x_short, "x_long": x_long})
        return float(outputs[0][0, 0]), float(outputs[1][0, 0])

    def predict_batch(self, x_short: np.ndarray, x_long: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        N = len(x_short)
        preds = []
        for b in range(0, N, batch_size):
            b_s = x_short[b : b + batch_size]
            b_l = x_long[b : b + batch_size]
            outs = self.session.run(None, {"x_short": b_s, "x_long": b_l})
            preds.extend(outs[0].flatten())
        return np.array(preds, dtype=np.float32)


class TFLiteVelocityPredictor(VelocityPredictor):
    """LiteRT / TFLite execution backend (optimized mobile inference)."""

    def __init__(self, model_path_or_interpreter: Union[str, Any], num_threads: int = 1) -> None:
        try:
            import ai_edge_litert.interpreter as litert
            Interpreter = litert.Interpreter
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                Interpreter = tflite.Interpreter
            except ImportError:
                import tensorflow.lite as tflite
                Interpreter = tflite.Interpreter

        if isinstance(model_path_or_interpreter, str):
            self.interpreter = Interpreter(model_path=model_path_or_interpreter, num_threads=num_threads)
        else:
            self.interpreter = model_path_or_interpreter

        self.interpreter.allocate_tensors()
        self.in_short_idx = self.interpreter.get_input_details()[0]["index"]
        self.in_long_idx = self.interpreter.get_input_details()[1]["index"]
        self.out_v_idx = self.interpreter.get_output_details()[0]["index"]
        self.out_var_idx = self.interpreter.get_output_details()[1]["index"]

    def predict_window(self, x_short: np.ndarray, x_long: np.ndarray) -> Tuple[float, float]:
        if x_short.ndim == 2:
            x_short = x_short[None, ...]
        if x_long.ndim == 2:
            x_long = x_long[None, ...]
        self.interpreter.set_tensor(self.in_short_idx, x_short)
        self.interpreter.set_tensor(self.in_long_idx, x_long)
        self.interpreter.invoke()
        v = self.interpreter.get_tensor(self.out_v_idx)
        var = self.interpreter.get_tensor(self.out_var_idx)
        return float(v[0, 0]), float(var[0, 0])

    def predict_batch(self, x_short: np.ndarray, x_long: np.ndarray) -> np.ndarray:
        N = len(x_short)
        preds = np.empty(N, dtype=np.float32)
        for i in range(N):
            self.interpreter.set_tensor(self.in_short_idx, x_short[i : i + 1])
            self.interpreter.set_tensor(self.in_long_idx, x_long[i : i + 1])
            self.interpreter.invoke()
            preds[i] = self.interpreter.get_tensor(self.out_v_idx)[0, 0]
        return preds


def create_predictor(
    kind: str = "torch",
    model: Optional[Any] = None,
    device: Optional[Any] = None,
    onnx_path: Optional[str] = None,
    tflite_path: Optional[str] = None,
    root_dir: Optional[str] = None,
) -> VelocityPredictor:
    """Factory creating appropriate predictor by type name."""
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    kind = kind.lower().strip()
    if kind == "torch":
        if model is None:
            from sih.models.inference import load_ai_model
            import torch
            dev = device or torch.device("cpu")
            model, _, _, _ = load_ai_model(dev, root_dir=root_dir)
        return TorchVelocityPredictor(model, device=device)

    elif kind == "onnx":
        target_path = onnx_path or os.path.join(root_dir, "models", "exported", "moe_velocity_model.onnx")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Exported ONNX model not found: {target_path}")
        return ONNXVelocityPredictor(target_path)

    elif kind in ("tflite", "litert"):
        target_path = tflite_path or os.path.join(root_dir, "models", "exported", "moe_velocity_model.tflite")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Exported TFLite model not found: {target_path}")
        return TFLiteVelocityPredictor(target_path)

    else:
        raise ValueError(f"Unknown predictor kind: {kind}. Expected 'torch', 'onnx', or 'tflite'.")
