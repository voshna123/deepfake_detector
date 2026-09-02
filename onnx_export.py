"""
ONNX export for the DeepfakeDetector model.

Exports the model to ONNX format with dynamic batch and sequence axes,
validates the export with onnxruntime, and reports the model file size.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    cfg: Dict[str, Any],
    output_path: str,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Export `model` to ONNX.

    Args:
        model       : Trained DeepfakeDetector (should be in eval mode).
        cfg         : Config dict (reads export / dataset sections).
        output_path : Destination .onnx file path.
        device      : Device used during tracing.

    Returns:
        Absolute path to the written .onnx file.
    """
    import onnx

    export_cfg = cfg.get("export", {})
    opset = export_cfg.get("opset_version", 17)
    input_shape: Tuple = tuple(export_cfg.get("onnx_input_shape", [1, 8, 3, 224, 224]))
    use_seq = cfg.get("dataset", {}).get("use_sequences", True)

    model = model.to(device).eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Dynamic axes: batch dimension is always dynamic;
    # sequence length is also dynamic when using temporal sequences.
    if use_seq:
        dynamic_axes = {
            "input":  {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"},
        }
    else:
        dynamic_axes = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    output_path = str(Path(output_path).with_suffix(".onnx"))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logger.info("Exporting model to ONNX (opset=%d) → %s", opset, output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
        )

    # Validate the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model check passed.")

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    limit_mb = export_cfg.get("size_limit_mb", 15)
    logger.info("ONNX model size: %.2f MB  (limit: %.0f MB)", size_mb, limit_mb)
    if size_mb > limit_mb:
        logger.warning(
            "Model size %.2f MB exceeds limit %.0f MB. "
            "Consider running INT8 quantization.",
            size_mb, limit_mb,
        )

    # Quick inference sanity check with onnxruntime
    _verify_onnx(output_path, dummy_input.cpu().numpy())

    return output_path


def _verify_onnx(onnx_path: str, dummy_np: np.ndarray) -> None:
    """Run a single inference pass with onnxruntime to verify the export."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy_np})
        logger.info(
            "ONNX runtime inference OK – output shape: %s",
            [o.shape for o in out],
        )
    except Exception as exc:
        logger.error("ONNX runtime verification failed: %s", exc)
