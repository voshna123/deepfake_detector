"""
Post-training INT8 quantization.

Two approaches are provided:
  1. PyTorch dynamic quantization  – quantises linear/GRU layers at export time;
     no calibration data needed.  Produces a serialised TorchScript module.

  2. ONNX INT8 quantization via onnxruntime-tools  – converts a float32 ONNX
     model to INT8 using dynamic or static quantization.

Both approaches target a final model size < 15 MB.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch dynamic INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_pytorch_dynamic(
    model: nn.Module,
    output_path: str,
) -> str:
    """
    Apply PyTorch dynamic INT8 quantization and save as TorchScript.

    Dynamic quantization quantises weights to INT8 at load time and
    computes activations in floating point – no calibration data is needed.

    Args:
        model       : Float32 nn.Module (should be in eval mode on CPU).
        output_path : Destination path for the quantised TorchScript file (.pt).

    Returns:
        Path to the saved quantised model.
    """
    model = model.cpu().eval()

    quantised = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )

    # Trace to TorchScript for deployment
    # Use a dummy sequence input
    dummy = torch.randn(1, 8, 3, 224, 224)
    try:
        scripted = torch.jit.trace(quantised, dummy, strict=False)
    except Exception:
        scripted = torch.jit.script(quantised)

    output_path = str(Path(output_path).with_suffix(".pt"))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scripted.save(output_path)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    logger.info("Quantised TorchScript saved → %s  (%.2f MB)", output_path, size_mb)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX INT8 quantization (onnxruntime)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_onnx_dynamic(
    onnx_model_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Apply dynamic INT8 quantization to an ONNX model using onnxruntime.

    Args:
        onnx_model_path : Path to the float32 .onnx file.
        output_path     : Destination path for quantised .onnx file.
                          Defaults to `<stem>_int8.onnx` in the same directory.

    Returns:
        Path to the quantised ONNX file.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error(
            "onnxruntime.quantization not available. "
            "Install with: pip install onnxruntime-tools"
        )
        raise

    src = Path(onnx_model_path)
    if output_path is None:
        output_path = str(src.parent / f"{src.stem}_int8.onnx")

    logger.info("Quantising ONNX model (dynamic INT8): %s → %s", src, output_path)
    quantize_dynamic(
        model_input=str(src),
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    orig_mb = src.stat().st_size / (1024 ** 2)
    quant_mb = Path(output_path).stat().st_size / (1024 ** 2)
    logger.info(
        "Original: %.2f MB → Quantised: %.2f MB  (reduction: %.1f%%)",
        orig_mb, quant_mb, 100 * (1 - quant_mb / orig_mb),
    )
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  Unified export + quantize pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_export_pipeline(
    model: nn.Module,
    cfg: dict,
    export_dir: str = "exports",
) -> dict:
    """
    Run the complete export pipeline:
      1. Export to float32 ONNX
      2. INT8-quantise the ONNX model
      3. Apply PyTorch dynamic quantization → TorchScript

    Returns a dict of output paths.
    """
    from .onnx_export import export_to_onnx

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model.eval().cpu()
    outputs = {}

    # Step 1: ONNX float32
    onnx_path = str(export_dir / "detector.onnx")
    onnx_path = export_to_onnx(model, cfg, onnx_path, device=torch.device("cpu"))
    outputs["onnx_fp32"] = onnx_path

    # Step 2: ONNX INT8
    try:
        int8_onnx_path = quantize_onnx_dynamic(onnx_path)
        outputs["onnx_int8"] = int8_onnx_path
    except Exception as exc:
        logger.warning("ONNX INT8 quantization failed: %s", exc)

    # Step 3: PyTorch INT8 TorchScript
    try:
        ts_path = str(export_dir / "detector_int8.pt")
        ts_path = quantize_pytorch_dynamic(model, ts_path)
        outputs["torchscript_int8"] = ts_path
    except Exception as exc:
        logger.warning("PyTorch quantization failed: %s", exc)

    logger.info("Export pipeline complete: %s", outputs)
    return outputs
