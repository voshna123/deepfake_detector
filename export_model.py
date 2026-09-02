"""
Export a trained checkpoint to ONNX + INT8 quantized formats.

Usage:
    python scripts/export_model.py --cfg configs/config.yaml --ckpt checkpoints/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export")


def main():
    parser = argparse.ArgumentParser(description="Export DeepfakeDetector to ONNX / INT8")
    parser.add_argument("--cfg",        default="configs/config.yaml")
    parser.add_argument("--ckpt",       required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--export_dir", default="exports",   help="Output directory")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Load model
    from src.models.detector import build_model, model_size_mb

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cpu()
    logger.info("Float32 model size: %.2f MB", model_size_mb(model))

    # Run export pipeline
    from src.export.quantize import full_export_pipeline

    outputs = full_export_pipeline(model, cfg, export_dir=args.export_dir)
    logger.info("Exported files:")
    for name, path in outputs.items():
        size_mb = Path(path).stat().st_size / (1024 ** 2)
        logger.info("  %-22s → %s  (%.2f MB)", name, path, size_mb)


if __name__ == "__main__":
    main()
