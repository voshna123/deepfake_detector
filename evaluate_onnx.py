"""
Evaluate exported ONNX models (FP32 and INT8) on the test dataset.

Usage:
    python scripts/evaluate_onnx.py --cfg configs/config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import yaml
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_onnx")


def run_onnx_inference(session, loader):
    all_probs, all_labels = [], []
    input_name = session.get_inputs()[0].name

    for inputs, labels in loader:
        x = inputs.numpy()
        logits = session.run(None, {input_name: x})[0]
        # softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.extend(probs[:, 1].tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/config.yaml")
    parser.add_argument("--fp32", default="exports/detector.onnx")
    parser.add_argument("--int8", default="exports/detector_int8.onnx")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    from src.data.augmentation import build_val_transforms
    from src.data.dataset import build_datasets
    from src.evaluation.metrics import compute_metrics

    val_tf = build_val_transforms(cfg)
    test_ds = build_datasets(cfg, split="test", val_transform=val_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
    )
    logger.info("Test samples: %d", len(test_ds))

    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

    models = {"FP32": args.fp32, "INT8": args.int8}
    results = {}

    for name, path in models.items():
        if not Path(path).exists():
            logger.warning("%s not found: %s", name, path)
            continue

        logger.info("Running %s inference: %s", name, path)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        session = ort.InferenceSession(path, sess_opts, providers=["CPUExecutionProvider"])

        t0 = time.perf_counter()
        labels, probs = run_onnx_inference(session, test_loader)
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(labels, probs, threshold)
        metrics["inference_time_s"] = round(elapsed, 2)
        metrics["ms_per_sample"] = round(elapsed / len(labels) * 1000, 3)
        results[name] = metrics

        size_mb = Path(path).stat().st_size / (1024 ** 2)
        logger.info(
            "%s | acc=%.4f auc=%.4f f1=%.4f | size=%.2f MB | %.1fs (%.3f ms/sample)",
            name, metrics["acc"], metrics["auc"], metrics["f1"],
            size_mb, elapsed, metrics["ms_per_sample"],
        )

    # Side-by-side summary
    print("\n" + "=" * 62)
    print(f"  {'Metric':<20} {'FP32':>18} {'INT8':>18}")
    print("=" * 62)
    for key in ["acc", "precision", "recall", "f1", "auc", "ap", "eer",
                "inference_time_s", "ms_per_sample"]:
        fp32_val = results.get("FP32", {}).get(key, "-")
        int8_val = results.get("INT8", {}).get(key, "-")
        fp32_str = f"{fp32_val:.4f}" if isinstance(fp32_val, float) else str(fp32_val)
        int8_str = f"{int8_val:.4f}" if isinstance(int8_val, float) else str(int8_val)
        print(f"  {key:<20} {fp32_str:>18} {int8_str:>18}")
    print("=" * 62)


if __name__ == "__main__":
    main()
