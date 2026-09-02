"""
Evaluation script – standard + cross-dataset.

Usage:
    # Evaluate on the test split
    python scripts/evaluate.py --cfg configs/config.yaml --ckpt checkpoints/best.pt

    # Cross-dataset evaluation (e.g. trained on FF++, tested on CelebDF)
    python scripts/evaluate.py --cfg configs/config.yaml --ckpt checkpoints/best.pt --cross
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepfakeDetector")
    parser.add_argument("--cfg",   default="configs/config.yaml")
    parser.add_argument("--ckpt",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--cross", action="store_true",
                        help="Run cross-dataset generalisation test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_plots", default="exports",
                        help="Directory to save ROC and CM plots")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = cfg["training"].get("use_amp", True) and device.type == "cuda"

    # ── Load model ────────────────────────────────────────────────────────────
    from src.models.detector import build_model

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    logger.info("Loaded checkpoint: %s  (epoch %d)", args.ckpt, ckpt.get("epoch", -1))

    # ── Test dataset ──────────────────────────────────────────────────────────
    from src.data.augmentation import build_val_transforms
    from src.data.dataset import build_datasets

    val_tf = build_val_transforms(cfg)
    test_ds = build_datasets(cfg, split="test", val_transform=val_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
    )

    # ── Standard evaluation ───────────────────────────────────────────────────
    import numpy as np
    from src.evaluation.metrics import (
        compute_metrics, print_metrics,
        plot_roc_curve, plot_confusion_matrix,
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs), threshold)
    print_metrics(metrics, title="Test Set Evaluation")

    Path(args.save_plots).mkdir(parents=True, exist_ok=True)
    plot_roc_curve(np.array(all_labels), np.array(all_probs),
                   save_path=f"{args.save_plots}/roc_test.png",
                   title="ROC – Test Set")
    preds = (np.array(all_probs) >= threshold).astype(int)
    plot_confusion_matrix(np.array(all_labels), preds,
                          save_path=f"{args.save_plots}/cm_test.png",
                          title="Confusion Matrix – Test Set")

    # ── Cross-dataset ──────────────────────────────────────────────────────────
    if args.cross:
        from src.data.dataset import CelebDFDataset, SequenceDataset
        from src.evaluation.metrics import evaluate_cross_dataset

        cross_loaders = {}
        cross_sources = cfg.get("evaluation", {}).get("cross_dataset_sources", ["celeb"])

        if "celeb" in cross_sources:
            celeb_ds = CelebDFDataset(
                root=cfg["paths"]["celeb_root"], split="test", transform=val_tf
            )
            use_seq = cfg["dataset"]["use_sequences"]
            if use_seq:
                celeb_ds = SequenceDataset(celeb_ds, seq_len=cfg["dataset"]["sequence_length"])
            cross_loaders["CelebDF"] = DataLoader(
                celeb_ds, batch_size=cfg["training"]["batch_size"],
                shuffle=False, num_workers=cfg["dataset"]["num_workers"],
            )

        if cross_loaders:
            results = evaluate_cross_dataset(model, cross_loaders, device, threshold, use_amp)
            for ds_name, m in results.items():
                print_metrics(m, title=f"Cross-dataset: {ds_name}")


if __name__ == "__main__":
    main()
