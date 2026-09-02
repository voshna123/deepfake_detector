"""
Main training script for the deepfake detector.

Usage:
    # Standard training
    python scripts/train.py --cfg configs/config.yaml

    # Resume from checkpoint
    python scripts/train.py --cfg configs/config.yaml --resume checkpoints/last.pt

    # Hyperparameter search (Optuna)
    python scripts/train.py --cfg configs/config.yaml --search
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
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train DeepfakeDetector")
    parser.add_argument("--cfg",    default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--search", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--device", default=None,
                        help="Device override: 'cpu', 'cuda', 'cuda:0', …")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        cfg["training"]["use_amp"] = False   # AMP not supported on CPU
    logger.info("Device: %s", device)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    from src.data.augmentation import build_train_transforms, build_val_transforms
    from src.data.dataset import build_datasets, build_weighted_sampler

    train_tf = build_train_transforms(cfg)
    val_tf   = build_val_transforms(cfg)

    logger.info("Building training dataset …")
    train_ds = build_datasets(cfg, split="train", train_transform=train_tf, val_transform=val_tf)
    logger.info("Training samples: %d", len(train_ds))

    logger.info("Building validation dataset …")
    val_ds = build_datasets(cfg, split="val", train_transform=train_tf, val_transform=val_tf)
    logger.info("Validation samples: %d", len(val_ds))

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"]["num_workers"]

    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    # ── Hyperparameter search ─────────────────────────────────────────────────
    if args.search:
        from src.training.trainer import run_hyperparameter_search
        study = run_hyperparameter_search(cfg, train_loader, val_loader, device)
        logger.info("Best params: %s", study.best_params)
        # Apply best params to cfg and continue with normal training
        cfg["training"]["learning_rate"] = study.best_params["lr"]
        cfg["model"]["backbone"] = study.best_params.get("backbone", cfg["model"]["backbone"])
        cfg["model"]["gru_hidden_size"] = study.best_params.get("gru_hidden_size", 256)

    # ── Model ────────────────────────────────────────────────────────────────
    from src.models.detector import build_model, count_parameters, model_size_mb

    model = build_model(cfg)
    logger.info("Parameters: %s  |  Size: %.1f MB",
                f"{count_parameters(model):,}", model_size_mb(model))

    # ── Trainer ──────────────────────────────────────────────────────────────
    from src.training.trainer import Trainer

    trainer = Trainer(model, train_loader, val_loader, cfg, device)

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info("Resuming from epoch %d", start_epoch)

    best = trainer.train()
    logger.info("Training finished. Best metrics: %s", best)


if __name__ == "__main__":
    main()
