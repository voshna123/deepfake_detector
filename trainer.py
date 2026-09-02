"""
Training loop for the deepfake detector.

Features:
  - Mixed-precision (AMP) via torch.cuda.amp
  - Class-weighted cross-entropy loss
  - AdamW optimizer with cosine / step / plateau scheduler
  - Warm-up epochs
  - Gradient clipping
  - Early stopping
  - TensorBoard logging
  - Optuna hyperparameter search integration
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: list, num_classes: int = 2) -> torch.Tensor:
    """
    Inverse-frequency class weights for cross-entropy.

    Returns a tensor of shape (num_classes,) on CPU.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def build_criterion(
    cfg: Dict[str, Any],
    train_labels: Optional[list] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Build cross-entropy loss, optionally with class weights."""
    use_weights = cfg.get("training", {}).get("use_class_weights", True)
    num_classes = cfg.get("model", {}).get("num_classes", 2)

    weight = None
    if use_weights and train_labels is not None:
        weight = compute_class_weights(train_labels, num_classes).to(device)
        logger.info("Class weights: %s", weight.tolist())

    return nn.CrossEntropyLoss(weight=weight)


# ─────────────────────────────────────────────────────────────────────────────
#  Optimizer & scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    lr = train_cfg.get("learning_rate", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-4)
    opt_name = train_cfg.get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> Optional[object]:
    train_cfg = cfg.get("training", {})
    sched_name = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg.get("epochs", 50)
    warmup = train_cfg.get("warmup_epochs", 3)

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=1e-6
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.1
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

    if warmup > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup],
        )

    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
#  Epoch runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    grad_clip: float,
    is_train: bool,
) -> Tuple[float, float]:
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_amp = scaler is not None
            with autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()  # type: ignore[union-attr]
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
#  Main Trainer class
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle.

    Args:
        model         : DeepfakeDetector (or any nn.Module).
        train_loader  : DataLoader for training split.
        val_loader    : DataLoader for validation split.
        cfg           : Full config dict.
        device        : torch.device to train on.
        trial         : Optional Optuna trial for pruning / reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: torch.device,
        trial=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.trial = trial

        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.val_interval = train_cfg.get("val_interval", 1)
        self.patience = train_cfg.get("early_stopping_patience", 10)
        self.best_metric_name = train_cfg.get("best_metric", "val_auc")
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"

        # Build criterion with class weights from training labels
        try:
            train_labels = train_loader.dataset.labels  # type: ignore[attr-defined]
        except AttributeError:
            train_labels = None
        self.criterion = build_criterion(cfg, train_labels, device)

        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch=len(train_loader)
        )
        self.scaler: Optional[GradScaler] = GradScaler("cuda") if self.use_amp else None

        # Checkpointing
        ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = ckpt_dir / "best.pt"
        self.last_ckpt_path = ckpt_dir / "last.pt"

        # TensorBoard
        log_dir = cfg.get("paths", {}).get("logs", "logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        # State
        self.best_val_metric = 0.0
        self.epochs_no_improve = 0
        self.global_step = 0

    # ── Evaluation ──────────────────────────────────────────────────────────

    def _evaluate(self) -> Dict[str, float]:
        """Run validation and compute AUC in addition to accuracy."""
        from src.evaluation.metrics import compute_metrics

        self.model.eval()
        all_probs: list = []
        all_labels: list = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels_np = labels.numpy()
                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(inputs)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels_np.tolist())

        return compute_metrics(
            np.array(all_labels), np.array(all_probs),
            threshold=self.cfg.get("evaluation", {}).get("threshold", 0.5),
        )

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "cfg": self.cfg,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
        return ckpt.get("epoch", 0)

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns the best validation metrics dict.
        """
        logger.info("Starting training for %d epochs on %s", self.epochs, self.device)
        best_metrics: Dict[str, float] = {}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.scaler, self.device,
                self.grad_clip, is_train=True,
            )
            elapsed = time.time() - t0

            # Log train metrics
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Validation
            val_metrics: Dict[str, float] = {}
            if epoch % self.val_interval == 0:
                val_metrics = self._evaluate()
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"Val/{k}", v, epoch)

                val_metric = val_metrics.get(
                    self.best_metric_name.replace("val_", ""),
                    val_metrics.get("acc", 0.0),
                )

                logger.info(
                    "Epoch %3d/%d | loss=%.4f acc=%.4f | val_acc=%.4f val_auc=%.4f | %.1fs",
                    epoch, self.epochs, train_loss, train_acc,
                    val_metrics.get("acc", 0.0), val_metrics.get("auc", 0.0), elapsed,
                )

                # Save best
                if val_metric > self.best_val_metric:
                    self.best_val_metric = val_metric
                    self.epochs_no_improve = 0
                    best_metrics = val_metrics
                    self._save_checkpoint(self.best_ckpt_path, epoch, val_metrics)
                    logger.info("  ↑ New best checkpoint saved  (%s=%.4f)", self.best_metric_name, val_metric)
                else:
                    self.epochs_no_improve += 1

                # Optuna reporting and pruning
                if self.trial is not None:
                    self.trial.report(val_metric, epoch)
                    import optuna
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                # Early stopping
                if self.epochs_no_improve >= self.patience:
                    logger.info("Early stopping after %d epochs without improvement.", self.patience)
                    break
            else:
                logger.info(
                    "Epoch %3d/%d | loss=%.4f acc=%.4f | %.1fs",
                    epoch, self.epochs, train_loss, train_acc, elapsed,
                )

            # Step scheduler (ReduceLROnPlateau needs the metric)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(self.best_val_metric)
                else:
                    self.scheduler.step()

        self._save_checkpoint(self.last_ckpt_path, epoch, val_metrics or {})
        self.writer.close()
        logger.info("Training complete. Best %s = %.4f", self.best_metric_name, self.best_val_metric)
        return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Optuna hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_search(base_cfg: Dict[str, Any], train_loader, val_loader, device):
    """
    Use Optuna to search over learning rate, batch size, GRU hidden size,
    and backbone choice.

    Returns the best Optuna study object.
    """
    import optuna
    from copy import deepcopy
    from src.models.detector import build_model

    hs_cfg = base_cfg.get("hyperparameter_search", {})
    n_trials = hs_cfg.get("n_trials", 30)
    timeout = hs_cfg.get("timeout_seconds", 7200)

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)
        lr_low, lr_high = hs_cfg.get("lr_range", [1e-5, 1e-3])
        cfg["training"]["learning_rate"] = trial.suggest_float("lr", lr_low, lr_high, log=True)
        cfg["model"]["backbone"] = trial.suggest_categorical(
            "backbone", hs_cfg.get("backbone_choices", ["mobilenet_v3_small"])
        )
        cfg["model"]["gru_hidden_size"] = trial.suggest_categorical(
            "gru_hidden_size", hs_cfg.get("gru_hidden_sizes", [128, 256])
        )
        # Trim epochs for speed during search
        cfg["training"]["epochs"] = min(cfg["training"]["epochs"], 15)
        cfg["training"]["early_stopping_patience"] = 5

        model = build_model(cfg)
        trainer = Trainer(model, train_loader, val_loader, cfg, device, trial=trial)
        best = trainer.train()
        return best.get("auc", best.get("acc", 0.0))

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Best trial: %s", study.best_trial.params)
    return study
