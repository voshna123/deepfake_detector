"""
Evaluation metrics and cross-dataset generalisation testing.

Key metrics:
  - Accuracy, Precision, Recall, F1
  - AUC-ROC
  - Equal Error Rate (EER)
  - Average Precision (AP)

All functions accept raw numpy arrays and are framework-agnostic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER).

    EER is the point where FAR == FRR.  Returned as a fraction in [0, 1].
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the index where |FPR - FNR| is minimised
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        labels    : Ground-truth binary labels (0 = real, 1 = fake).
        probs     : Predicted probability of being fake (in [0, 1]).
        threshold : Decision boundary for hard classification.

    Returns:
        Dictionary of metric names to float values.
    """
    preds = (probs >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    metrics["acc"] = float(accuracy_score(labels, preds))
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0))
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))

    if len(np.unique(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, probs))
        metrics["ap"] = float(average_precision_score(labels, probs))
        metrics["eer"] = compute_eer(labels, probs)
    else:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
        metrics["eer"] = float("nan")

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation") -> None:
    """Pretty-print a metrics dictionary."""
    border = "─" * 40
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(border)


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> None:
    """Plot and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cross_dataset(
    model,
    loaders: Dict[str, "torch.utils.data.DataLoader"],
    device: "torch.device",
    threshold: float = 0.5,
    use_amp: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate `model` on multiple held-out datasets.

    Args:
        model   : Trained DeepfakeDetector in eval mode.
        loaders : {dataset_name: DataLoader} mapping.
        device  : Inference device.
        threshold: Decision threshold.
        use_amp : Whether to use autocast during inference.

    Returns:
        {dataset_name: metrics_dict}
    """
    import torch
    from torch.amp import autocast

    results: Dict[str, Dict[str, float]] = {}
    model.eval()

    for ds_name, loader in loaders.items():
        all_probs: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                with autocast("cuda", enabled=use_amp):
                    logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_probs),
            threshold=threshold,
        )
        results[ds_name] = metrics
        logger.info("[%s] AUC=%.4f  Acc=%.4f  F1=%.4f",
                    ds_name, metrics.get("auc", float("nan")),
                    metrics["acc"], metrics["f1"])

    return results
