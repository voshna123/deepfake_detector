"""
Full deepfake detection model.

Architecture:
    Input  →  Backbone (MobileNetV3 / ShuffleNetV2 + CBAM)
           →  GRU (temporal reasoning over frame sequences)
           →  Classifier head  →  logits (real / fake)

Single-frame mode:
    When `use_sequences=False` the GRU step is skipped and the backbone
    output is passed directly to the classifier.  This allows the same
    model class to handle both frame-level and sequence-level inference.
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn

from .backbone import build_backbone


class DeepfakeDetector(nn.Module):
    """
    Lightweight deepfake detector.

    Args:
        backbone        : Any nn.Module with an `out_features` attribute.
        hidden_size     : GRU hidden state size.
        num_layers      : Number of GRU layers.
        gru_dropout     : Dropout between GRU layers (only active if num_layers>1).
        classifier_dropout: Dropout before the final linear layer.
        num_classes     : 2 for binary (real/fake).
        use_sequences   : If True, expect input shape (B, T, C, H, W);
                          if False, expect (B, C, H, W).
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int = 256,
        num_layers: int = 2,
        gru_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
        num_classes: int = 2,
        use_sequences: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_sequences = use_sequences
        feat_dim = backbone.out_features  # type: ignore[attr-defined]

        if use_sequences:
            self.gru = nn.GRU(
                input_size=feat_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=gru_dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            classifier_in = hidden_size
        else:
            self.gru = None  # type: ignore[assignment]
            classifier_in = feat_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) if use_sequences else (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        if self.use_sequences:
            B, T, C, H, W = x.shape
            # Process all frames through backbone in one batch
            frames = x.view(B * T, C, H, W)
            feats = self.backbone(frames)           # (B*T, feat_dim)
            feats = feats.view(B, T, -1)            # (B, T, feat_dim)
            # GRU: take output of the last time step
            gru_out, _ = self.gru(feats)            # (B, T, hidden)
            last_hidden = gru_out[:, -1, :]         # (B, hidden)
            logits = self.classifier(last_hidden)
        else:
            feats = self.backbone(x)                # (B, feat_dim)
            logits = self.classifier(feats)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (B, num_classes)."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def build_model(cfg: Dict[str, Any]) -> DeepfakeDetector:
    """Build the full detector from a config dictionary."""
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})

    backbone = build_backbone(cfg)

    model = DeepfakeDetector(
        backbone=backbone,
        hidden_size=model_cfg.get("gru_hidden_size", 256),
        num_layers=model_cfg.get("gru_num_layers", 2),
        gru_dropout=model_cfg.get("gru_dropout", 0.3),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.5),
        num_classes=model_cfg.get("num_classes", 2),
        use_sequences=dataset_cfg.get("use_sequences", True),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (float32 weights only)."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)
