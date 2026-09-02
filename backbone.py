"""
Lightweight CNN backbone with CBAM attention.

Supported backbones:
  - mobilenet_v3_small   (~2.5 M params, fastest)
  - mobilenet_v3_large   (~5.5 M params)
  - shufflenet_v2_x1_0   (~2.3 M params)

CBAM is inserted after the final feature extractor stage to let the
network emphasise spatially discriminative artefacts (e.g. blending
boundaries, texture inconsistencies) common in deepfakes.
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .cbam import CBAM


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 (small or large) with a CBAM attention head.

    The original classifier is stripped; a CBAM block is appended
    after the last convolutional feature map before adaptive average
    pooling.

    Output: flattened feature vector of size `out_features`.
    """

    def __init__(
        self,
        variant: str = "mobilenet_v3_small",
        pretrained: bool = True,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        weights = "DEFAULT" if pretrained else None

        if variant == "mobilenet_v3_small":
            base = tv_models.mobilenet_v3_small(weights=weights)
            # base.features outputs 576 channels (final Conv-BN-HSwish block)
            self._feature_channels = 576
            self.features = base.features          # outputs (B, 576, H, W)
        elif variant == "mobilenet_v3_large":
            base = tv_models.mobilenet_v3_large(weights=weights)
            self._feature_channels = 960
            self.features = base.features
        else:
            raise ValueError(f"Unknown MobileNetV3 variant: {variant}")

        self.cbam = CBAM(
            in_channels=self._feature_channels,
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # After pool: (B, channels, 1, 1) → flatten → (B, channels)
        self.out_features: int = self._feature_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)       # (B, C, H, W)
        feat = self.cbam(feat)        # CBAM attention
        feat = self.pool(feat)        # (B, C, 1, 1)
        return feat.flatten(1)        # (B, C)


class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2 x1.0 with CBAM inserted before the final global pool.

    Output: flattened feature vector of size `out_features`.
    """

    def __init__(
        self,
        pretrained: bool = True,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        base = tv_models.shufflenet_v2_x1_0(weights=weights)

        # ShuffleNetV2 x1.0: stage4 outputs 464 channels → conv5 (1024 ch) → pool
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5          # outputs (B, 1024, H, W)

        self._feature_channels = 1024
        self.cbam = CBAM(
            in_channels=self._feature_channels,
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features: int = self._feature_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x.flatten(1)


def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the backbone configured in `cfg['model']`."""
    model_cfg = cfg.get("model", {})
    backbone_name = model_cfg.get("backbone", "mobilenet_v3_small")
    pretrained = model_cfg.get("pretrained", True)
    reduction_ratio = model_cfg.get("cbam_reduction_ratio", 16)
    kernel_size = model_cfg.get("cbam_kernel_size", 7)

    if backbone_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        return MobileNetV3Backbone(
            variant=backbone_name,
            pretrained=pretrained,
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size,
        )
    elif backbone_name == "shufflenet_v2_x1_0":
        return ShuffleNetV2Backbone(
            pretrained=pretrained,
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
