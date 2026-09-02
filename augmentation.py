"""
Albumentations-based augmentation pipelines for deepfake detection.

Two transforms are provided:
  - train_transforms : heavy augmentation for training
  - val_transforms   : normalisation only (no augmentation)
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalisation statistics (used for pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_train_transforms(cfg: Dict[str, Any]) -> A.Compose:
    """
    Heavy augmentation pipeline for training.

    All probabilities and magnitudes are read from `cfg['augmentation']`.
    Falls back to sensible defaults if the key is missing.
    """
    aug = cfg.get("augmentation", {})
    image_size = cfg.get("preprocessing", {}).get("image_size", 224)

    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=aug.get("horizontal_flip_prob", 0.5)),
        A.Rotate(limit=aug.get("rotation_degrees", 10), p=0.5),
        A.ColorJitter(
            brightness=aug.get("brightness", 0.2),
            contrast=aug.get("contrast", 0.2),
            saturation=aug.get("saturation", 0.2),
            hue=aug.get("hue", 0.05),
            p=0.5,
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=aug.get("gaussian_blur_prob", 0.2),
        ),
        A.GaussNoise(
            p=aug.get("gaussian_noise_prob", 0.1),
        ),
        A.ImageCompression(
            quality_range=(
                aug.get("jpeg_quality_range", [50, 95])[0],
                aug.get("jpeg_quality_range", [50, 95])[1],
            ),
            p=aug.get("jpeg_compression_prob", 0.3),
        ),
        A.CoarseDropout(
            num_holes_range=(1, aug.get("cutout_holes", 1)),
            hole_height_range=(
                int(image_size * aug.get("cutout_size", 0.15)),
                int(image_size * aug.get("cutout_size", 0.15)),
            ),
            hole_width_range=(
                int(image_size * aug.get("cutout_size", 0.15)),
                int(image_size * aug.get("cutout_size", 0.15)),
            ),
            fill=0,
            p=aug.get("cutout_prob", 0.2),
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def build_val_transforms(cfg: Dict[str, Any]) -> A.Compose:
    """Minimal pipeline: resize + normalise only."""
    image_size = cfg.get("preprocessing", {}).get("image_size", 224)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_inference_transforms(image_size: int = 224) -> A.Compose:
    """Single-image inference pipeline (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def apply_transform(transform: A.Compose, image: np.ndarray) -> "torch.Tensor":
    """Apply an albumentations transform to a HxWx3 uint8 numpy array."""
    result = transform(image=image)
    return result["image"]
