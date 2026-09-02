"""
PyTorch Dataset implementations for deepfake detection.

Three datasets are provided:
  - CelebDFDataset   : loads pre-extracted face images from the CelebDF folder tree
  - FF_FrameDataset  : loads face frames extracted from FaceForensics++ via preprocessing
  - SequenceDataset  : wraps either dataset and returns fixed-length temporal sequences
  - CombinedDataset  : merges CelebDF + FF++ with optional sampling ratios
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

import albumentations as A


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    """Load an image file and return as HxWx3 uint8 RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
#  CelebDF dataset
# ─────────────────────────────────────────────────────────────────────────────

class CelebDFDataset(Dataset):
    """
    Loads pre-extracted CelebDF face images.

    Expected directory layout:
        <root>/
          Train/fake/*.jpg
          Train/real/*.jpg
          Val/fake/*.jpg
          Val/real/*.jpg
          Test/fake/*.jpg
          Test/real/*.jpg
    """

    SPLIT_MAP = {"train": "Train", "val": "Val", "test": "Test"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_dir = self.root / self.SPLIT_MAP[split]
        fake_dir = split_dir / "fake"
        real_dir = split_dir / "real"

        fake_paths = sorted(fake_dir.glob("*.jpg")) + sorted(fake_dir.glob("*.png"))
        real_paths = sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png"))

        self.samples: List[Tuple[str, int]] = (
            [(str(p), 1) for p in fake_paths] +
            [(str(p), 0) for p in real_paths]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = _load_image(path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
#  FaceForensics++ frame dataset (post-extraction)
# ─────────────────────────────────────────────────────────────────────────────

class FF_FrameDataset(Dataset):
    """
    Loads extracted face frames for FaceForensics++.

    Reads a manifest CSV written by `preprocessing.preprocess_ff_dataset` with
    columns: [frame_path, label, split, manipulation_type]
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        sample_ratio: float = 1.0,
    ):
        super().__init__()
        self.transform = transform

        df = pd.read_csv(manifest_csv)
        df = df[df["split"] == split].reset_index(drop=True)

        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)

        self.samples: List[Tuple[str, int]] = list(
            zip(df["frame_path"].tolist(), df["label"].tolist())
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = _load_image(path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
#  Sequence dataset (temporal batches for GRU)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Groups individual frames into fixed-length temporal sequences.

    Frames from the same video are kept together; sequences are formed by
    consecutive frames.  A sequence label is the label of its constituent
    frames (all frames in a sequence must share the same label).

    Args:
        base_dataset : CelebDFDataset or FF_FrameDataset (must expose `.samples`).
        seq_len      : Number of frames per sequence.
        stride       : Step between sequence start positions.
    """

    def __init__(self, base_dataset: Dataset, seq_len: int = 8, stride: int = 1):
        self.base = base_dataset
        self.seq_len = seq_len

        # Build sequences: list of (list_of_indices, label)
        self.sequences: List[Tuple[List[int], int]] = []
        n = len(self.base)

        i = 0
        while i + seq_len <= n:
            indices = list(range(i, i + seq_len))
            # Use label of first frame; skip if labels are mixed
            labels = [self.base.samples[j][1] for j in indices]  # type: ignore[attr-defined]
            if len(set(labels)) == 1:
                self.sequences.append((indices, labels[0]))
            i += stride

        # Fallback: if no sequences formed (small dataset), use padding
        if not self.sequences and n > 0:
            indices = list(range(n))
            while len(indices) < seq_len:
                indices.append(indices[-1])  # repeat last frame
            label = self.base.samples[0][1]  # type: ignore[attr-defined]
            self.sequences.append((indices[:seq_len], label))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_indices, label = self.sequences[idx]
        frames = []
        for fi in frame_indices:
            img_tensor, _ = self.base[fi]
            frames.append(img_tensor)
        # Stack → (seq_len, C, H, W)
        seq_tensor = torch.stack(frames, dim=0)
        return seq_tensor, label

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.sequences]


# ─────────────────────────────────────────────────────────────────────────────
#  Combined dataset
# ─────────────────────────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    """Merges multiple datasets into one and exposes a flat `labels` list."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self._offsets = []
        total = 0
        for ds in datasets:
            self._offsets.append(total)
            total += len(ds)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        for i, (ds, offset) in enumerate(zip(self.datasets, self._offsets)):
            next_offset = self._offsets[i + 1] if i + 1 < len(self.datasets) else self._total
            if idx < next_offset:
                return ds[idx - offset]
        raise IndexError(idx)

    @property
    def labels(self) -> List[int]:
        all_labels: List[int] = []
        for ds in self.datasets:
            all_labels.extend(ds.labels)  # type: ignore[attr-defined]
        return all_labels


# ─────────────────────────────────────────────────────────────────────────────
#  Weighted sampler helper
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that rebalances real vs fake classes
    so each class is sampled with equal probability.
    """
    labels = dataset.labels  # type: ignore[attr-defined]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset factory
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    cfg: Dict[str, Any],
    split: str,
    train_transform: Optional[A.Compose] = None,
    val_transform: Optional[A.Compose] = None,
) -> Dataset:
    """
    Build and (optionally) combine CelebDF and FF++ datasets based on `cfg`.

    Returns a single Dataset (possibly CombinedDataset + SequenceDataset).
    """
    transform = train_transform if split == "train" else val_transform
    sources = cfg.get("dataset", {}).get("sources", ["ff", "celeb"])
    use_seq = cfg.get("dataset", {}).get("use_sequences", True)
    seq_len = cfg.get("dataset", {}).get("sequence_length", 8)
    ff_ratio = cfg.get("dataset", {}).get("ff_sample_ratio", 1.0)

    parts: List[Dataset] = []

    if "celeb" in sources:
        celeb_root = cfg["paths"]["celeb_root"]
        celeb_ds = CelebDFDataset(root=celeb_root, split=split, transform=transform)
        if use_seq:
            celeb_ds = SequenceDataset(celeb_ds, seq_len=seq_len)
        parts.append(celeb_ds)

    if "ff" in sources:
        ff_manifest = cfg["paths"].get(
            "ff_manifest",
            os.path.join(cfg["paths"]["ff_frames"], "manifest.csv"),
        )
        if os.path.exists(ff_manifest):
            ff_ds = FF_FrameDataset(
                manifest_csv=ff_manifest,
                split=split,
                transform=transform,
                sample_ratio=ff_ratio,
            )
            if use_seq:
                ff_ds = SequenceDataset(ff_ds, seq_len=seq_len)
            parts.append(ff_ds)

    if not parts:
        raise RuntimeError(
            "No dataset could be loaded. "
            "Run scripts/preprocess_ff.py first, or check your config paths."
        )

    if len(parts) == 1:
        return parts[0]
    return CombinedDataset(parts)
