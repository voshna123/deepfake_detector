"""
Face extraction and alignment pipeline for FaceForensics++ videos.

Supports two backends:
  - MTCNN  (facenet-pytorch) – GPU-accelerated, preferred
  - dlib   – CPU-friendly fallback

Usage (standalone):
    python -m src.data.preprocessing \
        --video_dir data/FaceForensics/FaceForensics++_C23 \
        --out_dir   data/FaceForensics/frames \
        --cfg       configs/config.yaml
"""

from __future__ import annotations

import os
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Face Detector wrappers
# ─────────────────────────────────────────────────────────────────────────────

class MTCNNDetector:
    """MTCNN face detector (facenet-pytorch). Runs on GPU if available."""

    def __init__(self, image_size: int = 224, min_face_size: int = 40,
                 thresholds: Tuple[float, ...] = (0.6, 0.7, 0.9),
                 device: Optional[str] = None):
        from facenet_pytorch import MTCNN
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=14,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=0.709,
            post_process=False,
            device=self.device,
            keep_all=False,
        )
        self.image_size = image_size

    def detect_and_align(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            frame_rgb: HxWx3 uint8 numpy array (RGB).
        Returns:
            Aligned face crop as HxWx3 uint8, or None if no face found.
        """
        pil_img = Image.fromarray(frame_rgb)
        face_tensor = self.mtcnn(pil_img)
        if face_tensor is None:
            return None
        # face_tensor: C x H x W  float in [0,255]
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return face_np


class DlibDetector:
    """Dlib HOG + 68-landmark alignment fallback (CPU)."""

    def __init__(self, image_size: int = 224,
                 landmarks_path: str = "shape_predictor_68_face_landmarks.dat"):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        if not Path(landmarks_path).exists():
            logger.warning(
                "Dlib landmark model not found at %s. "
                "Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                landmarks_path,
            )
            self.predictor = None
        else:
            self.predictor = dlib.shape_predictor(landmarks_path)
        self.image_size = image_size

    def detect_and_align(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        dets = self.detector(gray, 1)
        if not dets:
            return None
        # Use the largest detected face
        det = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
        # Simple square crop around the bounding box
        x1, y1 = max(det.left(), 0), max(det.top(), 0)
        x2, y2 = min(det.right(), frame_rgb.shape[1]), min(det.bottom(), frame_rgb.shape[0])
        face = frame_rgb[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return face


def build_detector(cfg: dict):
    """Instantiate the configured face detector."""
    backend = cfg.get("preprocessing", {}).get("face_detector", "mtcnn").lower()
    image_size = cfg.get("preprocessing", {}).get("image_size", 224)
    min_face_size = cfg.get("preprocessing", {}).get("min_face_size", 40)
    threshold = cfg.get("preprocessing", {}).get("mtcnn_threshold", 0.9)

    if backend == "mtcnn":
        return MTCNNDetector(
            image_size=image_size,
            min_face_size=min_face_size,
            thresholds=(0.6, 0.7, threshold),
        )
    elif backend == "dlib":
        return DlibDetector(image_size=image_size)
    else:
        raise ValueError(f"Unknown face_detector backend: {backend}")


# ─────────────────────────────────────────────────────────────────────────────
#  Video frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def sample_frame_indices(total_frames: int, n_frames: int) -> List[int]:
    """Return `n_frames` evenly-spaced frame indices from a video."""
    if total_frames <= n_frames:
        return list(range(total_frames))
    step = total_frames / n_frames
    return [int(i * step) for i in range(n_frames)]


def extract_faces_from_video(
    video_path: str,
    detector,
    n_frames: int = 30,
    out_dir: Optional[str] = None,
    video_id: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Extract and align faces from `n_frames` evenly-sampled frames of a video.

    Args:
        video_path:  Path to the .mp4 file.
        detector:    A detector instance with `detect_and_align(frame_rgb)`.
        n_frames:    Number of frames to sample.
        out_dir:     If given, save face crops as JPEG files here.
        video_id:    Stem used for saved filenames (defaults to video filename stem).

    Returns:
        List of HxWx3 uint8 numpy arrays (may be shorter than n_frames if
        detection fails on some frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, n_frames)

    faces: List[np.ndarray] = []
    vid_id = video_id or Path(video_path).stem

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = detector.detect_and_align(rgb)
        if face is None:
            continue
        faces.append(face)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{vid_id}_frame{idx:05d}.jpg")
            pil_face = Image.fromarray(face)
            pil_face.save(save_path, "JPEG", quality=95)

    cap.release()
    return faces


# ─────────────────────────────────────────────────────────────────────────────
#  Batch preprocessing of the entire FF++ dataset
# ─────────────────────────────────────────────────────────────────────────────

# Manipulation types present in FaceForensics++
FF_FAKE_TYPES = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "FaceShifter",
    "NeuralTextures",
    "DeepFakeDetection",
]
FF_REAL_TYPE = "original"


def preprocess_ff_dataset(
    ff_root: str,
    out_root: str,
    detector,
    n_frames: int = 30,
    split_csv: Optional[str] = None,
    start_from: Optional[str] = None,
) -> str:
    """
    Walk the FF++ directory tree, extract faces from each video, and write
    a CSV manifest with columns: [frame_path, label, split, manipulation_type].

    `split_csv` (FF++_Metadata_Shuffled.csv) is used to assign train/val/test
    splits if provided; otherwise all videos are assigned "train".

    Returns the path to the generated manifest CSV.
    """
    ff_root = Path(ff_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build split lookup from shuffled metadata if available
    split_lookup: dict = {}
    if split_csv and Path(split_csv).exists():
        import pandas as pd
        meta = pd.read_csv(split_csv)
        # Assign splits: 70/15/15
        n = len(meta)
        for i, row in meta.iterrows():
            if i < int(0.70 * n):
                split_lookup[row["File Path"]] = "train"
            elif i < int(0.85 * n):
                split_lookup[row["File Path"]] = "val"
            else:
                split_lookup[row["File Path"]] = "test"

    manifest_rows: List[dict] = []

    all_types = FF_FAKE_TYPES + [FF_REAL_TYPE]

    # Skip types before start_from if resuming
    if start_from:
        if start_from in all_types:
            all_types = all_types[all_types.index(start_from):]
            logger.info("Resuming from: %s", start_from)
        else:
            logger.warning("start_from '%s' not found, processing all types", start_from)

    # Collect already-extracted rows from previously completed types
    for split in ("train", "val", "test"):
        for label_dir in ("fake", "real"):
            base = Path(out_root) / split / label_dir
            if not base.exists():
                continue
            for type_dir in base.iterdir():
                if not type_dir.is_dir() or type_dir.name in [t for t in all_types]:
                    continue
                lbl = 0 if label_dir == "real" else 1
                for fp in sorted(type_dir.glob("*.jpg")):
                    manifest_rows.append({
                        "frame_path": str(fp),
                        "label": lbl,
                        "split": split,
                        "manipulation_type": type_dir.name,
                    })
    if manifest_rows:
        logger.info("Loaded %d rows from previously completed types", len(manifest_rows))

    for manip_type in all_types:
        label = 0 if manip_type == FF_REAL_TYPE else 1
        video_dir = ff_root / manip_type
        if not video_dir.exists():
            logger.warning("Directory not found, skipping: %s", video_dir)
            continue

        videos = sorted(video_dir.glob("*.mp4"))
        logger.info("Processing %s: %d videos", manip_type, len(videos))

        for video_path in videos:
            rel_path = f"{manip_type}/{video_path.name}"
            split = split_lookup.get(rel_path, "train")
            face_out_dir = out_root / split / ("fake" if label == 1 else "real") / manip_type

            saved_faces = extract_faces_from_video(
                str(video_path),
                detector=detector,
                n_frames=n_frames,
                out_dir=str(face_out_dir),
                video_id=video_path.stem,
            )

            for face in saved_faces:
                # Reconstruct the saved path
                _ = face  # actual path was saved inside extract_faces_from_video
            # Collect file paths written
            if face_out_dir.exists():
                for fp in sorted(face_out_dir.glob(f"{video_path.stem}_frame*.jpg")):
                    manifest_rows.append({
                        "frame_path": str(fp),
                        "label": label,
                        "split": split,
                        "manipulation_type": manip_type,
                    })

    manifest_path = out_root / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_path", "label", "split", "manipulation_type"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info("Manifest written to %s  (%d rows)", manifest_path, len(manifest_rows))
    else:
        logger.warning("No faces extracted – manifest is empty.")

    return str(manifest_path)
