"""
Inference pipeline for the deepfake detector.

Supports three backends:
  - PyTorch (float32 or quantised TorchScript)
  - ONNX Runtime (float32 or INT8)

The `DeepfakePredictor` class wraps face detection, preprocessing,
batching, and model inference into a single callable object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet stats used during training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize, convert to float32, normalize – returns (C, H, W)."""
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)   # HWC → CHW


class DeepfakePredictor:
    """
    High-level predictor for single images, video files, or batches.

    Args:
        model_path    : Path to `.onnx` or `.pt` (TorchScript) model.
        face_detector : "mtcnn" | "dlib" | None (skip face detection).
        image_size    : Face crop resolution expected by the model.
        seq_len       : Temporal sequence length (match training config).
        threshold     : Decision boundary for fake/real.
        device        : "cpu" | "cuda" (PyTorch backend only).
    """

    def __init__(
        self,
        model_path: str,
        face_detector: str = "mtcnn",
        image_size: int = 224,
        seq_len: int = 8,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.image_size = image_size
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device
        self.model_path = str(model_path)

        self._init_model()
        self._init_detector(face_detector)

    # ── Backend initialisation ───────────────────────────────────────────────

    def _init_model(self) -> None:
        ext = Path(self.model_path).suffix.lower()
        if ext == ".onnx":
            self._backend = "onnx"
            self._init_onnx()
        elif ext == ".pt":
            self._backend = "torch"
            self._init_torch()
        else:
            raise ValueError(f"Unsupported model extension: {ext}")

    def _init_onnx(self) -> None:
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._sess = ort.InferenceSession(self.model_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        # Detect expected input rank (4=single-frame, 5=sequence)
        self._onnx_input_ndim = len(self._sess.get_inputs()[0].shape)
        logger.info("ONNX backend loaded from %s (input ndim=%d)",
                    self.model_path, self._onnx_input_ndim)

    def _init_torch(self) -> None:
        import torch
        self._torch = torch
        self._torch_model = torch.jit.load(self.model_path, map_location=self.device)
        self._torch_model.eval()
        logger.info("TorchScript backend loaded from %s", self.model_path)

    def _init_detector(self, face_detector: Optional[str]) -> None:
        if face_detector is None or face_detector == "none":
            self._detector = None
            return
        # Import lazily to avoid hard dependency at module level
        from src.data.preprocessing import MTCNNDetector, DlibDetector
        if face_detector == "mtcnn":
            self._detector = MTCNNDetector(image_size=self.image_size)
        elif face_detector == "dlib":
            self._detector = DlibDetector(image_size=self.image_size)
        else:
            raise ValueError(f"Unknown face_detector: {face_detector}")

    # ── Core inference ───────────────────────────────────────────────────────

    def _infer(self, seq_np: np.ndarray) -> Tuple[float, str]:
        """
        Run model on a prepared sequence array.

        Args:
            seq_np : float32 numpy array of shape (1, T, C, H, W).
        Returns:
            (fake_probability, label_string)
        """
        if self._backend == "onnx":
            # seq_np is (1, T, C, H, W); collapse to (1, C, H, W) for single-frame models
            inp = seq_np
            if getattr(self, "_onnx_input_ndim", 5) == 4 and inp.ndim == 5:
                inp = inp[:, 0, :, :, :]   # take first frame → (1, C, H, W)
            logits = self._sess.run(None, {self._input_name: inp})[0]
            probs = _softmax(logits[0])
        else:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(seq_np).to(self.device)
                logits = self._torch_model(t).cpu().numpy()[0]
            probs = _softmax(logits)

        fake_prob = float(probs[1])
        label = "FAKE" if fake_prob >= self.threshold else "REAL"
        return fake_prob, label

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_image(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Classify a single image (face crop or uncropped image).

        Args:
            image: File path string or HxWx3 uint8 RGB numpy array.

        Returns:
            dict with keys: label, fake_probability, real_probability
        """
        if isinstance(image, str):
            img = np.array(Image.open(image).convert("RGB"))
        else:
            img = image

        # Face detection
        face = self._detect_face(img)
        if face is None:
            face = img   # fall back to full image

        frame = _normalize(face, self.image_size)   # (C, H, W)
        # Repeat single frame to fill sequence
        seq = np.stack([frame] * self.seq_len, axis=0)  # (T, C, H, W)
        seq = seq[np.newaxis].astype(np.float32)         # (1, T, C, H, W)

        fake_prob, label = self._infer(seq)
        return {
            "label": label,
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
        }

    def predict_video(
        self,
        video_path: str,
        n_frames: int = 16,
        progress_callback=None,
    ) -> Dict:
        """
        Classify a video file by sampling `n_frames` evenly-spaced frames.

        Args:
            video_path        : Path to video (.mp4, .avi, etc.).
            n_frames          : Number of frames to sample.
            progress_callback : Optional callable(current, total) for GUI updates.

        Returns:
            dict with keys: label, fake_probability, real_probability, frame_results
        """
        from src.data.preprocessing import sample_frame_indices

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total, n_frames)

        face_frames: List[np.ndarray] = []
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detected = self._detect_face(rgb)
            face = detected if detected is not None else rgb
            face_frames.append(_normalize(face, self.image_size))
            if progress_callback:
                progress_callback(i + 1, len(indices))
        cap.release()

        if not face_frames:
            return {"label": "ERROR", "fake_probability": 0.0,
                    "real_probability": 0.0, "frame_results": []}

        # Build overlapping sequences of length seq_len
        frame_probs: List[float] = []
        for start in range(0, len(face_frames), max(1, self.seq_len // 2)):
            chunk = face_frames[start: start + self.seq_len]
            while len(chunk) < self.seq_len:
                chunk.append(chunk[-1])     # pad with last frame
            seq = np.stack(chunk, axis=0)[np.newaxis].astype(np.float32)
            prob, _ = self._infer(seq)
            frame_probs.append(prob)

        avg_prob = float(np.mean(frame_probs))
        label = "FAKE" if avg_prob >= self.threshold else "REAL"
        return {
            "label": label,
            "fake_probability": avg_prob,
            "real_probability": 1.0 - avg_prob,
            "frame_results": frame_probs,
        }

    def predict_batch(self, paths: List[str]) -> List[Dict]:
        """Classify a list of image or video paths."""
        results = []
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                result = self.predict_video(path)
            else:
                result = self.predict_image(path)
            result["path"] = path
            results.append(result)
        return results

    def _detect_face(self, img: np.ndarray) -> Optional[np.ndarray]:
        if self._detector is None:
            return None
        try:
            return self._detector.detect_and_align(img)
        except Exception as exc:
            logger.debug("Face detection failed: %s", exc)
            return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()
