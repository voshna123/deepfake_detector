"""
Preprocess FaceForensics++ videos → extracted face crops.

Usage:
    python scripts/preprocess_ff.py [--cfg configs/config.yaml] [--workers 4]

What it does:
    1. Reads config to locate the FF++ video directory.
    2. Iterates every manipulation type and the original (real) videos.
    3. Samples `preprocessing.frames_per_video` frames from each video.
    4. Detects and aligns faces with MTCNN or Dlib.
    5. Saves face crops as JPEG under data/FaceForensics/frames/{split}/{real|fake}/
    6. Writes a manifest CSV with frame_path, label, split, manipulation_type.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as top-level script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_ff")


def main():
    parser = argparse.ArgumentParser(description="Preprocess FaceForensics++ videos")
    parser.add_argument("--cfg", default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (1 = sequential)")
    parser.add_argument("--start_from", default=None,
                        help="Resume from this manipulation type (e.g. NeuralTextures)")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ff_root  = cfg["paths"]["ff_root"]
    out_root = cfg["paths"]["ff_frames"]
    n_frames = cfg["preprocessing"]["frames_per_video"]
    split_csv = cfg["paths"].get("ff_train_csv")

    logger.info("FF++ root   : %s", ff_root)
    logger.info("Output root : %s", out_root)
    logger.info("Frames/video: %d", n_frames)

    from src.data.preprocessing import build_detector, preprocess_ff_dataset

    detector = build_detector(cfg)
    logger.info("Face detector: %s", cfg["preprocessing"]["face_detector"])

    manifest_path = preprocess_ff_dataset(
        ff_root=ff_root,
        out_root=out_root,
        detector=detector,
        n_frames=n_frames,
        split_csv=split_csv,
        start_from=args.start_from,
    )
    logger.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
