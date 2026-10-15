"""Local data pipeline for the UCF-Crime dataset.

Scans a raw dataset directory on the local hard drive, transcodes every
.mp4 video to 10 FPS at 128x171 with OpenCV, validates each processed video
contains a minimum of 16 frames, and generates three manifest CSV files
(manifest_train.csv, manifest_val.csv, manifest_test.csv) with columns:
video_path, label (0=normal, 1=anomaly), category, split.
"""

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcode raw UCF-Crime videos and generate train/val/test manifests."
    )
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to the raw UCF-Crime dataset directory on the local hard drive.")
    parser.add_argument("--processed-dir", type=str, default="./data/ucf_crime_processed",
                        help="Directory where transcoded videos are written.")
    parser.add_argument("--manifest-dir", type=str, default=None,
                        help="Directory for the manifest CSVs (default: same as --processed-dir).")
    parser.add_argument("--test-list", type=str, default=None,
                        help="Optional official UCF-Crime test list (one filename per line). "
                             "Matching videos are assigned to the test split; the rest are "
                             "split 90/10 into train/val.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Validation fraction carved from the train split (default: 0.1).")
    parser.add_argument("--target-fps", type=int, default=10,
                        help="Target FPS for transcoding (default: 10).")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=[128, 171],
                        help="Resize dimensions for transcoding (default: 128 171).")
    parser.add_argument("--min-frames", type=int, default=16,
                        help="Minimum number of frames a processed video must contain (default: 16).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split (default: 42).")
    return parser.parse_args()


def determine_label_and_category(rel_parts: tuple, filename: str) -> Optional[tuple]:
    """Infer (label, category) from UCF-Crime directory structure and filename.

    Returns (label, category) or None if the video cannot be classified.
    """
    label = None
    for part in rel_parts[:-1]:
        lowered = part.lower()
        if "normal" in lowered:
            label = 0
            break
        if "anomal" in lowered:
            label = 1
            break

    lowered_name = filename.lower()
    if label is None:
        if "normal" in lowered_name:
            label = 0
        elif "anomal" in lowered_name:
            label = 1

    if label is None:
        return None

    if label == 0:
        return 0, "Normal"

    parent = rel_parts[-2] if len(rel_parts) >= 2 else ""
    if parent and "anomal" not in parent.lower() and "normal" not in parent.lower():
        return 1, parent

    match = re.match(r"([A-Za-z]+)", filename)
    category = match.group(1) if match else "Unknown"
    return 1, category


def transcode_video(src_path: Path, dst_path: Path, target_fps: int,
                    resize_h: int, resize_w: int, min_frames: int) -> Optional[int]:
    """Transcode one video to target FPS at (resize_h x resize_w).

    Returns the number of frames written, or None if the video was
    corrupted or produced fewer than min_frames.
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        logger.warning("Corrupted (unopenable) video, dropping: %s", src_path)
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    frame_stride = max(1, round(source_fps / target_fps))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             target_fps, (resize_w, resize_h))
    if not writer.isOpened():
        cap.release()
        logger.warning("Could not open VideoWriter, dropping: %s", src_path)
        return None

    written = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            resized = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
            written += 1
        idx += 1
    cap.release()
    writer.release()

    if written < min_frames:
        logger.warning("Too short after transcode (%d frames < %d), dropping: %s",
                       written, min_frames, src_path)
        dst_path.unlink(missing_ok=True)
        return None

    check = cv2.VideoCapture(str(dst_path))
    valid = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if valid < min_frames:
        logger.warning("Output verification failed (%d frames), dropping: %s", valid, src_path)
        dst_path.unlink(missing_ok=True)
        return None

    return written


def load_test_list(test_list_path: str) -> set:
    """Load the official test list (one video filename per line)."""
    names = set()
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(Path(name).name)
    return names


def stratified_split(records: List[dict], val_fraction: float, test_fraction: float, seed: int) -> None:
    """Assign train/val/test splits stratified by (label, category) in place."""
    rng = random.Random(seed)
    groups = {}
    for rec in records:
        groups.setdefault((rec["label"], rec["category"]), []).append(rec)

    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(round(n * test_fraction))) if n > 2 else 0
        remaining = n - n_test
        n_val = max(1, int(round(remaining * val_fraction))) if remaining > 1 else 0
        for i, rec in enumerate(group):
            if i < n_test:
                rec["split"] = "test"
            elif i < n_test + n_val:
                rec["split"] = "val"
            else:
                rec["split"] = "train"


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        logger.error("Raw dataset directory not found: %s", raw_dir)
        sys.exit(1)

    processed_dir = Path(args.processed_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else processed_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    test_names = load_test_list(args.test_list) if args.test_list else set()

    video_files = sorted(p for p in raw_dir.rglob("*.mp4") if p.is_file())
    if not video_files:
        logger.error("No .mp4 files found under: %s", raw_dir)
        sys.exit(1)
    logger.info("Found %d raw .mp4 files under %s", len(video_files), raw_dir)

    records: List[dict] = []
    rejected: List[dict] = []

    for src_path in tqdm(video_files, desc="Transcoding"):
        rel_parts = src_path.relative_to(raw_dir).parts
        classified = determine_label_and_category(rel_parts, src_path.name)
        if classified is None:
            logger.warning("Cannot infer label, dropping: %s", src_path)
            rejected.append({"video": str(src_path), "reason": "unknown_label"})
            continue

        label, category = classified
        dst_path = processed_dir / category / src_path.name

        written = transcode_video(src_path, dst_path, args.target_fps,
                                  args.resize[0], args.resize[1], args.min_frames)
        if written is None:
            rejected.append({"video": str(src_path), "reason": "corrupted_or_short"})
            continue

        split = "test" if src_path.name in test_names else None
        records.append({
            "video_path": str(dst_path),
            "label": label,
            "category": category,
            "split": split,
            "num_frames": written,
        })

    if test_names:
        unmatched = [r for r in records if r["split"] is None]
        stratified_split(unmatched, args.val_fraction, 0.0, args.seed)
    else:
        stratified_split(records, args.val_fraction, 0.1, args.seed)

    for split_name in ["train", "val", "test"]:
        split_records = [r for r in records if r["split"] == split_name]
        df = pd.DataFrame(split_records, columns=["video_path", "label", "category", "split"])
        out_path = manifest_dir / f"manifest_{split_name}.csv"
        df.to_csv(out_path, index=False)
        n_anom = int((df["label"] == 1).sum())
        n_norm = int((df["label"] == 0).sum())
        logger.info("manifest_%s.csv: %d videos (%d anomalous, %d normal)",
                    split_name, len(df), n_anom, n_norm)
        if split_name == "train" and (n_anom == 0 or n_norm == 0):
            logger.error("Train split must contain both normal and anomalous videos.")
            sys.exit(1)

    rejected_path = manifest_dir / "data_quality_report.csv"
    pd.DataFrame(rejected, columns=["video", "reason"]).to_csv(rejected_path, index=False)
    logger.info("Dropped %d videos (see %s)", len(rejected), rejected_path)
    logger.info("Processed %d videos into %s", len(records), processed_dir)
    logger.info("Manifests written to %s", manifest_dir)


if __name__ == "__main__":
    main()
