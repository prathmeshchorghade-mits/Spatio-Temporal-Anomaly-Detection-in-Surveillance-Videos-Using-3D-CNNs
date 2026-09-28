"""MIL Bag Dataset for UCF-Crime video anomaly detection.

Provides PyTorch Datasets for MIL bag-pair training and full-video
validation scoring, along with custom collate functions.
"""

import os
import random
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.dataset.transforms import (
    VideoTransform,
    temporal_segment_sampling,
)

logger = logging.getLogger(__name__)


class UCFCrimeManifest:
    """Loads and manages the UCF-Crime dataset manifest CSV.

    Expected CSV columns: video_path, label (0=normal, 1=anomalous), category, split
    """

    def __init__(self, manifest_path: str):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.df = pd.read_csv(manifest_path)

        required_cols = ["video_path", "label", "category", "split"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Manifest missing required column: {col}")

        self.normal_videos: List[str] = self.df[self.df["label"] == 0][
            "video_path"
        ].tolist()
        self.anomalous_videos: List[str] = self.df[self.df["label"] == 1][
            "video_path"
        ].tolist()
        self.all_videos: List[str] = self.df["video_path"].tolist()

        self._path_to_cat = dict(zip(self.df["video_path"], self.df["category"]))
        self._path_to_label = dict(zip(self.df["video_path"], self.df["label"]))

    def get_category(self, video_path: str) -> str:
        """Returns the anomaly category for a given video path."""
        return self._path_to_cat.get(video_path, "unknown")

    def get_label(self, video_path: str) -> int:
        """Returns the label (0 or 1) for a given video path."""
        return self._path_to_label.get(video_path, 0)


# ---------------------------------------------------------------------------
# Video I/O helpers (OpenCV-based, consistent with transforms.py)
# ---------------------------------------------------------------------------

def _get_video_frame_count_and_fps(video_path: str) -> Tuple[int, float]:
    """Returns (total_frames, fps) for a video using OpenCV metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0  # fallback
    return total, fps


def _read_clip_from_video(
    video_path: str,
    start_frame: int,
    clip_length: int,
    frame_stride: int = 1,
) -> Optional[np.ndarray]:
    """Reads *clip_length* frames starting at *start_frame* with a given stride.

    Returns:
        np.ndarray of shape (T, H, W, 3) uint8 RGB, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: List[np.ndarray] = []
    idx = 0
    while len(frames) < clip_length:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if len(frames) == 0:
        return None

    # Pad if too short (repeat last frame)
    while len(frames) < clip_length:
        frames.append(frames[-1].copy())

    return np.stack(frames, axis=0)  # (T, H, W, C)


# ---------------------------------------------------------------------------
# Training dataset — returns (anomalous_bag, normal_bag) pairs
# ---------------------------------------------------------------------------

class MILBagDataset(Dataset):
    """MIL bag-pair dataset for training.

    Each ``__getitem__`` returns ``(anomalous_bag, normal_bag)`` where each
    bag is a ``Tensor`` of shape ``(clips_per_bag, C, T, H, W)``.
    """

    def __init__(
        self,
        manifest: UCFCrimeManifest,
        transform: VideoTransform,
        clips_per_bag: int = 32,
        clip_length: int = 16,
        target_fps: int = 10,
    ):
        self.manifest = manifest
        self.transform = transform
        self.clips_per_bag = clips_per_bag
        self.clip_length = clip_length
        self.target_fps = target_fps

        self.normal_videos = list(manifest.normal_videos)
        self.anomalous_videos = list(manifest.anomalous_videos)

        if not self.normal_videos or not self.anomalous_videos:
            raise ValueError(
                "Manifest must contain both normal and anomalous videos."
            )

    def __len__(self) -> int:
        return min(len(self.normal_videos), len(self.anomalous_videos))

    # ----- internal helpers -----

    def _extract_bag(self, video_path: str) -> Optional[torch.Tensor]:
        """Sample *clips_per_bag* clips from *video_path* via segment sampling."""
        try:
            total_frames, source_fps = _get_video_frame_count_and_fps(video_path)
            if total_frames == 0:
                return None

            # Source-frame stride to hit target_fps
            frame_stride = max(1, round(source_fps / self.target_fps))

            # Segment sampling (returns start-frame indices at target_fps scale)
            # We need total "target-fps frames"
            target_total = total_frames // frame_stride
            start_indices = temporal_segment_sampling(
                total_frames=target_total,
                num_segments=self.clips_per_bag,
                clip_length=self.clip_length,
                jitter=True,
            )
            if not start_indices:
                return None

            clips: List[torch.Tensor] = []
            for si in start_indices:
                source_start = si * frame_stride
                clip_np = _read_clip_from_video(
                    video_path, source_start, self.clip_length, frame_stride
                )
                if clip_np is None:
                    continue
                clip_tensor = self.transform(clip_np)  # (C, T, H, W)
                clips.append(clip_tensor)

            if len(clips) < self.clips_per_bag:
                # pad with last clip
                while len(clips) < self.clips_per_bag:
                    clips.append(clips[-1].clone())

            return torch.stack(clips[: self.clips_per_bag], dim=0)

        except Exception as e:
            logger.warning(f"Failed to extract bag from {video_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_retries = 10
        for attempt in range(max_retries):
            anom_idx = (idx + random.randint(0, len(self.anomalous_videos) - 1)) % len(
                self.anomalous_videos
            )
            norm_idx = (idx + random.randint(0, len(self.normal_videos) - 1)) % len(
                self.normal_videos
            )

            anom_bag = self._extract_bag(self.anomalous_videos[anom_idx])
            norm_bag = self._extract_bag(self.normal_videos[norm_idx])

            if anom_bag is not None and norm_bag is not None:
                return anom_bag, norm_bag

        raise RuntimeError(
            "Failed to load a valid bag-pair after maximum retries."
        )


# ---------------------------------------------------------------------------
# Validation / test dataset — scores ALL clips from each video
# ---------------------------------------------------------------------------

class MILBagValidationDataset(Dataset):
    """Scores *every* clip from each video with a sliding window.

    Returns ``(clips_tensor, label, video_path, timestamps)`` per video.
    """

    def __init__(
        self,
        manifest: UCFCrimeManifest,
        transform: VideoTransform,
        clip_length: int = 16,
        clip_stride: int = 8,
        target_fps: int = 10,
    ):
        self.manifest = manifest
        self.transform = transform
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.target_fps = target_fps

        self.videos = list(manifest.all_videos)
        self.labels = [manifest.get_label(v) for v in self.videos]

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int):
        video_path = self.videos[idx]
        label = self.labels[idx]

        try:
            total_frames, source_fps = _get_video_frame_count_and_fps(video_path)
            if total_frames == 0:
                raise ValueError("Empty video")

            frame_stride = max(1, round(source_fps / self.target_fps))
            target_total = total_frames // frame_stride

            clips: List[torch.Tensor] = []
            timestamps: List[Tuple[float, float]] = []

            for si in range(0, target_total - self.clip_length + 1, self.clip_stride):
                source_start = si * frame_stride
                clip_np = _read_clip_from_video(
                    video_path, source_start, self.clip_length, frame_stride
                )
                if clip_np is None:
                    continue
                clip_tensor = self.transform(clip_np)
                clips.append(clip_tensor)

                start_sec = si / self.target_fps
                end_sec = (si + self.clip_length) / self.target_fps
                timestamps.append((start_sec, end_sec))

            if not clips:
                raise ValueError("No clips extracted")

            clips_tensor = torch.stack(clips, dim=0)  # (N, C, T, H, W)
            return clips_tensor, label, video_path, timestamps

        except Exception as e:
            logger.error(f"Validation load error for {video_path}: {e}")
            dummy = torch.zeros(1, 3, self.clip_length, 112, 112)
            return dummy, label, video_path, [(0.0, 0.0)]


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def mil_collate_fn(batch):
    """Stack bag-pairs into (anom_batch, norm_batch) each of shape
    ``(B, clips_per_bag, C, T, H, W)``."""
    anom_bags = [item[0] for item in batch]
    norm_bags = [item[1] for item in batch]
    return torch.stack(anom_bags, dim=0), torch.stack(norm_bags, dim=0)


def val_collate_fn(batch):
    """Variable-length clips per video — return lists, not stacked tensors."""
    clips_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    paths_list = [item[2] for item in batch]
    timestamps_list = [item[3] for item in batch]
    return clips_list, labels_list, paths_list, timestamps_list
