"""Configuration module for spatio-temporal video anomaly detection.

This module provides the central ProjectConfig dataclass containing all hyperparameters,
path configurations, and constants for data preparation, model training (MIL 3D CNN),
validation, and video inference. It also includes utility functions for reproducible
seed initialization, hardware device selection, and JSON serialization.
"""

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class ProjectConfig:
    """Central configuration for spatio-temporal video anomaly detection.

    Attributes:
        # Clip settings
        CLIP_LENGTH: Number of frames per spatio-temporal clip (default: 16).
        CLIP_STRIDE: Temporal stride (frames) between consecutive clips (default: 8).
        TARGET_FPS: Target frame rate for video decoding and sampling (default: 10).

        # Spatial settings
        RESIZE_H: Intermediate resize height for input frames (default: 128).
        RESIZE_W: Intermediate resize width for input frames (default: 171).
        CROP_SIZE: Spatial crop dimension (112x112) for 3D CNN input (default: 112).
        INFERENCE_H: Inference spatial height (default: 112).
        INFERENCE_W: Inference spatial width (default: 144).

        # Normalization
        KINETICS_MEAN: Channel-wise mean for Kinetics-400 normalization (RGB)
            (default: [0.43216, 0.394666, 0.37645]).
        KINETICS_STD: Channel-wise standard deviation for Kinetics-400 normalization (RGB)
            (default: [0.22803, 0.22145, 0.216989]).

        # MIL settings
        CLIPS_PER_BAG: Number of clips sampled per video bag in MIL training (default: 32).
        RANKING_MARGIN: Margin parameter m in the MIL hinge ranking loss (default: 0.5).
        LAMBDA_SPARSE: Regularization weight for temporal sparsity loss (default: 8e-5).
        LAMBDA_SMOOTH: Regularization weight for temporal smoothness loss (default: 8e-5).
        TOPK: Number of top-k clips to average in MIL ranking; None means max-only MIL (default: None).

        # Training settings
        LR: Initial learning rate for Adam/AdamW optimizer (default: 1e-4).
        WEIGHT_DECAY: Weight decay (L2 penalty) for optimizer (default: 5e-5).
        EPOCHS: Maximum number of training epochs (default: 60).
        PATIENCE: Early stopping patience epochs without validation improvement (default: 12).
        BATCH_SIZE: Number of bag-pairs (1 positive + 1 negative video bag) per batch (default: 4).
        AMP: Whether to use Automatic Mixed Precision (torch.cuda.amp) (default: True).
        GRAD_CLIP: Maximum gradient norm for gradient clipping (default: 1.0).

        # Inference settings
        EMA_ALPHA: Exponential moving average smoothing factor for anomaly scores (default: 0.3).
        ENTER_THRESHOLD: Baseline anomaly score threshold to open an event interval (default: 0.5).
        EXIT_DELTA: Hysteresis delta; event closes when score < (ENTER_THRESHOLD - EXIT_DELTA) (default: 0.1).
        RUN_LENGTH: Persistence requirement (consecutive clips) to trigger/close event (default: 3).

        # Paths (Google Drive defaults for Colab workflow)
        DATA_ROOT: Google Drive path to UCF-Crime dataset and manifests.
        CHECKPOINT_DIR: Google Drive directory to store model checkpoints (.pt).
        RUNS_DIR: Google Drive directory to store logs, TensorBoard runs, and metrics.
        OUTPUT_DIR: Google Drive directory for CSV timelines and annotated MP4 outputs.

        # Seeds
        SEED: Global random seed for reproducible runs (default: 42).
    """

    # Clip settings
    CLIP_LENGTH: int = 16
    CLIP_STRIDE: int = 8
    TARGET_FPS: int = 10

    # Spatial settings
    RESIZE_H: int = 128
    RESIZE_W: int = 171
    CROP_SIZE: int = 112
    INFERENCE_H: int = 112
    INFERENCE_W: int = 144

    # Normalization
    KINETICS_MEAN: List[float] = field(
        default_factory=lambda: [0.43216, 0.394666, 0.37645]
    )
    KINETICS_STD: List[float] = field(
        default_factory=lambda: [0.22803, 0.22145, 0.216989]
    )

    # MIL settings
    CLIPS_PER_BAG: int = 32
    RANKING_MARGIN: float = 0.5
    LAMBDA_SPARSE: float = 8e-5
    LAMBDA_SMOOTH: float = 8e-5
    TOPK: Optional[int] = None

    # Training settings
    LR: float = 1e-4
    WEIGHT_DECAY: float = 5e-5
    EPOCHS: int = 60
    PATIENCE: int = 12
    BATCH_SIZE: int = 4
    AMP: bool = True
    GRAD_CLIP: float = 1.0

    # Inference settings
    EMA_ALPHA: float = 0.3
    ENTER_THRESHOLD: float = 0.5
    EXIT_DELTA: float = 0.1
    RUN_LENGTH: int = 3

    # Paths (Local defaults)
    DATA_ROOT: str = "./data"
    CHECKPOINT_DIR: str = "./checkpoints"
    RUNS_DIR: str = "./runs"
    OUTPUT_DIR: str = "./outputs"

    # Seed
    SEED: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        """Instantiate ProjectConfig from a dictionary.

        Supports both uppercase and lowercase keys for flexibility.

        Args:
            data: Dictionary containing configuration key-value pairs.

        Returns:
            ProjectConfig: Config instance initialized with the provided values.
        """
        field_dict = {f.name: f for f in fields(cls)}
        field_dict_lower = {f.name.lower(): f.name for f in fields(cls)}

        kwargs: Dict[str, Any] = {}
        for key, val in data.items():
            if key in field_dict:
                kwargs[key] = val
            elif key.lower() in field_dict_lower:
                target_key = field_dict_lower[key.lower()]
                kwargs[target_key] = val

        return cls(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Allow fallback case-insensitive attribute access (e.g. cfg.lr or cfg.clip_length)."""
        upper = name.upper()
        if upper in self.__dict__:
            return self.__dict__[upper]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Configures Python's random module, NumPy, Python hash seed, PyTorch CPU/CUDA
    generators, and sets cuDNN to deterministic mode.

    Args:
        seed: Integer seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the primary computing device available.

    Returns:
        torch.device: 'cuda' device if CUDA is available, 'mps' if Apple Silicon is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_config(config: ProjectConfig, path: Union[str, Path]) -> None:
    """Serialize and save a ProjectConfig dataclass instance to a JSON file.

    Args:
        config: The ProjectConfig instance to serialize.
        path: Destination file path (str or Path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=4)


def load_config(path: Union[str, Path]) -> ProjectConfig:
    """Load a ProjectConfig instance from a JSON file.

    Supports case-insensitive key mapping to ensure compatibility.

    Args:
        path: Path to the JSON configuration file (str or Path).

    Returns:
        ProjectConfig: Loaded configuration instance.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ProjectConfig.from_dict(data)


__all__ = [
    "ProjectConfig",
    "set_seed",
    "get_device",
    "save_config",
    "load_config",
]
