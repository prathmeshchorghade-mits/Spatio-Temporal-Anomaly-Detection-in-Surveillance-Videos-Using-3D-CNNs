"""Training package for spatio-temporal video anomaly detection."""

from training.config import (
    ProjectConfig,
    get_device,
    load_config,
    save_config,
    set_seed,
)

__all__ = [
    "ProjectConfig",
    "set_seed",
    "get_device",
    "save_config",
    "load_config",
]
