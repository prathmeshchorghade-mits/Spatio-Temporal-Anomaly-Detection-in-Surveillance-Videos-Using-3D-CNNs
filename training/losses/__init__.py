"""Loss functions subpackage for Phase 2 dense MIL training."""

from training.losses.dense_mil import dense_mil_loss, topk_mean, proximity_topk_mean

__all__ = ["dense_mil_loss", "topk_mean", "proximity_topk_mean"]
