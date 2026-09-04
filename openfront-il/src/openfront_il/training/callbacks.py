"""Checkpointing cadence, early stopping, and metric logging for
``bc_trainer.py``. No metric implementations shared with ``evaluation/``
— see ``training/README.md`` for why that boundary is deliberate.
"""

from __future__ import annotations

from pathlib import Path

from openfront_il.models.checkpoint_format import CheckpointMetadata, save_checkpoint
from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Tracks validation loss across epochs; signals stop after
    ``patience`` consecutive epochs with no improvement greater than
    ``min_delta``.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        """Records one epoch's validation loss. Returns True if training
        should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def log_epoch_metrics(
    epoch: int, train_loss: float, val_loss: float, val_metrics: dict[str, float]
) -> None:
    metrics_str = ", ".join(f"{name}={value:.4f}" for name, value in val_metrics.items())
    logger.info(
        "epoch %d: train_loss=%.4f val_loss=%.4f %s", epoch, train_loss, val_loss, metrics_str
    )


def checkpoint_if_best(
    policy: PolicyNetwork,
    metadata: CheckpointMetadata,
    val_loss: float,
    best_val_loss: float,
    checkpoint_dir: Path,
) -> float:
    """Saves ``policy`` to ``checkpoint_dir/best.pt`` if ``val_loss``
    improves on ``best_val_loss``.

    Returns:
        The new best validation loss (``val_loss`` if it improved,
        otherwise ``best_val_loss`` unchanged).
    """
    if val_loss < best_val_loss:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(checkpoint_dir / "best.pt", policy, metadata)
        logger.info("new best checkpoint: val_loss=%.4f", val_loss)
        return val_loss
    return best_val_loss
