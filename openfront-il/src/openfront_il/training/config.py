"""pydantic config loaded from ``configs/training.yaml`` and
``configs/model.yaml``. No hardcoded hyperparameters belong in
``training/`` code — everything tunable comes from these two files
through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    seed: int
    checkpoint_dir: str
    early_stopping_patience: int
    action_type_loss_weight: float = 1.0
    unit_type_loss_weight: float = 1.0
    tile_target_loss_weight: float = 1.0
    target_player_loss_weight: float = 1.0
    noop_downweight: float = 1.0
    # Fraction of noop examples kept each epoch (all real actions are
    # always kept). Kept noops are re-weighted by 1 / this in the loss, so
    # the model still learns the true human noop rate.
    noop_keep_fraction: float = 1.0
    # Log progress every N training batches; 0 = only once per epoch.
    log_every_batches: int = 0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    # Hardware. Defaults are the safe, portable choice (they're what tests
    # use); configs/training.yaml sets the fast ones.
    device: Literal["auto", "cuda", "cpu"] = "auto"
    num_workers: int = 0
    mixed_precision: bool = False


class SpatialEncoderArchConfig(BaseModel):
    latent_dim: int = 128
    conv_channels: list[int] = [
        16,
        32,
        64,
    ]  # noqa: RUF012 - pydantic model field, not a mutable class attr


class PolicyNetworkArchConfig(BaseModel):
    hidden_dim: int = 128


class ModelConfig(BaseModel):
    spatial_encoder: SpatialEncoderArchConfig = SpatialEncoderArchConfig()
    policy_network: PolicyNetworkArchConfig = PolicyNetworkArchConfig()


def load_training_config(path: Path) -> TrainingConfig:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return TrainingConfig.model_validate(raw)


def load_model_config(path: Path) -> ModelConfig:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return ModelConfig.model_validate(raw)
