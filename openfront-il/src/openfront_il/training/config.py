"""pydantic config loaded from ``configs/training.yaml`` and
``configs/model.yaml``. No hardcoded hyperparameters belong in
``training/`` code — everything tunable comes from these two files
through this module.
"""

from __future__ import annotations

from pathlib import Path

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
    noop_downweight: float = 1.0
    val_fraction: float = 0.1
    test_fraction: float = 0.1


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
