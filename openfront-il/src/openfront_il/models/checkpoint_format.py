"""The saved-checkpoint schema a later RL stage will load, plus
``save_checkpoint()``/``load_checkpoint()`` helpers shared by
``training/`` and (later) the RL stage.

This is the single most important file for the RL handoff (see
``docs/rl_handoff.md``, once written) — a later stage depends on loading
this checkpoint and knowing exactly which submodules (e.g. the spatial
encoder) are meant to be reused versus reinitialized (e.g. a value head
BC never trained). Any breaking change to this schema must bump
``FORMAT_VERSION`` and be called out explicitly in ``docs/rl_handoff.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.models.spatial_encoder import SpatialEncoder

# v2: SpatialEncoder returns its feature map alongside the pooled latent,
# and PolicyNetwork gained the tile_target / target_player heads (plus
# PolicyNetworkConfig.player_feature_dim). See docs/rl_handoff.md.
FORMAT_VERSION = 2


class SpatialEncoderConfig(BaseModel):
    in_channels: int
    latent_dim: int = 128
    conv_channels: tuple[int, ...] = (16, 32, 64)


class PolicyNetworkConfig(BaseModel):
    flat_dim: int
    action_type_count: int
    unit_type_count: int
    player_feature_dim: int
    hidden_dim: int = 128


class CheckpointMetadata(BaseModel):
    """Everything needed to reconstruct the exact architecture a saved
    state dict was trained with, plus training provenance.
    """

    format_version: int = FORMAT_VERSION
    spatial_encoder_config: SpatialEncoderConfig
    policy_network_config: PolicyNetworkConfig
    # Opaque to this module by design: training/ owns the shape of its own
    # config (learning rate, epochs, etc.) and checkpoint_format.py must
    # not need to change every time a training hyperparameter is added.
    training_config: dict[str, Any] = {}
    seed: int | None = None


def build_policy_network(metadata: CheckpointMetadata) -> PolicyNetwork:
    """Constructs an untrained `PolicyNetwork` matching `metadata`'s
    architecture config (no weights loaded — see `load_checkpoint` for that).
    """
    spatial_encoder = SpatialEncoder(**metadata.spatial_encoder_config.model_dump())
    return PolicyNetwork(
        spatial_encoder=spatial_encoder, **metadata.policy_network_config.model_dump()
    )


def save_checkpoint(path: Path, policy: PolicyNetwork, metadata: CheckpointMetadata) -> None:
    """Saves `policy`'s weights alongside the config needed to rebuild its
    architecture and the training provenance that produced it.
    """
    torch.save(
        {"metadata": metadata.model_dump(), "model_state_dict": policy.state_dict()},
        path,
    )


def load_checkpoint(path: Path) -> tuple[PolicyNetwork, CheckpointMetadata]:
    """Loads a checkpoint written by `save_checkpoint`, reconstructing the
    exact architecture it was trained with and restoring its weights.

    `weights_only=False` is deliberate: this always loads a checkpoint
    this project wrote itself (never an untrusted, arbitrary file), and
    `metadata` is a plain dict of Python primitives (pydantic's
    `model_dump()`), not custom classes — `torch.load`'s
    `weights_only=True` default is a defense against loading arbitrary,
    untrusted pickle payloads, which does not apply here.

    Raises:
        ValueError: if the checkpoint's format_version doesn't match
            `FORMAT_VERSION`.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata = CheckpointMetadata.model_validate(raw["metadata"])
    if metadata.format_version != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format_version {metadata.format_version} != expected {FORMAT_VERSION}"
        )
    policy = build_policy_network(metadata)
    policy.load_state_dict(raw["model_state_dict"])
    return policy, metadata
