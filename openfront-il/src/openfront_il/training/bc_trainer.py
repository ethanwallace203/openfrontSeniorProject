"""The supervised behavior-cloning training loop: loads
``data/processed/``, builds a ``PolicyNetwork``, trains with per-head
cross-entropy (legality-masked for ``action_type``), and checkpoints the
best epoch via ``callbacks.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from openfront_il.features.action_encoder import ACTION_TYPES, UNIT_TYPES
from openfront_il.features.example_loader import ExampleTable
from openfront_il.features.observation_encoder import FLAT_FEATURES, SPATIAL_CHANNELS
from openfront_il.models.checkpoint_format import (
    CheckpointMetadata,
    PolicyNetworkConfig,
    SpatialEncoderConfig,
    build_policy_network,
)
from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.training.callbacks import EarlyStopping, checkpoint_if_best, log_epoch_metrics
from openfront_il.training.config import ModelConfig, TrainingConfig
from openfront_il.training.seed import seed_everything
from openfront_il.utils import paths
from openfront_il.utils.logging import get_logger

logger = get_logger(__name__)


def game_split_bucket(game_id: str) -> int:
    """A deterministic 0-99 bucket for a gameId, used for game-level
    splitting (see ``docs/imitation_learning_plan.md`` Section 4: split at
    the game level, never the tick level, so correlated ticks from the
    same game never leak across train/val/test).
    """
    digest = hashlib.sha256(game_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def split_game_ids(
    game_ids: list[str], *, val_fraction: float, test_fraction: float
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically splits ``game_ids`` by hashing each id into a
    stable 0-99 bucket, so adding a new game never reshuffles existing
    games' splits.
    """
    val_cutoff = int(val_fraction * 100)
    test_cutoff = val_cutoff + int(test_fraction * 100)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for game_id in game_ids:
        bucket = game_split_bucket(game_id)
        if bucket < val_cutoff:
            val.append(game_id)
        elif bucket < test_cutoff:
            test.append(game_id)
        else:
            train.append(game_id)
    return train, val, test


class BCDataset(Dataset[dict[str, Tensor]]):
    """A thin ``torch.utils.data.Dataset`` wrapper around
    ``features.example_loader.ExampleTable`` — see that module's
    docstring for the actual loading/reconstruction logic and its scale
    tradeoffs (shared with ``evaluation/``, which uses the same table).
    """

    def __init__(self, game_ids: list[str]) -> None:
        self.game_ids = game_ids
        self._table = ExampleTable(game_ids)

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self._table.example(index)
        return {
            "spatial": torch.from_numpy(example["spatial"]),
            "flat": torch.from_numpy(example["flat"]),
            "action_type": torch.tensor(int(example["action_type"]), dtype=torch.long),
            "unit_type": torch.tensor(int(example["unit_type"]), dtype=torch.long),
            "action_type_legal": torch.from_numpy(example["legality_mask"]),
        }


def collate_examples(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Custom ``collate_fn`` for ``BCDataset``'s ``DataLoader``.

    Different games' maps have different ``(H, W)`` — a real, measured
    fact (Europe is 2904x1672; other maps are far smaller) —
    ``SpatialEncoder``'s adaptive pooling handles any size *per example*,
    but a batch mixing examples from different games needs every spatial
    tensor stacked into one tensor, and torch's default ``collate_fn``
    requires equal shapes to do that. This pads every spatial tensor in
    the batch to the batch's own max ``(H, W)`` with zeros before
    stacking.

    Zero-padding is a defensible simplification for a prototype, not a
    fully correct one: every channel here is 0 for "nothing here" (no
    land, no ownership, no fallout — see ``observation_encoder.py``), so
    padded cells read as neutral open ocean, but they still get averaged
    into `SpatialEncoder`'s ``AdaptiveAvgPool2d`` alongside real map data,
    diluting it slightly more for a small map padded up to match a large
    one in the same batch. A fully correct fix would mask padded regions
    out of the pool; not implemented here.
    """
    max_height = max(item["spatial"].shape[1] for item in batch)
    max_width = max(item["spatial"].shape[2] for item in batch)

    padded_spatial = []
    for item in batch:
        spatial = item["spatial"]
        pad_bottom = max_height - spatial.shape[1]
        pad_right = max_width - spatial.shape[2]
        padded_spatial.append(functional.pad(spatial, (0, pad_right, 0, pad_bottom)))

    return {
        "spatial": torch.stack(padded_spatial),
        "flat": torch.stack([item["flat"] for item in batch]),
        "action_type": torch.stack([item["action_type"] for item in batch]),
        "unit_type": torch.stack([item["unit_type"] for item in batch]),
        "action_type_legal": torch.stack([item["action_type_legal"] for item in batch]),
    }


def masked_cross_entropy(
    logits: Tensor, target: Tensor, legal_mask: Tensor, *, weight: Tensor | None = None
) -> Tensor:
    """Cross-entropy with illegal classes' logits set to ``-inf`` before
    the softmax, per ``training/README.md``'s contract: "mask out illegal
    actions before computing loss."

    The true label's own logit is never masked out, even if
    ``legal_mask`` marks it illegal (the coarse mask can be wrong for a
    handful of action types — see ``docs/imitation_learning_plan.md``
    Section 2) — masking out the correct class would make cross-entropy
    compute ``-inf`` log-probability for it and produce a ``NaN`` loss.
    """
    target_mask = functional.one_hot(target, num_classes=logits.shape[-1]).bool()
    effective_mask = legal_mask | target_mask
    masked_logits = logits.masked_fill(~effective_mask, float("-inf"))
    return functional.cross_entropy(masked_logits, target, weight=weight)


def _compute_loss(
    policy: PolicyNetwork,
    batch: dict[str, Tensor],
    training_config: TrainingConfig,
    action_type_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    logits = policy(batch["spatial"], batch["flat"])
    action_loss = (
        masked_cross_entropy(
            logits["action_type"],
            batch["action_type"],
            batch["action_type_legal"],
            weight=action_type_weight,
        )
        * training_config.action_type_loss_weight
    )
    unit_loss = (
        functional.cross_entropy(logits["unit_type"], batch["unit_type"])
        * training_config.unit_type_loss_weight
    )
    return action_loss + unit_loss, logits


def _evaluate(
    policy: PolicyNetwork,
    data_loader: DataLoader[dict[str, Tensor]],
    training_config: TrainingConfig,
    action_type_weight: Tensor,
) -> tuple[float, dict[str, float]]:
    policy.eval()
    total_loss = 0.0
    total_examples = 0
    correct_action_type = 0
    correct_unit_type = 0
    with torch.no_grad():
        for batch in data_loader:
            loss, logits = _compute_loss(policy, batch, training_config, action_type_weight)
            batch_size = batch["action_type"].shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            correct_action_type += int(
                (logits["action_type"].argmax(dim=-1) == batch["action_type"]).sum().item()
            )
            correct_unit_type += int(
                (logits["unit_type"].argmax(dim=-1) == batch["unit_type"]).sum().item()
            )
    if total_examples == 0:
        return 0.0, {"action_type_accuracy": 0.0, "unit_type_accuracy": 0.0}
    return total_loss / total_examples, {
        "action_type_accuracy": correct_action_type / total_examples,
        "unit_type_accuracy": correct_unit_type / total_examples,
    }


def build_model_for_config(model_config: ModelConfig) -> tuple[PolicyNetwork, CheckpointMetadata]:
    """Builds an untrained `PolicyNetwork` whose input/output dimensions
    match `features/`'s current tensor shapes exactly (`len(...)` on the
    relevant `features/` constants), with architecture hyperparameters
    (latent dim, hidden dim, conv channels) from `model_config`.
    """
    spatial_config = SpatialEncoderConfig(
        in_channels=len(SPATIAL_CHANNELS),
        latent_dim=model_config.spatial_encoder.latent_dim,
        conv_channels=tuple(model_config.spatial_encoder.conv_channels),
    )
    policy_config = PolicyNetworkConfig(
        flat_dim=len(FLAT_FEATURES),
        action_type_count=len(ACTION_TYPES),
        unit_type_count=len(UNIT_TYPES),
        hidden_dim=model_config.policy_network.hidden_dim,
    )
    metadata = CheckpointMetadata(
        spatial_encoder_config=spatial_config, policy_network_config=policy_config
    )
    return build_policy_network(metadata), metadata


def train(training_config: TrainingConfig, model_config: ModelConfig) -> Path:
    """Runs the full BC training loop against every game in
    ``data/processed/``.

    Returns:
        The path to the best checkpoint written during training.

    Raises:
        RuntimeError: if ``data/processed/`` has no processed games.
    """
    seed_everything(training_config.seed)

    game_ids = sorted(path.stem for path in paths.PROCESSED_DIR.glob("*.parquet"))
    if not game_ids:
        raise RuntimeError("no processed games found in data/processed/ — run build_dataset first")

    train_ids, val_ids, test_ids = split_game_ids(
        game_ids,
        val_fraction=training_config.val_fraction,
        test_fraction=training_config.test_fraction,
    )
    logger.info(
        "split: %d train, %d val, %d test game(s)", len(train_ids), len(val_ids), len(test_ids)
    )
    if not train_ids:
        raise RuntimeError("game-level split produced zero training games — fetch more games")

    train_dataset = BCDataset(train_ids)
    train_loader: DataLoader[dict[str, Tensor]] = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
    )
    val_loader: DataLoader[dict[str, Tensor]] | None = None
    if val_ids:
        val_dataset = BCDataset(val_ids)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=training_config.batch_size, collate_fn=collate_examples
            )

    policy, metadata = build_model_for_config(model_config)
    metadata.training_config = training_config.model_dump()
    metadata.seed = training_config.seed
    optimizer = torch.optim.Adam(policy.parameters(), lr=training_config.learning_rate)

    action_type_weight = torch.ones(len(ACTION_TYPES))
    action_type_weight[ACTION_TYPES.index("noop")] = training_config.noop_downweight

    early_stopping = EarlyStopping(patience=training_config.early_stopping_patience)
    best_val_loss = float("inf")
    checkpoint_dir = Path(training_config.checkpoint_dir)

    for epoch in range(training_config.epochs):
        policy.train()
        total_train_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = _compute_loss(policy, batch, training_config, action_type_weight)
            # torch's own stub gap marks Tensor.backward as untyped, not ours to fix.
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            batch_size = batch["action_type"].shape[0]
            total_train_loss += loss.item() * batch_size
            total_examples += batch_size
        train_loss = total_train_loss / max(total_examples, 1)

        if val_loader is not None:
            val_loss, val_metrics = _evaluate(
                policy, val_loader, training_config, action_type_weight
            )
        else:
            val_loss, val_metrics = train_loss, {}

        log_epoch_metrics(epoch, train_loss, val_loss, val_metrics)
        best_val_loss = checkpoint_if_best(
            policy, metadata, val_loss, best_val_loss, checkpoint_dir
        )
        if val_loader is not None and early_stopping.step(val_loss):
            logger.info("early stopping at epoch %d", epoch)
            break

    if best_val_loss == float("inf"):
        # No validation set (e.g. too few games) — checkpoint the final
        # training-loss state instead of leaving no checkpoint at all.
        checkpoint_if_best(policy, metadata, train_loss, float("inf"), checkpoint_dir)

    return checkpoint_dir / "best.pt"
