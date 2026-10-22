"""The supervised behavior-cloning training loop: loads
``data/processed/``, builds a ``PolicyNetwork``, trains with per-head
cross-entropy (legality-masked for ``action_type``, ``tile_target`` and
``target_player``), and checkpoints the best epoch via ``callbacks.py``.

The two argument heads are trained only on rows whose action type takes
that argument (``action_encoder.TILE_TARGET_ACTION_TYPES`` /
``PLAYER_TARGET_ACTION_TYPES``) — a noop has no tile, so teaching the
tile head anything from it would be noise. Both are teacher-forced: the
policy is conditioned on the *true* action type, so their losses measure
"given the right action, did it pick the right argument."
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from openfront_il.features.action_encoder import ACTION_TYPES, UNIT_TYPES
from openfront_il.features.example_loader import ExampleTable
from openfront_il.features.observation_encoder import (
    FLAT_FEATURES,
    PLAYER_FEATURES,
    SPATIAL_CHANNELS,
)
from openfront_il.models.checkpoint_format import (
    CheckpointMetadata,
    PolicyNetworkConfig,
    SpatialEncoderConfig,
    build_policy_network,
)
from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.training.callbacks import (
    EarlyStopping,
    checkpoint_if_best,
    log_epoch_metrics,
    log_progress,
)
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
        self.table = ExampleTable(game_ids)

    def __len__(self) -> int:
        return len(self.table)

    def close(self) -> None:
        self.table.close()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self.table.example(index)
        return {
            "spatial": torch.from_numpy(example["spatial"]),
            "flat": torch.from_numpy(example["flat"]),
            "action_type": torch.tensor(int(example["action_type"]), dtype=torch.long),
            "unit_type": torch.tensor(int(example["unit_type"]), dtype=torch.long),
            "action_type_legal": torch.from_numpy(example["legality_mask"]),
            "target_x": torch.tensor(int(example["target_x"]), dtype=torch.long),
            "target_y": torch.tensor(int(example["target_y"]), dtype=torch.long),
            "tile_legal": torch.from_numpy(example["tile_legal"]),
            "player_features": torch.from_numpy(example["player_features"]),
            "player_legal": torch.from_numpy(example["player_legal"]),
            "target_player": torch.tensor(int(example["target_player"]), dtype=torch.long),
        }


class MapGroupedBatchSampler(Sampler[list[int]]):
    """Yields batches of example indices where (1) noop examples are
    subsampled to ``noop_keep_fraction`` and (2) every example in a batch
    comes from a map of the same ``(height, width)``.

    Grouping by map size means ``collate_examples`` never pads: a batch is
    exactly one map's size. Mixing maps padded each batch to its tallest
    height *and* widest width — measured on the real 96-game dataset, a
    1976x3560 map with a 3200x2076 one padded to 11.4M tiles, larger than
    any real map, which ran a 24GB GPU out of memory and spent most of the
    GPU's time on padding.

    ``shuffle=True`` (training): a fresh random noop subset, shuffled
    within each map group, with batch order shuffled across groups —
    every epoch differs. ``shuffle=False`` (validation): the same
    deterministic subset and order every epoch, so validation loss stays
    comparable across epochs.
    """

    def __init__(
        self,
        table: ExampleTable,
        *,
        batch_size: int,
        noop_keep_fraction: float,
        shuffle: bool,
        seed: int,
    ) -> None:
        if not 0.0 < noop_keep_fraction <= 1.0:
            raise ValueError(f"noop_keep_fraction must be in (0, 1], got {noop_keep_fraction}")
        self.batch_size = batch_size
        self.noop_keep_fraction = noop_keep_fraction
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        rows = table.rows
        if len(rows) == 0:
            self._is_noop = np.zeros(0, dtype=bool)
            self._group = np.zeros(0, dtype=np.int64)
            return
        self._is_noop = rows["action_type"].to_numpy() == ACTION_TYPES.index("noop")
        shapes = {game_id: table.map_shape(game_id) for game_id in rows["game_id"].unique()}
        shape_of_row = rows["game_id"].map(shapes)
        # One integer group id per distinct map size.
        self._group = pd.factorize(shape_of_row)[0].astype(np.int64)

    def _select(self, rng: np.random.Generator) -> np.ndarray:
        keep = ~self._is_noop | (rng.random(len(self._is_noop)) < self.noop_keep_fraction)
        return np.flatnonzero(keep)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self._epoch if self.shuffle else 0))
        self._epoch += 1
        selected = self._select(rng)

        batches: list[list[int]] = []
        groups = self._group[selected]
        for group in np.unique(groups):
            members = selected[groups == group]
            if self.shuffle:
                members = rng.permutation(members)
            for start in range(0, len(members), self.batch_size):
                batches.append(members[start : start + self.batch_size].tolist())
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return iter(batches)

    def __len__(self) -> int:
        """Expected batch count — exact for ``shuffle=False``; for
        training it varies by a few batches epoch to epoch with the random
        noop subset, which is fine for progress/ETA logging."""
        total = 0
        for group in np.unique(self._group):
            in_group = self._group == group
            expected = (~self._is_noop[in_group]).sum() + self.noop_keep_fraction * (
                self._is_noop[in_group].sum()
            )
            total += math.ceil(expected / self.batch_size)
        return total


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
    out of the pool; not implemented here. ``tile_legal`` is padded with
    ``False`` the same way, so a padded cell is never a legal tile target.

    Player candidate counts also differ per example (players die, games
    differ in size), so ``player_features``/``player_legal`` are padded to
    the batch's max ``N``, and a ``player_valid`` mask marks the real
    candidates — ``PolicyNetwork`` uses it to keep padded slots out of
    the ``target_player`` softmax.
    """
    max_height = max(item["spatial"].shape[1] for item in batch)
    max_width = max(item["spatial"].shape[2] for item in batch)
    max_candidates = max(item["player_features"].shape[0] for item in batch)

    padded_spatial = []
    padded_tile_legal = []
    padded_player_features = []
    padded_player_legal = []
    player_valid = []
    for item in batch:
        spatial = item["spatial"]
        pad_bottom = max_height - spatial.shape[1]
        pad_right = max_width - spatial.shape[2]
        padded_spatial.append(functional.pad(spatial, (0, pad_right, 0, pad_bottom)))
        padded_tile_legal.append(
            functional.pad(item["tile_legal"], (0, pad_right, 0, pad_bottom), value=False)
        )

        candidate_count = item["player_features"].shape[0]
        pad_candidates = max_candidates - candidate_count
        padded_player_features.append(
            functional.pad(item["player_features"], (0, 0, 0, pad_candidates))
        )
        padded_player_legal.append(
            functional.pad(item["player_legal"], (0, pad_candidates), value=False)
        )
        player_valid.append(torch.arange(max_candidates) < candidate_count)

    return {
        "spatial": torch.stack(padded_spatial),
        "flat": torch.stack([item["flat"] for item in batch]),
        "action_type": torch.stack([item["action_type"] for item in batch]),
        "unit_type": torch.stack([item["unit_type"] for item in batch]),
        "action_type_legal": torch.stack([item["action_type_legal"] for item in batch]),
        "target_x": torch.stack([item["target_x"] for item in batch]),
        "target_y": torch.stack([item["target_y"] for item in batch]),
        "tile_legal": torch.stack(padded_tile_legal),
        "player_features": torch.stack(padded_player_features),
        "player_legal": torch.stack(padded_player_legal),
        "player_valid": torch.stack(player_valid),
        "target_player": torch.stack([item["target_player"] for item in batch]),
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


def tile_target_labels(
    policy: PolicyNetwork, logits: dict[str, Tensor], batch: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Selects the rows that have a tile target and flattens them for a
    per-cell cross-entropy.

    Returns:
        ``(row selector [B] bool, flat cell logits [R, H'*W'], cell
        targets [R], cell legality [R, H'*W'])`` for the ``R`` selected
        rows.
    """
    encoder = policy.spatial_encoder
    tile_logits = logits["tile_target"]
    cell_height, cell_width = tile_logits.shape[1:]
    has_target = batch["target_x"] >= 0
    targets = encoder.cell_index(
        batch["target_x"][has_target], batch["target_y"][has_target], cell_height, cell_width
    )
    cell_legal = encoder.downsample_mask(batch["tile_legal"][has_target])
    return has_target, tile_logits[has_target].flatten(1), targets, cell_legal.flatten(1)


def _argument_loss(logits: Tensor, target: Tensor, legal_mask: Tensor) -> Tensor:
    """``masked_cross_entropy``, or a zero loss when no row in the batch
    takes this argument (``cross_entropy`` over zero rows is ``NaN``)."""
    if target.numel() == 0:
        return logits.new_zeros(())
    return masked_cross_entropy(logits, target, legal_mask)


def resolve_device(requested: str) -> torch.device:
    """``"auto"`` -> CUDA when available, else CPU; ``"cuda"`` fails loudly
    if CUDA isn't available rather than silently training on CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def _autocast(device: torch.device, enabled: bool) -> AbstractContextManager[object]:
    """bf16 autocast on CUDA when ``enabled``; a no-op otherwise.

    bf16 rather than fp16: Ampere GPUs (e.g. a 3090) run it on tensor
    cores at the same speed, and its fp32-sized exponent range means no
    ``GradScaler`` is needed to avoid underflowing gradients. Loss
    functions (``cross_entropy``) are autocast to fp32 internally.
    """
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    # non_blocking pairs with the DataLoader's pin_memory to overlap the
    # host->GPU copy with compute; it's a plain copy on CPU.
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def _make_loader(
    dataset: BCDataset,
    training_config: TrainingConfig,
    *,
    shuffle: bool,
    device: torch.device,
    persistent: bool,
) -> DataLoader[dict[str, Tensor]]:
    """DataLoader with ``num_workers`` background processes rebuilding
    examples (decompressing each map is the CPU-bound bottleneck — see
    ``training/README.md``) while the model trains on the previous batch.

    On Linux workers are forked, so they inherit the parent's
    ``ExampleTable`` (rows + snapshot index, not the snapshots themselves —
    see ``features/example_loader.py``) instead of each unpickling a copy.
    Fork is requested explicitly rather than inherited: Python 3.14
    changed Linux's default start method to forkserver. (Python 3.12+ warns
    that forking a multi-threaded process "may lead to deadlocks"; that's
    the long-standing PyTorch DataLoader default on Linux and is safe as
    long as workers don't touch CUDA — they only run numpy decoding.)

    Batches come from ``MapGroupedBatchSampler`` (noop subsampling + one
    map size per batch). ``persistent`` keeps workers alive between
    epochs: right for the training loader, which runs continuously; wrong
    for validation, which runs briefly once per epoch — measured, its 8
    persistent workers sat idle holding ~7GB of RAM alongside the
    training workers for the whole run.
    """
    workers = training_config.num_workers
    use_fork = workers > 0 and sys.platform == "linux"
    batch_sampler = MapGroupedBatchSampler(
        dataset.table,
        batch_size=training_config.batch_size,
        noop_keep_fraction=training_config.noop_keep_fraction,
        shuffle=shuffle,
        seed=training_config.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_examples,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent and workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        multiprocessing_context="fork" if use_fork else None,
    )


def _compute_loss(
    policy: PolicyNetwork,
    batch: dict[str, Tensor],
    training_config: TrainingConfig,
    action_type_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    logits = policy(
        batch["spatial"],
        batch["flat"],
        batch["player_features"],
        batch["player_valid"],
        action_type=batch["action_type"],
    )
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

    _, tile_logits, tile_targets, tile_legal = tile_target_labels(policy, logits, batch)
    tile_loss = (
        _argument_loss(tile_logits, tile_targets, tile_legal)
        * training_config.tile_target_loss_weight
    )

    has_player = batch["target_player"] >= 0
    player_loss = (
        _argument_loss(
            logits["target_player"][has_player],
            batch["target_player"][has_player],
            (batch["player_legal"] & batch["player_valid"])[has_player],
        )
        * training_config.target_player_loss_weight
    )
    return action_loss + unit_loss + tile_loss + player_loss, logits


def _evaluate(
    policy: PolicyNetwork,
    data_loader: DataLoader[dict[str, Tensor]],
    training_config: TrainingConfig,
    action_type_weight: Tensor,
    device: torch.device | None = None,
) -> tuple[float, dict[str, float]]:
    device = device if device is not None else torch.device("cpu")
    policy.eval()
    total_loss = 0.0
    total_examples = 0
    correct_action_type = 0
    correct_unit_type = 0
    tile_examples = 0
    correct_tile = 0
    player_examples = 0
    correct_player = 0
    with torch.no_grad():
        for cpu_batch in data_loader:
            batch = _to_device(cpu_batch, device)
            with _autocast(device, training_config.mixed_precision):
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

            _, tile_logits, tile_targets, _ = tile_target_labels(policy, logits, batch)
            tile_examples += len(tile_targets)
            correct_tile += int((tile_logits.argmax(dim=-1) == tile_targets).sum().item())

            has_player = batch["target_player"] >= 0
            player_examples += int(has_player.sum().item())
            player_predictions = logits["target_player"][has_player].argmax(dim=-1)
            correct_player += int(
                (player_predictions == batch["target_player"][has_player]).sum().item()
            )
    if total_examples == 0:
        return 0.0, {
            "action_type_accuracy": 0.0,
            "unit_type_accuracy": 0.0,
            "tile_target_accuracy": 0.0,
            "target_player_accuracy": 0.0,
        }
    return total_loss / total_examples, {
        "action_type_accuracy": correct_action_type / total_examples,
        "unit_type_accuracy": correct_unit_type / total_examples,
        # Over only the rows that take each argument; 0.0 when there are none.
        "tile_target_accuracy": correct_tile / max(tile_examples, 1),
        "target_player_accuracy": correct_player / max(player_examples, 1),
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
        player_feature_dim=len(PLAYER_FEATURES),
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

    device = resolve_device(training_config.device)
    logger.info(
        "training on %s (mixed_precision=%s, num_workers=%d)",
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        training_config.mixed_precision and device.type == "cuda",
        training_config.num_workers,
    )

    train_dataset = BCDataset(train_ids)
    train_loader = _make_loader(
        train_dataset, training_config, shuffle=True, device=device, persistent=True
    )
    val_loader: DataLoader[dict[str, Tensor]] | None = None
    if val_ids:
        val_dataset = BCDataset(val_ids)
        if len(val_dataset) > 0:
            val_loader = _make_loader(
                val_dataset, training_config, shuffle=False, device=device, persistent=False
            )
    logger.info(
        "examples: %d train, %d val (before noop subsampling at keep=%.2f); "
        "~%d train batches per epoch",
        len(train_dataset),
        len(val_dataset) if val_loader is not None else 0,
        training_config.noop_keep_fraction,
        len(train_loader),
    )

    policy, metadata = build_model_for_config(model_config)
    policy.to(device)
    metadata.training_config = training_config.model_dump()
    metadata.seed = training_config.seed
    optimizer = torch.optim.Adam(policy.parameters(), lr=training_config.learning_rate)

    action_type_weight = torch.ones(len(ACTION_TYPES), device=device)
    # Dividing by the keep fraction undoes the subsampling in expectation:
    # each kept noop stands in for 1/keep of them, so the model still
    # learns the true human noop rate (how often to act at all).
    action_type_weight[ACTION_TYPES.index("noop")] = (
        training_config.noop_downweight / training_config.noop_keep_fraction
    )

    early_stopping = EarlyStopping(patience=training_config.early_stopping_patience)
    best_val_loss = float("inf")
    checkpoint_dir = Path(training_config.checkpoint_dir)

    for epoch in range(training_config.epochs):
        policy.train()
        total_train_loss = 0.0
        total_examples = 0
        expected_batches = len(train_loader)
        epoch_start = time.monotonic()
        for batch_index, cpu_batch in enumerate(train_loader, start=1):
            batch = _to_device(cpu_batch, device)
            optimizer.zero_grad()
            with _autocast(device, training_config.mixed_precision):
                loss, _ = _compute_loss(policy, batch, training_config, action_type_weight)
            # torch's own stub gap marks Tensor.backward as untyped, not ours to fix.
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            batch_size = batch["action_type"].shape[0]
            total_train_loss += loss.item() * batch_size
            total_examples += batch_size
            if (
                training_config.log_every_batches
                and batch_index % training_config.log_every_batches == 0
            ):
                log_progress(
                    epoch,
                    batch_index,
                    expected_batches,
                    total_train_loss / total_examples,
                    total_examples,
                    time.monotonic() - epoch_start,
                )
        train_loss = total_train_loss / max(total_examples, 1)

        if val_loader is not None:
            val_loss, val_metrics = _evaluate(
                policy, val_loader, training_config, action_type_weight, device
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

    train_dataset.close()
    if val_loader is not None:
        val_dataset.close()
    return checkpoint_dir / "best.pt"
