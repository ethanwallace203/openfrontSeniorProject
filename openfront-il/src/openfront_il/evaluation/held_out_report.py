"""Loads a trained checkpoint and a held-out slice of ``data/processed/``,
runs inference, computes ``metrics.py``'s functions, and writes a markdown
report.

Per this prototype's actual success criteria
(``docs/imitation_learning_plan.md`` Section 5), the report is built to
make two things legible: does the model beat a trivial baseline on the
*non-noop* examples specifically (not raw accuracy, which a noop-only
predictor can win by default on an imbalanced distribution), and did the
checkpoint load through the same ``checkpoint_format.py`` a later RL stage
would use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from openfront_il.evaluation import metrics
from openfront_il.features.action_encoder import ACTION_TYPES, UNIT_TYPES
from openfront_il.features.example_loader import ExampleTable
from openfront_il.models.checkpoint_format import CheckpointMetadata, load_checkpoint
from openfront_il.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceResults:
    action_type_predictions: NDArray[np.int64]
    action_type_labels: NDArray[np.int64]
    action_type_legality_masks: NDArray[np.bool_]
    unit_type_predictions: NDArray[np.int64]
    unit_type_labels: NDArray[np.int64]
    # Argument heads: one entry per example that takes that argument only.
    # `*_legal_counts` is how many choices the legality mask left open —
    # the denominator of the uniform-over-legal baseline.
    tile_target_correct: NDArray[np.bool_]
    tile_target_legal_counts: NDArray[np.int64]
    target_player_correct: NDArray[np.bool_]
    target_player_legal_counts: NDArray[np.int64]


@dataclass(frozen=True)
class HeldOutMetrics:
    checkpoint_path: Path
    format_version: int
    num_examples: int
    action_type_accuracy: float
    action_type_baseline_accuracy: float
    action_type_non_noop_accuracy: float
    action_type_legality_violation_rate: float
    unit_type_accuracy: float
    unit_type_baseline_accuracy: float
    tile_target_examples: int
    tile_target_accuracy: float
    tile_target_baseline_accuracy: float
    target_player_examples: int
    target_player_accuracy: float
    target_player_baseline_accuracy: float


def _masked_argmax(logits: torch.Tensor, legal: torch.Tensor, target: int) -> tuple[int, int]:
    """Argmax over the legal choices (with the true target never masked
    out, matching training's ``masked_cross_entropy``), plus how many
    choices that left open."""
    legal = legal.clone()
    legal[target] = True
    masked = logits.masked_fill(~legal, float("-inf"))
    return int(masked.argmax().item()), int(legal.sum().item())


def run_inference(
    checkpoint_path: Path, game_ids: list[str]
) -> tuple[InferenceResults, CheckpointMetadata]:
    """Loads ``checkpoint_path`` and runs its policy over every example in
    ``game_ids``'s processed parquet files.

    One example at a time (not batched) — simpler than wiring up a
    `DataLoader` purely for evaluation, at the cost of some speed; fine
    for a prototype's held-out set.

    The argument heads are conditioned on the *true* action type, the same
    teacher forcing training uses, so their accuracy reads as "given the
    right action, did it pick the right tile / player" — separate from
    whether ``action_type`` itself was right.
    """
    policy, metadata = load_checkpoint(checkpoint_path)
    policy.eval()
    table = ExampleTable(game_ids)

    action_type_predictions: list[int] = []
    action_type_labels: list[int] = []
    legality_masks: list[NDArray[np.bool_]] = []
    unit_type_predictions: list[int] = []
    unit_type_labels: list[int] = []
    tile_correct: list[bool] = []
    tile_legal_counts: list[int] = []
    player_correct: list[bool] = []
    player_legal_counts: list[int] = []

    with torch.no_grad():
        for index in range(len(table)):
            example = table.example(index)
            spatial = torch.from_numpy(example["spatial"]).unsqueeze(0)
            flat = torch.from_numpy(example["flat"]).unsqueeze(0)
            player_features = torch.from_numpy(example["player_features"]).unsqueeze(0)
            player_valid = torch.ones(player_features.shape[:2], dtype=torch.bool)
            action_type = torch.tensor([int(example["action_type"])])
            logits = policy(spatial, flat, player_features, player_valid, action_type=action_type)

            if int(example["target_x"]) >= 0:
                encoder = policy.spatial_encoder
                tile_logits = logits["tile_target"][0]
                cell_height, cell_width = tile_logits.shape
                target_cell = int(
                    encoder.cell_index(
                        torch.tensor(int(example["target_x"])),
                        torch.tensor(int(example["target_y"])),
                        cell_height,
                        cell_width,
                    ).item()
                )
                cell_legal = encoder.downsample_mask(
                    torch.from_numpy(example["tile_legal"]).unsqueeze(0)
                )[0]
                predicted, legal_count = _masked_argmax(
                    tile_logits.flatten(), cell_legal.flatten(), target_cell
                )
                tile_correct.append(predicted == target_cell)
                tile_legal_counts.append(legal_count)

            target_player = int(example["target_player"])
            if target_player >= 0:
                predicted, legal_count = _masked_argmax(
                    logits["target_player"][0],
                    torch.from_numpy(example["player_legal"]),
                    target_player,
                )
                player_correct.append(predicted == target_player)
                player_legal_counts.append(legal_count)

            action_type_predictions.append(int(logits["action_type"].argmax(dim=-1).item()))
            unit_type_predictions.append(int(logits["unit_type"].argmax(dim=-1).item()))
            action_type_labels.append(int(example["action_type"]))
            unit_type_labels.append(int(example["unit_type"]))
            legality_masks.append(np.asarray(example["legality_mask"], dtype=bool))

    table.close()
    results = InferenceResults(
        action_type_predictions=np.array(action_type_predictions, dtype=np.int64),
        action_type_labels=np.array(action_type_labels, dtype=np.int64),
        action_type_legality_masks=(
            np.stack(legality_masks)
            if legality_masks
            else np.zeros((0, len(ACTION_TYPES)), dtype=bool)
        ),
        unit_type_predictions=np.array(unit_type_predictions, dtype=np.int64),
        unit_type_labels=np.array(unit_type_labels, dtype=np.int64),
        tile_target_correct=np.array(tile_correct, dtype=bool),
        tile_target_legal_counts=np.array(tile_legal_counts, dtype=np.int64),
        target_player_correct=np.array(player_correct, dtype=bool),
        target_player_legal_counts=np.array(player_legal_counts, dtype=np.int64),
    )
    return results, metadata


def compute_metrics(
    checkpoint_path: Path, format_version: int, results: InferenceResults
) -> HeldOutMetrics:
    noop_index = ACTION_TYPES.index("noop")
    return HeldOutMetrics(
        checkpoint_path=checkpoint_path,
        format_version=format_version,
        num_examples=len(results.action_type_labels),
        action_type_accuracy=metrics.accuracy(
            results.action_type_predictions, results.action_type_labels
        ),
        action_type_baseline_accuracy=metrics.most_common_class_baseline_accuracy(
            results.action_type_labels, len(ACTION_TYPES)
        ),
        action_type_non_noop_accuracy=metrics.non_excluded_class_accuracy(
            results.action_type_predictions, results.action_type_labels, noop_index
        ),
        action_type_legality_violation_rate=metrics.legality_violation_rate(
            results.action_type_predictions, results.action_type_legality_masks
        ),
        unit_type_accuracy=metrics.accuracy(
            results.unit_type_predictions, results.unit_type_labels
        ),
        unit_type_baseline_accuracy=metrics.most_common_class_baseline_accuracy(
            results.unit_type_labels, len(UNIT_TYPES)
        ),
        tile_target_examples=len(results.tile_target_correct),
        tile_target_accuracy=metrics.fraction_true(results.tile_target_correct),
        tile_target_baseline_accuracy=metrics.uniform_over_legal_baseline_accuracy(
            results.tile_target_legal_counts
        ),
        target_player_examples=len(results.target_player_correct),
        target_player_accuracy=metrics.fraction_true(results.target_player_correct),
        target_player_baseline_accuracy=metrics.uniform_over_legal_baseline_accuracy(
            results.target_player_legal_counts
        ),
    )


def render_markdown_report(result: HeldOutMetrics, game_ids: list[str]) -> str:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Held-out evaluation report",
        "",
        f"Generated: {generated_at}",
        f"Checkpoint: `{result.checkpoint_path}` (format_version={result.format_version})",
        f"Held-out games: {len(game_ids)} ({', '.join(game_ids)})",
        f"Examples evaluated: {result.num_examples}",
        "",
        "Per `docs/imitation_learning_plan.md` Section 5, success here means "
        "beating the trivial baseline on **non-noop** examples specifically — "
        "not raw accuracy, which a noop-only predictor can win by default on "
        "an imbalanced distribution.",
        "",
        "## `action_type` head",
        "",
        f"- Accuracy: {result.action_type_accuracy:.4f}",
        f"- Most-common-class baseline: {result.action_type_baseline_accuracy:.4f}",
        f"- Non-noop accuracy: {result.action_type_non_noop_accuracy:.4f}",
        f"- Legality violation rate: {result.action_type_legality_violation_rate:.4f} "
        "(diagnostic only for the action types `docs/imitation_learning_plan.md` "
        "Section 2 lists as backed by a real legality signal)",
        "",
        "## `unit_type` head",
        "",
        f"- Accuracy: {result.unit_type_accuracy:.4f}",
        f"- Most-common-class baseline: {result.unit_type_baseline_accuracy:.4f}",
        "",
        "## Argument heads",
        "",
        "Scored only on examples whose action type takes that argument, "
        "conditioned on the true action type. The baseline is the expected "
        "accuracy of picking uniformly among the choices the legality mask "
        "leaves open.",
        "",
        f"### `tile_target` ({result.tile_target_examples} examples)",
        "",
        f"- Accuracy (exact feature-map cell): {result.tile_target_accuracy:.4f}",
        f"- Uniform-over-legal baseline: {result.tile_target_baseline_accuracy:.4f}",
        "",
        f"### `target_player` ({result.target_player_examples} examples)",
        "",
        f"- Accuracy: {result.target_player_accuracy:.4f}",
        f"- Uniform-over-legal baseline: {result.target_player_baseline_accuracy:.4f}",
        "",
    ]
    return "\n".join(lines)


def evaluate_and_report(
    checkpoint_path: Path, game_ids: list[str], output_path: Path
) -> HeldOutMetrics:
    """Runs inference, computes metrics, and writes the markdown report to
    ``output_path`` (a specific file path — the caller decides whether
    that's under ``docs/`` or elsewhere; see ``evaluation/README.md``).
    """
    results, metadata = run_inference(checkpoint_path, game_ids)
    result = compute_metrics(checkpoint_path, metadata.format_version, results)
    report = render_markdown_report(result, game_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("wrote held-out report to %s", output_path)
    return result
