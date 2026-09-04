#!/usr/bin/env python
"""Thin CLI entry point — evaluates a trained checkpoint against the
held-out test split (same game-level split `training/bc_trainer.py` used,
recomputed deterministically from `configs/training.yaml`'s fractions —
see `docs/imitation_learning_plan.md` Section 4). Report-writing logic
lives in `src/openfront_il/evaluation/held_out_report.py`; splitting logic
lives in `src/openfront_il/training/bc_trainer.py`. See scripts/README.md.

Usage: python scripts/05_evaluate.py [checkpoint_path]
"""

import sys
from pathlib import Path

from openfront_il.evaluation.held_out_report import evaluate_and_report
from openfront_il.training.bc_trainer import split_game_ids
from openfront_il.training.config import load_training_config
from openfront_il.utils.paths import PROCESSED_DIR, PROJECT_ROOT


def main() -> None:
    default_checkpoint = PROJECT_ROOT / "checkpoints" / "best.pt"
    checkpoint_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_checkpoint
    if not checkpoint_path.exists():
        print(f"no checkpoint found at {checkpoint_path} - run scripts/04_train_bc.py first")
        return

    training_config = load_training_config(PROJECT_ROOT / "configs" / "training.yaml")
    game_ids = sorted(path.stem for path in PROCESSED_DIR.glob("*.parquet"))
    _, _, test_ids = split_game_ids(
        game_ids,
        val_fraction=training_config.val_fraction,
        test_fraction=training_config.test_fraction,
    )
    if not test_ids:
        print("no held-out test games available in data/processed/ - fetch and process more games")
        return

    output_path = PROJECT_ROOT / "docs" / "held_out_report.md"
    result = evaluate_and_report(checkpoint_path, test_ids, output_path)
    print(f"wrote report to {output_path}")
    print(
        f"action_type: accuracy={result.action_type_accuracy:.4f} "
        f"baseline={result.action_type_baseline_accuracy:.4f} "
        f"non_noop={result.action_type_non_noop_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
