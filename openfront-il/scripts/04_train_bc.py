#!/usr/bin/env python
"""Thin CLI entry point — all logic lives in
src/openfront_il/training/bc_trainer.py. See scripts/README.md.
"""

from openfront_il.training.bc_trainer import train
from openfront_il.training.config import load_model_config, load_training_config
from openfront_il.utils.paths import PROJECT_ROOT


def main() -> None:
    training_config = load_training_config(PROJECT_ROOT / "configs" / "training.yaml")
    model_config = load_model_config(PROJECT_ROOT / "configs" / "model.yaml")
    checkpoint_path = train(training_config, model_config)
    print(f"best checkpoint written to {checkpoint_path}")


if __name__ == "__main__":
    main()
