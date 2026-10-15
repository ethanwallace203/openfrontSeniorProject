#!/usr/bin/env python
"""Thin CLI entry point — all logic lives in
src/openfront_il/training/bc_trainer.py. See scripts/README.md.
"""

import os

# Must be set before torch initializes CUDA (i.e. before importing the
# trainer). Batches come from maps of many different sizes, and without
# expandable segments PyTorch's cached GPU memory fragments into chunks
# sized for earlier batches — a real run hit "CUDA out of memory" with
# 9GB reserved but unusable. setdefault: an explicit env value still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from openfront_il.training.bc_trainer import train  # noqa: E402
from openfront_il.training.config import load_model_config, load_training_config  # noqa: E402
from openfront_il.utils.paths import PROJECT_ROOT  # noqa: E402


def main() -> None:
    training_config = load_training_config(PROJECT_ROOT / "configs" / "training.yaml")
    model_config = load_model_config(PROJECT_ROOT / "configs" / "model.yaml")
    checkpoint_path = train(training_config, model_config)
    print(f"best checkpoint written to {checkpoint_path}")


if __name__ == "__main__":
    main()
