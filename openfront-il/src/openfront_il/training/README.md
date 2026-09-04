# training/

## Responsibility

Owns the behavior-cloning training loop: loads `data/processed/`, builds a
model from `models/`, runs supervised cross-entropy training per action
head (masking out illegal actions via the legality mask from
`features/action_encoder.py`), logs metrics, and checkpoints via
`models/checkpoint_format.py`.

## Allowed to import / depend on

- `src/openfront_il/models/`, `src/openfront_il/features/` (to load
  processed data and know tensor shapes / head names). This includes
  `features/example_loader.py`'s `ExampleTable`, which handles reading
  `data/processed/*.parquet` *and* reconstructing each example's spatial
  tensor on demand from `data/snapshots/` — see that module's docstring
  for why spatial tensors aren't stored in the parquet files directly.
  `bc_trainer.py`'s `BCDataset` is a thin `torch.utils.data.Dataset`
  wrapper around it. `evaluation/` uses the same `ExampleTable` for the
  identical reason, which is exactly why this logic lives in `features/`
  (below both) instead of being duplicated in each, or training reaching
  into `replay/` directly (which an earlier draft of this file assumed).
- `src/openfront_il/utils/` (paths, logging, seeding).
- `torch`, `pyyaml` (for `configs/training.yaml`), `pandas`/`pyarrow`
  (for `data/processed/*.parquet`).

## Must NOT import from

- `evaluation/` — training never depends on the evaluation package. Any
  metric needed *during* training (e.g. running val accuracy) is defined
  locally in `callbacks.py`, not imported from `evaluation/metrics.py`. If
  the two genuinely need to share a metric implementation, that
  implementation belongs in a lower layer both can import (e.g.
  `features/` or `utils/`) — do not create a `training/ → evaluation/`
  edge to get there.
- `ingestion/`, `replay/` — training reads `data/processed/` (plus
  `data/snapshots/` transitively, only through `features/example_loader.py`
  as described above) — it never imports `replay/` or `ingestion/` directly.

## Contract

| File | Responsibility |
|---|---|
| `config.py` | `TrainingConfig` from `configs/training.yaml` (lr, batch size, epochs, seed, per-head loss weights, checkpoint dir, split fractions) and `ModelConfig` from `configs/model.yaml` (architecture hyperparameters only — `in_channels`/`flat_dim`/head sizes are derived from `features/` constants at training time, not configured) |
| `bc_trainer.py` | `split_game_ids()` (deterministic, game-level, per `docs/imitation_learning_plan.md` Section 4), `BCDataset` (reads `data/processed/*.parquet` plus `data/snapshots/` for on-demand spatial reconstruction — see above), `masked_cross_entropy()`, and `train()`: the loop itself — forward pass through `models.PolicyNetwork`, per-head cross-entropy with the `action_type` head's legality mask applied before the loss, backward, optimizer step, checkpointing the best epoch via `callbacks.py` |
| `callbacks.py` | checkpointing cadence, early stopping, metric logging (via `utils/logging.py`, not `print`) |
| `seed.py` | one function, called at the start of every training script, that seeds `numpy`/`torch`/`random` from `config.seed` |

## Reproducibility

Every run must log the exact `configs/training.yaml` (plus
`configs/model.yaml`, `configs/data.yaml`) it used alongside its
checkpoint, so a checkpoint is never separated from the config that
produced it. `seed.py` is called once, at the very start of
`scripts/04_train_bc.py`, before any data loading or model construction.

## Style

Same as `src/openfront_il/README.md`. No hardcoded hyperparameters —
everything tunable comes from `configs/training.yaml` via `config.py`.

## Measured: CPU training on real maps is slow with `configs/model.yaml`'s
defaults

`SpatialEncoder`'s convolutions run over the *actual* map resolution —
2904x1672 for Europe, real games' full size, not a downsampled thumbnail
(see `models/spatial_encoder.py`; there is no resizing step anywhere in
this pipeline). Training end-to-end against real fetched games (not the
tiny synthetic tensors `tests/unit/models/` and the fast-path of
`tests/unit/training/test_bc_trainer.py`'s end-to-end test use) with
`configs/training.yaml`'s and `configs/model.yaml`'s out-of-the-box
defaults (10 epochs, `conv_channels: [16, 32, 64]`) did not complete a
single epoch within several minutes on CPU, across only 2 games and 689
total examples. This is a real, load-bearing scale limitation, not a
hypothetical one: expect either a GPU, a much smaller
`configs/model.yaml` (fewer/narrower conv layers), fewer `epochs`, or a
downsampling step this prototype does not currently implement, before
attempting a real training run — don't assume the shipped defaults are
tuned for CPU-only iteration speed.
