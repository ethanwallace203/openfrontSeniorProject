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
| `bc_trainer.py` | `split_game_ids()` (deterministic, game-level, per `docs/imitation_learning_plan.md` Section 4), `BCDataset` (reads `data/processed/*.parquet` plus `data/snapshots/` for on-demand spatial reconstruction — see above), `masked_cross_entropy()`, and `train()`: the loop itself — forward pass through `models.PolicyNetwork`, per-head cross-entropy with legality masks applied before the loss (`action_type`, and the `tile_target`/`target_player` argument heads, which are teacher-forced on the true action type and only trained on rows whose action type takes that argument), backward, optimizer step, checkpointing the best epoch via `callbacks.py` |
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

## GPU training (CUDA)

`bc_trainer.train()` uses CUDA automatically when available (`device:
auto` in `configs/training.yaml`; `cuda` to fail loudly instead of falling
back to CPU), with bf16 autocast (`mixed_precision`) and `num_workers`
background processes rebuilding examples. GPU-trained checkpoints load
on CPU (`load_checkpoint` maps to CPU).

Measured on an RTX 3090 (24GB) with `configs/model.yaml`'s defaults, on
synthetic Europe-size (2904x1672) inputs:

| | per example |
|---|---|
| CPU training (12 threads) | ~737 ms |
| 3090 training, fp32, batch 16 | ~47 ms (17.8GB peak) |
| 3090 training, bf16, batch 16 | ~24 ms (13.0GB peak) |
| 3090, batch 32 | out of memory, fp32 and bf16 |
| Rebuilding one example from snapshots (one CPU core) | ~117 ms |

The last row is why `num_workers` matters: at ~117 ms to rebuild vs ~24
ms to train, fewer than ~5 workers leave the GPU idle. Smaller maps are
proportionally cheaper on every row.

### RAM

Measured on a synthetic Europe-size game (300 ticks, ~269KB per snapshot
line — in the range of real games), `batch_size: 16`, 8 workers:

| | RAM |
|---|---|
| Loading a whole game into memory (the old `ExampleTable`) | ~2.6x the snapshot file's size — ~87GB for a 100-game, ~33GB dataset |
| `ExampleTable` now (on-demand reads from a byte-offset index) | ~2MB per game + ~20MB once |
| Peak total during an epoch (main + 8 workers), `uint8` spatial | ~16GB |
| Same, if spatial were float32 | ~45GB |

The epoch peak is dominated by decoded batches waiting in the prefetch
queue (`num_workers` x 2 batches), not by the dataset, so it stays
roughly constant as the number of games grows. To lower it: fewer
`num_workers` or a smaller `batch_size`. Workers are still forked
explicitly on Linux (see `_make_loader`) so none of them pickles a copy
of the table.

### Real dataset (96 public games, 70 maps)

2.12M examples, **82% noop**. Three measures keep an epoch practical:

- **Noop subsampling** (`noop_keep_fraction`, default 0.1): each epoch
  trains on every real action plus a fresh random 10% of noops (~26% of
  rows); kept noops are weighted 1/0.1 in the loss so the learned noop
  rate stays the true human one. Validation uses one fixed subset so its
  loss is comparable across epochs.
- **Map-grouped batches** (`MapGroupedBatchSampler`): every batch comes
  from one map size, so nothing is padded. Mixed-map batches padded to
  the batch's tallest height *and* widest width — up to 11.4M tiles here,
  bigger than any real map — which ran the 3090 out of memory.
- **Non-persistent validation workers**, and `scripts/04_train_bc.py`
  defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so varying
  batch shapes don't fragment GPU memory.

Measured on the 3090, `batch_size: 8`, 8 workers, 1.69M training
examples:

| | Before | After |
|---|---|---|
| One epoch | ~15 h | **~2.5 h** (49 ex/s steady, GPU 96% busy) |
| Peak GPU memory | out of memory at batch 16; ~22GB at 8 | ~14GB |
| Peak system RAM (whole machine) | ~38GB + swap | ~27GB, no swap |

`log_every_batches` prints batches done, running loss, examples/sec and
the epoch's ETA, so a multi-hour epoch shows progress.
