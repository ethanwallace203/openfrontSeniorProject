# features/

## Responsibility

Pure functions that turn one validated `Snapshot` (from `replay/`) into
training-ready tensors and labels: `(raw snapshot) -> (spatial tensor, flat
vector, action head labels, legality mask)`. Also owns walking the full
`data/snapshots/` directory and writing `data/processed/*.parquet`.

**No I/O side effects belong in the encoder functions themselves** —
`observation_encoder.py` and `action_encoder.py` are pure transforms, easy
to unit test with in-memory fixtures. All file I/O is isolated to
`dataset_builder.py`.

## Allowed to import / depend on

- Standard library, `numpy`, `pandas`/`pyarrow`, `pydantic` (for the
  `Snapshot` type from `replay/`).
- `src/openfront_il/replay/` (to consume `Snapshot`), `src/openfront_il/utils/`.

## Must NOT import from

- `models/`, `training/`, `evaluation/` — this is a hard rule. Feature
  encoding must never know about the network architecture that will
  consume its output, only about the flat tensor shapes it produces
  (documented below), so that either side can change independently.
- `engine/` or `ingestion/`.

## Design principle (carried over from prior art)

Compress only the large, high-dimensional part of the state — the map/tile
grid — through a learned spatial representation (the encoder architecture
itself lives in `models/spatial_encoder.py`; this package only produces the
*raw* tensor it will consume). Pass every small, exact fact **straight
through, uncompressed**: diplomacy bits, per-player scalars, units in
flight, the legality mask. Do not invent a learned encoding for small
discrete facts — it throws away exactness for no benefit at this scale.

## Contract

| File | Input | Output |
|---|---|---|
| `observation_encoder.py` | `(SnapshotHeader, TickSnapshot, viewer_small_id)` | `encode_spatial` -> `np.ndarray[len(SPATIAL_CHANNELS), H, W]`; `encode_flat` -> `np.ndarray[len(FLAT_FEATURES)]`; `encode_observation` returns both. Every tensor is from `viewer_small_id`'s perspective (owned-by-me/ally/enemy, not raw owner IDs) — see the module docstring for why. Channel/feature order lives in the `SPATIAL_CHANNELS`/`FLAT_FEATURES` list constants, never hardcoded elsewhere. |
| `action_encoder.py` | one raw intent dict (or `None` for "no action this window") + a `LegalityMask` | `encode_action` -> an `EncodedAction` (multi-head: `action_type`, `unit_type`, plus raw — not one-hot — `target_x/y`, `target_player_small_id`, `amount`), or `None` for an intent type excluded from the modeled surface (`UNMODELED_INTENT_TYPES`); `encode_legality_mask` -> `np.ndarray[len(ACTION_TYPES)]` bool. See `docs/imitation_learning_plan.md` Section 2 for the full taxonomy and why tile/player targets aren't one-hot encoded. |
| `dataset_builder.py` | `data/snapshots/*.jsonl` (via `replay/loader.py`) | `data/processed/{gameId}.parquet` — one row per (snapshot, acting human, modeled action) example: `game_id`, `tick`, `client_id`, `small_id`, one column per `FLAT_FEATURES` entry, `action_type`, `unit_type`, `target_player_small_id`, `target_x`, `target_y`, `amount`, `legality_mask` (list<bool>). **Spatial tensors are not stored here** — each row's `(game_id, tick)` is enough to re-derive one via `encode_spatial()` from `data/snapshots/` on demand; storing the decoded `(C,H,W)` array again would multiply the storage cost `engine/bridge/README.md` already measured, for zero benefit given `encode_spatial()` is cheap and pure. This is a real, deliberate efficiency tradeoff for a prototype (per-example random access into a snapshot JSONL isn't fast) — see the module docstring. |
| `example_loader.py` | a list of `gameId`s | `ExampleTable`: reads those games' `data/processed/*.parquet` rows and reconstructs each one's spatial tensor via `encode_spatial()`, in plain numpy (no torch). Exists so `training/`'s `BCDataset` and `evaluation/`'s inference loop share exactly one implementation of "load processed rows + rebuild spatial" instead of duplicating it or one importing the other (neither is allowed to import the other — see their READMEs). Same scale tradeoff as `dataset_builder.py`: loads referenced games' full snapshot files into memory up front. |

## Class imbalance and split strategy

Handled in `training/`, not here — this package's job stops at producing a
faithful, lossless-where-possible tensor representation of each tick. See
`docs/imitation_learning_plan.md` for the class-imbalance and
train/val/test split decisions that consume this package's output.

## Style

Same as `src/openfront_il/README.md`, with one addition: every public
function here should be testable with a hand-constructed `Snapshot` fixture
and no disk access — if a test needs a real file, it belongs in
`dataset_builder.py`'s tests, not the encoders'.
