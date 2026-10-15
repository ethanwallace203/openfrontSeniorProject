# models/

## Responsibility

Network architecture definitions only: the spatial encoder, the multi-head
policy network, and the checkpoint schema a later RL stage will load. No
training loop, no data loading, no optimizer, no I/O beyond
loading/saving a checkpoint via `checkpoint_format.py`.

## Allowed to import / depend on

- `torch`, `numpy`, `pydantic` (for the checkpoint schema).
- `src/openfront_il/utils/` (logging only — never `paths.py`'s data
  directories; a model module should never know where `data/` lives).

## Must NOT import from

- `features/` — models consume tensors of a documented shape, not
  `Snapshot` objects or anything that requires re-deriving that shape from
  a raw snapshot. If a shape changes, `features/README.md`'s contract
  table is what both sides agree on, not a shared import.
- `training/`, `evaluation/` — a hard rule. Architecture must be usable
  (and testable) without a training loop existing.

## Contract

| File | Provides |
|---|---|
| `spatial_encoder.py` | `SpatialEncoder(nn.Module)`: small convnet, `spatial tensor [B,C,H,W] -> (feature map [B, conv_channels[-1], H', W'], latent [B, latent_dim])`, `H' = H // downsample_factor`. `in_channels`/`latent_dim`/`conv_channels` are constructor args, not imported from `features/`. The latent uses adaptive pooling so it's fixed-size for any real map's `(H, W)`; the feature map keeps location for the `tile_target` head. Also `downsample_mask()` (tile-level bool mask -> cell-level, aligned with the feature map) and `cell_index()` (tile `(x, y)` -> flat cell index). |
| `policy_network.py` | `PolicyNetwork(nn.Module)`: `(spatial [B,C,H,W], flat [B,F], player_features [B,N,P], player_valid [B,N], action_type [B] \| None) -> {"action_type", "unit_type", "tile_target" [B,H',W'], "target_player" [B,N]}` logits. The two argument heads are pointer-style (score a variable number of cells/candidates) and conditioned on `action_type` — see the module docstring and `docs/rl_handoff.md`. Head *sizes* (`action_type_count`, `unit_type_count`, `player_feature_dim`) are plain constructor ints — `training/` is what connects `features/`'s `len(ACTION_TYPES)`/`len(UNIT_TYPES)`/`len(PLAYER_FEATURES)` to them; this file only agrees on the head *names* as a convention, never imports `features/`. Legality masking is the caller's job, except padded candidate slots (`player_valid == False`). |
| `checkpoint_format.py` | `SpatialEncoderConfig`/`PolicyNetworkConfig`/`CheckpointMetadata` (pydantic): everything needed to rebuild the exact architecture plus training provenance (`training_config` is an opaque dict — this file must not need to change every time a training hyperparameter is added) and a `format_version` int. `build_policy_network()`, `save_checkpoint()`, `load_checkpoint()` — the latter two used by both `training/` and (later) the RL stage. |

## Checkpoint format stability

`checkpoint_format.py` is the single most important file for the RL
handoff (see `docs/rl_handoff.md`) — a later RL stage depends on being able
to load this checkpoint and know exactly which submodules (e.g. the
spatial encoder) are meant to be reused versus reinitialized (e.g. a value
head BC never trained). Any breaking change to this schema must bump its
version field and be called out explicitly in `docs/rl_handoff.md`.

## Style

Same as `src/openfront_il/README.md`. Every `nn.Module` here should be
constructible and runnable on random/dummy input with no dataset or
training loop present — that's the unit test bar for this package.
