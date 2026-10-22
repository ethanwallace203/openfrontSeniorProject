# RL Handoff

What a follow-on RL fine-tuning stage (PPO / self-play) needs from this
imitation-learning prototype: the checkpoint format, the observation/
action interfaces to reuse unchanged, and which parts of the trained
policy to keep versus reinitialize.

## Checkpoint format

Load via `src/openfront_il/models/checkpoint_format.py`:

```python
from openfront_il.models.checkpoint_format import load_checkpoint

policy, metadata = load_checkpoint(Path("checkpoints/best.pt"))
```

`metadata.format_version` is `2` as of this writing (`FORMAT_VERSION` in
`checkpoint_format.py`). **Check it before trusting the checkpoint's
shape** — a future version bump means `SpatialEncoderConfig`/
`PolicyNetworkConfig`'s fields may have changed; this document should be
updated alongside any such bump (see `models/README.md`'s "Checkpoint
format stability" section, which makes that a hard rule for this project,
not just a suggestion).

### Version history

| Version | Change |
|---|---|
| 1 | `action_type` + `unit_type` heads only. `SpatialEncoder.forward` returned just the pooled latent. |
| 2 | `SpatialEncoder.forward` returns `(feature_map, latent)`. `PolicyNetwork` gained the `tile_target` and `target_player` pointer heads (plus an `action_type_embedding` that conditions them), and its `forward` takes player candidates. `PolicyNetworkConfig` gained `player_feature_dim`. **v1 checkpoints are rejected by `load_checkpoint`** — retrain rather than migrate; a v1 checkpoint has no argument-head weights to carry over. |

`metadata.training_config` is an opaque dict (whatever
`TrainingConfig.model_dump()` produced) — useful for provenance
(learning rate, seed, loss weights actually used) but not something an RL
stage should parse structurally, since `training/config.py` can change
that shape freely without bumping `FORMAT_VERSION`.

## The policy's interface

```python
logits = policy(
    spatial,           # [B, len(SPATIAL_CHANNELS), H, W]   encode_spatial()
    flat,              # [B, len(FLAT_FEATURES)]            encode_flat()
    player_features,   # [B, N, len(PLAYER_FEATURES)]       encode_player_candidates()
    player_valid,      # [B, N] bool — False for padded candidate slots
    action_type=None,  # [B] long — which action the argument heads should target for
)
```

| Head | Shape | Meaning |
|---|---|---|
| `action_type` | `[B, len(ACTION_TYPES)]` | What kind of action. |
| `unit_type` | `[B, len(UNIT_TYPES)]` | What to build/upgrade (only meaningful for `build_unit`/`upgrade_structure`). |
| `tile_target` | `[B, H', W']` | One logit per feature-map cell; `H' = H // policy.spatial_encoder.downsample_factor` (8 with the default config). Only meaningful for `TILE_TARGET_ACTION_TYPES`. |
| `target_player` | `[B, N]` | One logit per candidate from `encode_player_candidates()` (index 0 is always terra nullius, i.e. "expand into unowned land"). Only meaningful for `PLAYER_TARGET_ACTION_TYPES`. |

The argument heads are **conditioned on `action_type`** (an embedding
added to the trunk's output) — the right tile for a spawn isn't the right
tile for a City. During BC they were trained with the *true* action type
(teacher forcing). An acting policy should therefore sample in order:

1. Sample `a ~ action_type` (masked by `encode_legality_mask`).
2. Re-run (or reuse the trunk) with `action_type=a`.
3. If `a` takes a tile: mask `tile_target` with
   `policy.spatial_encoder.downsample_mask(encode_tile_target_mask(...))`,
   sample a cell, then pick a concrete tile inside that 8x8 cell (e.g. the
   first tile in it that passes the engine's own legality check).
4. If `a` takes a player: mask `target_player` with
   `encode_player_target_mask(...) & player_valid`, sample a candidate,
   and map it back through the `small_ids` array
   `encode_player_candidates()` returned.
5. `unit_type` only for build/upgrade; troop/gold amounts have no head
   yet (see gaps below).

For PPO, the action's log-probability is the sum of the log-probs of each
head actually used for that action type (the factorized form) — a head the
action didn't use contributes nothing.

## What to reuse vs. reinitialize

| Component | Reuse or reinitialize | Why |
|---|---|---|
| `policy.spatial_encoder` (`SpatialEncoder`) | **Reuse.** Load its weights directly. | As of v2 the `tile_target` loss backpropagates through its *feature map*, not only its pooled average, so its conv filters were trained to encode *where* things are (good spawn/build/boat cells), not just *what* the board looks like overall. Both its outputs — the `[C, H', W']` feature map and the pooled latent — are usable by new heads. |
| `policy.trunk`, `policy.action_type_head`, `policy.unit_type_head` | **Reuse as an initialization, expect to fine-tune.** | These learned to imitate the *behavior-cloning* action distribution, not to optimize reward — RL fine-tuning should adjust them, not freeze them, but starting from BC's weights rather than random init is the entire point of this prototype (faster convergence, avoiding a fully-random policy's early-training chaos). |
| `policy.action_type_embedding`, `policy.tile_key`/`tile_query`, `policy.player_key`/`player_query` (the argument heads) | **Reuse as an initialization, expect to fine-tune.** | Same reasoning as the action heads. These are what make the BC prior *usable* for RL: without them RL would have to learn targeting over thousands of cells from a sparse win/loss reward alone. |
| A value head / critic | **Does not exist in this checkpoint — must be added new.** | BC never trained a value function (there was no reward signal, no bootstrapped return). Any PPO-style RL stage needs to attach a fresh value head (e.g. on `policy.trunk`'s output, or on the pooled latent + flat features) and train it from scratch. |

## Observation and action interfaces to reuse unchanged

An RL environment wrapping the real engine should produce observations
and interpret actions using the **exact same encoders** this prototype's
BC data pipeline used, not a reimplementation:

- `src/openfront_il/features/observation_encoder.py`:
  `encode_spatial(header, tick, viewer_small_id)`,
  `encode_flat(tick, viewer_small_id)` and
  `encode_player_candidates(header, tick, viewer_small_id)` — same
  `SPATIAL_CHANNELS`/`FLAT_FEATURES`/`PLAYER_FEATURES` order the
  checkpoint's `SpatialEncoderConfig.in_channels`/
  `PolicyNetworkConfig.flat_dim`/`PolicyNetworkConfig.player_feature_dim`
  were sized against. **If these change, the checkpoint's first layer
  shapes are no longer valid** — either regenerate the checkpoint from
  scratch or write an explicit migration (e.g. zero-padding new channels
  into an old conv layer), don't silently feed a resized tensor into a
  shape-mismatched layer.
- `src/openfront_il/features/action_encoder.py`: `ACTION_TYPES`,
  `UNIT_TYPES`, `TILE_TARGET_ACTION_TYPES`, `PLAYER_TARGET_ACTION_TYPES`,
  `encode_action`, `encode_legality_mask`, `encode_tile_target_mask`,
  `encode_player_target_mask`. An RL action space built independently
  (e.g. directly off the engine's raw intent union) would not line up
  with `policy`'s head sizes or with which head each action type uses,
  and would make the reused heads meaningless.

## Known gaps (by design, for now)

- **No troop/gold amount head.** `EncodedAction.amount` is on the label
  but never predicted. Attacks, boats and donations need one; a small
  regression or binned head on the trunk (conditioned on action type like
  the argument heads) is the natural addition.
- **No unit pointer.** `upgrade_structure`, `delete_unit`, `cancel_boat`
  and `move_warship`'s `unitIds` refer to specific units, which no head
  selects. A pointer over the viewer's unit list (same shape as
  `target_player`) would cover them.
- **Cell, not tile, precision.** `tile_target` picks an 8x8-tile cell;
  choosing the exact tile is left to play-time code. A coarse-to-fine
  refinement (a second head over the tiles inside the chosen cell) is the
  upgrade path if that proves too lossy.
- **Small receptive field.** Three 3x3 conv layers see roughly 22 tiles of
  context per cell, so `tile_target`'s judgments are local. Dilated
  convs, more layers, or the vendored `engine/openfront-ai` spatial
  autoencoder latent are the options for longer-range context.
- **Player targets are PlayerIDs.** To emit a player-targeted intent,
  map the chosen candidate's smallID back to that player's
  `PlayerScalars.id` (snapshot format v2) — never its `clientID`, which
  differs even for humans and is null for nations/tribes.
- **Coarse legality.** `docs/imitation_learning_plan.md`'s legality-mask
  coverage table still applies, and `encode_tile_target_mask` is a
  necessary condition only (no build-spacing, port-adjacency or range
  checks). An RL environment enforcing hard legality constraints needs a
  *complete* legality oracle from the live engine — this prototype's
  masks were never meant to be that.

## Known scale limitations that don't affect the checkpoint's validity, but will affect training the *next* stage

- `features/example_loader.py` reads snapshot ticks from disk on demand
  via a byte-offset index, so BC training memory no longer scales with
  dataset size. An RL self-play loop doesn't go through this path at all
  (it interacts with a live engine instance, not `data/snapshots/`).
- `encode_spatial` returns `uint8` (every channel is 0/1);
  `SpatialEncoder` casts to float itself, so an RL environment can pass
  either dtype.
- `training/bc_trainer.py`'s `collate_examples` zero-pads spatial tensors
  to a batch's max `(H, W)` when games with different map sizes share a
  batch (see that function's docstring) — a training-time detail that
  does not affect the saved checkpoint's architecture or validity, only
  how it was trained. Padded cells are always masked out of the
  `tile_target` loss.
