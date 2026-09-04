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

`metadata.format_version` is `1` as of this writing (`FORMAT_VERSION` in
`checkpoint_format.py`). **Check it before trusting the checkpoint's
shape** — a future version bump means `SpatialEncoderConfig`/
`PolicyNetworkConfig`'s fields may have changed; this document should be
updated alongside any such bump (see `models/README.md`'s "Checkpoint
format stability" section, which makes that a hard rule for this project,
not just a suggestion).

`metadata.training_config` is an opaque dict (whatever
`TrainingConfig.model_dump()` produced) — useful for provenance
(learning rate, seed, loss weights actually used) but not something an RL
stage should parse structurally, since `training/config.py` can change
that shape freely without bumping `FORMAT_VERSION`.

## What to reuse vs. reinitialize

`policy` is a `models.policy_network.PolicyNetwork` with three parts:

| Component | Reuse or reinitialize | Why |
|---|---|---|
| `policy.spatial_encoder` (`SpatialEncoder`) | **Reuse.** Load its weights directly. | BC trained it on a real, if narrow, distribution of human-played board states. Its output is architecture-agnostic (a plain `latent_dim`-length vector via adaptive pooling — see `models/spatial_encoder.py`), so it drops into a different downstream head cleanly. |
| `policy.trunk`, `policy.action_type_head`, `policy.unit_type_head` | **Reuse as an initialization, expect to fine-tune.** | These learned to imitate the *behavior-cloning* action distribution, not to optimize reward — RL fine-tuning should adjust them, not freeze them, but starting from BC's weights rather than random init is the entire point of this prototype (faster convergence, avoiding a fully-random policy's early-training chaos). |
| A value head / critic | **Does not exist in this checkpoint — must be added new.** | BC never trained a value function (there was no reward signal, no bootstrapped return). Any PPO-style RL stage needs to attach a fresh value head (e.g. sharing `policy.spatial_encoder` + `policy.trunk`'s features, with its own new output layer) and train it from scratch. |

## Observation and action interfaces to reuse unchanged

An RL environment wrapping the real engine should produce observations
and interpret actions using the **exact same encoders** this prototype's
BC data pipeline used, not a reimplementation:

- `src/openfront_il/features/observation_encoder.py`:
  `encode_spatial(header, tick, viewer_small_id)` and
  `encode_flat(tick, viewer_small_id)` — same `SPATIAL_CHANNELS`/
  `FLAT_FEATURES` order the checkpoint's `SpatialEncoderConfig.in_channels`/
  `PolicyNetworkConfig.flat_dim` were sized against. **If these change,
  the checkpoint's first layer shapes are no longer valid** — either
  regenerate the checkpoint from scratch or write an explicit migration
  (e.g. zero-padding new channels into an old conv layer), don't silently
  feed a resized tensor into a shape-mismatched layer.
- `src/openfront_il/features/action_encoder.py`: `ACTION_TYPES`,
  `UNIT_TYPES`, `encode_action`, `encode_legality_mask`. An RL action
  space built independently (e.g. directly off the engine's raw intent
  union) would not line up with `policy`'s two head sizes
  (`action_type_count`, `unit_type_count` in `PolicyNetworkConfig`) and
  would make the reused heads meaningless.

## What this prototype does *not* hand off (known gaps, by design)

Per `docs/imitation_learning_plan.md` Section 2, the biggest scope cut in
this prototype is that **tile and player targets are not part of any
discrete head** — `EncodedAction.target_x`/`target_y`/
`target_player_small_id` are raw, unlearned fields on the label, never
fed into `PolicyNetwork`'s forward pass. This means:

- The reused `action_type`/`unit_type` heads tell an RL policy *what kind*
  of action a human would take and *what to build*, but say nothing about
  *where* or *whom*. An RL stage needs its own targeting mechanism from
  scratch — e.g. a spatial (H×W heatmap) head sharing
  `policy.spatial_encoder`'s feature map for tile targets, and a
  relative-ranking scheme (nearest/largest/etc. enemy) for player targets,
  since a fixed per-game player roster has no stable identity across
  games to learn a fixed class index for.
- `docs/imitation_learning_plan.md`'s legality-mask coverage table still
  applies: only some action types have a real backing signal in
  `LegalityMask`. An RL environment enforcing hard legality constraints
  (e.g. masking the policy's action distribution to true legal moves) will
  need a *complete* legality oracle from the live engine — this
  prototype's coarse, partially-backed mask was never meant to be that.

## Known scale limitations that don't affect the checkpoint's validity, but will affect training the *next* stage

- `features/example_loader.py` and `training/bc_trainer.py`'s `BCDataset`
  load each referenced game's entire snapshot file into memory. An RL
  self-play loop generating its own rollouts doesn't go through this path
  at all (it interacts with a live engine instance, not
  `data/snapshots/`), so this limitation is specific to *this prototype's
  training data loading* and is not inherited by the RL stage — noted
  here only so it isn't mistaken for a constraint on the checkpoint
  itself.
- `training/bc_trainer.py`'s `collate_examples` zero-pads spatial tensors
  to a batch's max `(H, W)` when games with different map sizes share a
  batch (see that function's docstring) — a training-time detail that
  does not affect the saved checkpoint's architecture or validity, only
  how it was trained.
