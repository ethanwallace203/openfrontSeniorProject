# Architecture

## Data flow

```
                         ┌─────────────────────────────────────────┐
                         │           api.openfront.io               │
                         └───────────────────┬───────────────────────┘
                                              │ GET /game/{gameId}
                                              ▼
                         ┌─────────────────────────────────────────┐
   Node / TypeScript     │  engine/bridge/src/fetchGame.ts           │
   (only place that      │  writes data/raw/{gameId}.json (immutable)│
   may touch the real    └───────────────────┬───────────────────────┘
   engine)                                   │
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  engine/bridge/src/replayGame.ts          │
                         │  boots engine/openfront-ai (submodule)    │
                         │  headlessly, replays deterministically,   │
                         │  ticks the sim, records (state, action)   │
                         └───────────────────┬───────────────────────┘
                                              │ per-tick snapshots
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  data/snapshots/{gameId}.jsonl            │
                         │  full board state + per-player scalars +  │
                         │  diplomacy + legality mask + action taken │
                         └───────────────────┬───────────────────────┘
════════════════════════════════ file boundary ════════════════════════════
                                              │
                                              ▼
   Python                 ┌─────────────────────────────────────────┐
   (everything            │  src/openfront_il/features/               │
   downstream of raw      │    observation_encoder.py (state->tensor) │
   snapshots)             │    action_encoder.py (action->labels+mask)│
                          │    dataset_builder.py                     │
                          └───────────────────┬───────────────────────┘
                                              │ parquet
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  data/processed/*.parquet                 │
                         └───────────────────┬───────────────────────┘
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  src/openfront_il/training/bc_trainer.py  │
                         │  spatial_encoder + policy_network,        │
                         │  cross-entropy per action head,           │
                         │  legality mask applied to loss            │
                         └───────────────────┬───────────────────────┘
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  checkpoint (models/checkpoint_format.py) │
                         └───────────────────┬───────────────────────┘
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  src/openfront_il/evaluation/             │
                         │    held_out_report.py -> docs/ report     │
                         └───────────────────┬───────────────────────┘
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │  docs/rl_handoff.md: what a later RL      │
                         │  stage reuses from this checkpoint        │
                         └─────────────────────────────────────────┘
```

## Why the Node/Python split

The OpenFront engine and its deterministic simulation exist only in
TypeScript/JavaScript — there is no Python port, and reimplementing the
simulation logic in Python would risk silent divergence from the real game
rules used by the actual matches we're imitating. So anything that needs to
*run the game* (fetching finished games, replaying them tick-by-tick,
recording the exact state and action at each tick) must run in Node.

Everything after that point — turning snapshots into tensors, defining and
training a policy network, evaluating it — has no dependency on the engine
itself, only on the *shape* of the data it already produced. That work is
better served by Python's ML ecosystem (PyTorch, NumPy, pandas/pyarrow).

## Why files-on-disk, not RPC

A prototype's failure modes should be easy to isolate. A file boundary means:

- Each stage can be re-run, inspected, or replaced independently without the
  other language runtime needing to be alive.
- Snapshot files can be diffed, spot-checked, or replayed into a debugger
  without any live process coordination.
- There's no shared-memory or process-lifecycle bug class to debug — a
  `.jsonl` file is either well-formed or it isn't.

The cost (throughput, unified schema you must keep in sync — see
`src/openfront_il/replay/README.md`) is acceptable at prototype scale and
is the same tradeoff the reference project's `bridge/` design converged on
for its persistent process, applied here in its simplest form (plain files,
no persistent process at all) since we don't need interactive step/reset.

## Design decisions carried over from prior art

- **Observation encoding**: compress only the large, high-dimensional state
  (the map/tile grid) through a learned spatial encoder. Pass small exact
  facts (diplomacy bits, per-player scalars, units in flight, legality
  masks) straight through, uncompressed. See
  `src/openfront_il/features/README.md`.
- **Action space**: structured, multi-head discrete — not a single scalar.
  Each head is masked by a legality mask recorded at replay time. See
  `src/openfront_il/features/README.md` and `models/README.md`.
- **Checkpoint reuse boundary**: the spatial encoder and action heads BC
  trains are meant to be reused by a later RL stage; anything BC never
  trains (e.g. a value head) is reinitialized. See `docs/rl_handoff.md`.

## Dependency direction (enforced, not aspirational)

```
ingestion, replay  ──X──▶  features, models, training, evaluation
features, models   ──X──▶  training, evaluation
training           ──X──▶  evaluation
```

Arrows marked X must never exist. If a change seems to need one, the
responsibility is in the wrong layer — restructure instead of adding the
import. See coding standards in the project root prompt / `pyproject.toml`
lint config for how this is enforced (import-linter style layering checks
are a natural addition once the layers stabilize).
