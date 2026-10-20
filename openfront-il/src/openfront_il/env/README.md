# env/

## Responsibility

Runs the policy against a **live** game. Today that is `daemon_client.py`,
the Python side of `engine/bridge/src/ilDaemon.ts`; the Gymnasium env
(`docs/live_play_plan.md` phase 5) will live here too.

## Why this package runs a live process

Everywhere else, Python and Node talk only through files on disk (see
`ARCHITECTURE.md`). That works because the offline pipeline never needs
interactive reset/step. Live play does: the policy must see tick N before
it can choose the intents that shape tick N+1, so there is no file to read
ahead of time.

This package is therefore the second sanctioned exception, after
`ingestion/orchestrator.py`, and is held to the same rule: it runs engine
code **only by invoking the Node CLI as a subprocess**, never by importing
it. The observations it receives are built by the same
`buildTerrainHeader`/`buildTickSnapshot` code that wrote the training
snapshots, and are parsed into the same `replay/snapshot_schema.py`
models — so live and training observations come from one source.

## Allowed to import / depend on

- Standard library (`subprocess`, `threading`, `queue`, ...), `pydantic`.
- `src/openfront_il/replay/snapshot_schema.py`, `src/openfront_il/utils/`.

## Must NOT import from

- `engine/` — subprocess only.
- `training/`, `evaluation/`.

## Contract

| File | Talks to | Provides |
|---|---|---|
| `daemon_client.py` | `ilDaemon.ts` over stdio (protocol documented in that file's header) | `DaemonClient` (context manager): `reset(...) -> ResetReply`, `step(intents, ticks) -> StepReply`, `close()`. `parse_reply(line, model)` — the pure protocol parser. Errors: `DaemonError` (request refused; client still usable), `DaemonCrashedError` (process gone, hung or out of sync; client unusable, message carries the daemon's stderr tail) |

Behavior a caller must know (pinned by `engine/bridge/tests/ilDaemon.test.ts`):

- `StepReply.alive` is false both before the agent spawns and after it
  dies; use `tick.spawnPhase` to tell them apart. An agent that never
  spawns is eliminated when the spawn phase ends.
- `tick.legal` has no entry for the agent until it spawns.
- Invalid intents come back in `StepReply.rejected`; valid intents in the
  same step still apply.
- Set `OPENFRONT_LOG_LEVEL=DEBUG` to see the engine's own log lines.

## Tests

- `tests/unit/env/test_daemon_client.py` — `parse_reply` only, no process.
- `tests/integration/test_daemon_process.py` (`slow`) — a scripted fake
  daemon for crash / hang / stderr-flood handling, plus the real daemon
  end to end through the training featurizer.
