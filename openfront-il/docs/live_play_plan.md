# Live Play Plan: BC Policy → Gymnasium Env → Rendered Game

How to get the behavior-cloned policy acting inside a running OpenFront game:
first headless (a Gymnasium env, for evaluation and the later RL stage), then
in the real rendered browser client.

## Architecture

```
                         ┌──────────────── Python ────────────────┐
                         │                                         │
 ilDaemon.ts (headless) ─┼─▶ DaemonClient ─▶ OpenFrontEnv (gym)    │
   bootGameFromStartInfo │        │              │                 │
   + buildTickSnapshot   │        ▼              ▼                 │
                         │   snapshot_schema ─▶ PolicyAgent        │
 ilPlay.ts (live lobby) ─┼─▶ live driver  ───▶  │ encode_* (as-is) │
   play.ts-style client  │                      │ policy forward   │
   + buildTickSnapshot   │                      │ ActionDecoder    │
                         │                      ▼                  │
                         │                 engine Intent JSON ─────┼─▶ back to Node
                         └─────────────────────────────────────────┘
```

Two rules hold throughout:

1. **One observation source.** Both Node entry points emit observations by
   calling the same `buildTerrainHeader` / `buildTickSnapshot` from
   `engine/bridge/src/snapshotWriter.ts` that produced the training data.
   No adapter between formats, so no train/serve skew in observations.
2. **No edits to the `engine/openfront-ai` submodule.** Our code imports
   only the vendored engine from it and owns everything else.

## Phase order at a glance

| # | Phase | Depends on | Est. |
|---|---|---|---|
| 0 | Decisions + setup | — | ½ day |
| 1 | Observation daemon (`ilDaemon.ts`) | 0 | 1 day |
| 2 | Python daemon client | 1 | ½–1 day |
| 3 | Action decoder (minimal) | 0 (parallel with 1–2) | 1–2 days |
| 4 | `PolicyAgent` (checkpoint → intents) | 2, 3 | 1 day |
| 5 | Gymnasium env + headless eval script | 4 | 1–2 days |
| 6 | Rendered live play (`ilPlay.ts`) | 4 | 2–3 days |
| 7 | Full action coverage + docs | 5 | 2–4 days |

Phases 1–2 and 3 touch disjoint code and can be done in parallel. A
headless agent playing games is reached at the end of phase 5 (~1 week);
watching it in the browser is phase 6.

---

## Phase 0 — Decisions and setup

- **Decision interval = 10 ticks.** `replayGame.ts` defaults
  `snapshotEvery` to 10 (`OPENFRONT_SNAPSHOT_EVERY`), and `play.ts` uses
  `DECISION_TICKS = 10`. Keep the env's ticks-per-step equal to whatever
  `snapshotEvery` the training data used; record it in a config, not a
  literal.
- **Spawn strategy.** Training snapshots are only written when
  `!game.inSpawnPhase()` (`replayGame.ts:315`), so the policy has never
  observed the spawn phase. Start with a **heuristic spawn** (random valid
  land tile, or the largest land region) and leave model-chosen spawns as a
  later experiment.
- **Dependencies.** Add `gymnasium` to `pyproject.toml`. No new Node deps.
- **Config.** Add `configs/env.yaml`: map, seed, bots, difficulty, nations,
  ticks per step, max episode ticks, troop-fraction defaults (phase 3).

**Done when:** config file exists and is loaded by a pydantic model, the
same way `training/config.py` does it.

## Phase 1 — Observation daemon (Node) — DONE

**Files:** `engine/bridge/src/ilDaemon.ts`, `engine/bridge/tests/ilDaemon.test.ts`,
`npm run daemon`. The protocol is documented in `ilDaemon.ts`'s header.

What was built, and where it departed from the original plan:

- **Does not use the submodule's `EnvSession`.** Its `bridge/common.ts`
  imports `openfront/src/client/webbot/obsCore`, which only exists in
  openfront-ai's patched engine fork, not the vendored engine here.
  Instead `replayGame.ts`'s boot sequence was split into an exported
  `bootGameFromStartInfo(info)`, so live games boot through the same code
  the replay tests hash-verify. `normalizeWinner` is exported too.
- **Game type defaults to Public** (matches `configs/data.yaml`). In
  Singleplayer the spawn phase only ends when the human spawns
  (`SpawnExecution.ts`), so an agent that never spawns would freeze the
  game. Public runs the fixed 300-tick spawn timer. `gameType` is a reset
  option (phase 6's private lobby will pass `"Private"`).
- **One game per process**, ops `reset` / `step` / `shutdown` only — no
  session IDs (gymnasium's `AsyncVectorEnv` already gives one process per
  env). `save_record` is deferred to phase 5, its first user.
- Intents are validated against the engine's `IntentSchema`; invalid ones
  come back in `rejected` (index + reason) while valid ones still apply.
- Seeds map deterministically to the gameID (`lobbyCreatedAt` fixed at 0),
  so a seed reproduces a game exactly.

**Engine behavior found while testing** (each pinned by a test):

- **`legal` has no agent entry until it spawns.** `buildTickSnapshot` only
  computes legality for living players. Phase 4 must treat a missing entry
  as "spawn or noop only" — not change the builder, which also produces
  the training data.
- **An agent that never spawns is eliminated**: the game runs on, the
  agent stays at 0 tiles with `alive: false`. Phase 2/5 should end the
  episode on `alive == false` once `tick.spawnPhase` is false.

## Open issue — spawn labels in the training data

Found during phase 1. `replayGame.ts` writes snapshots only when
`!game.inSpawnPhase()`, but spawn intents accumulate through the spawn
phase and are attached to the first snapshot after it (tick 310 in real
games). So every `spawn` label sits on a state where the spawn phase is
over, the player already owns its starting land, and
`encode_legality_mask` marks spawn illegal. Training doesn't NaN
(`masked_cross_entropy` un-masks the true label) but the model never sees
a spawn-phase state, and the `spawn_phase` flat feature is always 0. The
mask is also wrong on its own: players re-pick spawns (2–5 spawn intents
each), so spawn should be legal whenever `spawnPhase` is true, not only
at 0 tiles.

Effect on play: only the first ~300 ticks; phase 4's heuristic spawn
sidesteps it. Fix options:

- **A (recommended):** snapshot the spawn phase too, and make spawn legal
  iff `spawnPhase`. Requires re-replaying, rebuilding and retraining.
- **B:** treat `spawn` as unmodeled; keep the heuristic spawn permanently.

## Phase 2 — Python daemon client — DONE

**Files:** `src/openfront_il/env/daemon_client.py` (+ `env/README.md`),
`tests/unit/env/test_daemon_client.py`,
`tests/integration/test_daemon_process.py`. Done criterion met: 50 live
steps go through the unchanged `encode_observation` /
`encode_player_candidates`.

Decisions, and departures from the sketch above:

- **Documented architecture exception.** `env/` is the second package
  (after `ingestion/orchestrator.py`) allowed to run engine code, and
  only as a subprocess. `ARCHITECTURE.md`, `README.md` and
  `engine/README.md` updated.
- **Launches `node_modules/.bin/tsx` directly**, not `npm run daemon`:
  npm's banner goes to stdout and would corrupt the first reply, and npm
  is an extra process between us and the daemon.
- **stderr is drained by a thread into a 200-line tail** (not a log
  file): an unread stderr pipe fills and deadlocks the daemon, and the
  tail is attached to crash errors. Also forwarded to the logger at DEBUG.
- **stdout is read by a thread into a queue** so every request has a
  timeout (default 120 s). Any timeout, exit or malformed reply marks the
  client broken and kills the process — a late reply would otherwise be
  read as the answer to the next request.
- **Two error types:** `DaemonError` (refused request, client usable) and
  `DaemonCrashedError` (client unusable, carries the stderr tail).
- **Strict envelope models** (`ResetReply`, `StepReply`, `extra="forbid"`)
  wrap the existing snapshot models, so protocol drift fails loudly.
- **Defaults live only in `ilDaemon.ts`**; `reset()` sends only the
  options the caller set.

## Phase 3 — Action decoder (minimal)

**New module:** `src/openfront_il/features/action_decoder.py`, next to
`action_encoder.py`, which is its specification: every field
`encode_action` reads from a raw intent is a field the decoder must write.

Input: chosen `action_type`, `unit_type`, tile cell `(cy, cx)` on the
`H' × W'` grid, player candidate index, plus the current `TickSnapshot`,
terrain, the `small_ids` array from `encode_player_candidates`, and the
viewer's small ID. Output: one engine `Intent` dict, or `None` (noop).

Minimal coverage first: **`noop`, `spawn`, `attack`, `boat`, `build_unit`**.
Every other action type decodes to `None` and is counted.

- **Cell → tile.** Scan the 8×8 block (the encoder's
  `downsample_factor`) for the first tile that passes the same rule
  `encode_tile_target_mask` uses for that action/unit type (own land for
  builds, unowned land for spawn, shore/water for boats). If none passes,
  return `None`.
- **Candidate → player.** Map `small_ids[idx]` to the engine player ID via
  the tick's `players` list. Index 0 is terra nullius ("expand") → the
  attack intent's no-target form.
- **Unmodeled amounts.** Troops/gold have no head: use fixed fractions from
  `configs/env.yaml` (e.g. attack with 20% of current troops).

**Tests** (`tests/unit/features/test_action_decoder.py`): **round trip.**
For every intent in `tests/fixtures/USR3SJiv.snapshots.jsonl` of a covered
type, `decode(encode(intent))` must reproduce the type, unit and target
player exactly, place the tile inside the same 8×8 cell, and differ only in
amounts. This test is the main defense against silently rejected intents.

**Done when:** the round-trip test passes for all five covered types.

## Phase 4 — PolicyAgent

**New module:** `src/openfront_il/agent/policy_agent.py`. Both the
Gymnasium env and live play use it.

- Load the policy with `load_checkpoint`, put it in eval mode, and fail fast
  if `metadata.format_version` is not the expected one.
- `act(header, tick, me) -> list[Intent]` follows the sampling order in
  `docs/rl_handoff.md`:
  1. encode the observation and player candidates;
  2. sample `action_type` with `encode_legality_mask` applied;
  3. re-run with `action_type=a` for the conditioned heads;
  4. sample tile cell / candidate / unit type with their masks;
  5. hand the result to `ActionDecoder`.
- Support `greedy` (argmax) and `sample` (with temperature) modes. Use
  greedy for debugging and sampling for evaluation.
- Spawn phase: delegate to the phase-0 heuristic, not the model.
- Keep counters: action-type histogram, decode failures (`None` from the
  decoder), uncovered-type picks.

**Done when:** `act` returns intents for every tick of a fixture game
without exceptions.

## Phase 5 — Gymnasium env + headless evaluation

**New module:** `src/openfront_il/env/openfront_env.py`.

- `OpenFrontEnv(gym.Env)` wraps `DaemonClient`.
  - `reset(seed=...)` → observation dict, info.
  - `step(action)` takes the **structured action** (dict with
    `action_type`, `unit_type`, `tile_cell`, `target_player`), decodes it,
    sends it, and steps `ticks_per_step`.
  - `observation_space`: `Dict` of `spatial`, `flat`, `player_features`,
    `player_valid`, padded to a fixed max map size and max candidates
    (the policy already handles padding through `player_valid`).
  - `action_space`: `Dict` of `Discrete`s matching the four heads.
  - Reward: placeholder only (change in owned tiles plus a win bonus),
    documented as "for the RL stage to replace".
  - `terminated` on a win or the agent dying, `truncated` on max episode
    ticks.
  - `render_mode="rgb_array"`: color the tile-owner map. Do **not** try to
    embed the browser client here; phase 6 covers that.
  - Track illegal and failed decodes in `info`.
- Validate with `gymnasium.utils.env_checker.check_env`.

**New script:** `scripts/06_run_agent.py` (+ `make play-headless`). It runs
N episodes with `PolicyAgent` and writes a short report: survival ticks,
peak tiles, win rate, action-type histogram, decode-failure rate. Optionally
calls the daemon's `save_record` to keep a replayable game record for each
episode.

**Done when:** `check_env` passes and the script completes 10 episodes on a
small map (Onion or Pangaea, few bots).

## Phase 6 — Rendered live play

Goal: watch the agent play, or play against it, in the real browser client.

1. **Run OpenFront locally.** Start the dev server from
   `engine/openfront-ai/openfront` (see its README; the submodule also ships
   a `run-openfront` skill describing the steps). Create a private lobby in
   the browser.
2. **New file:** `engine/bridge/src/ilPlay.ts`. It's a copy of the
   submodule's `bridge/play.ts`, because its `PlaySession` isn't exported.
   The only change: where it emits
   `{"event":"obs", ...buildObs(...)}`, emit the snapshot header once at
   start and a `buildTickSnapshot` tick every decision interval instead.
   The websocket join, turn mirroring and intent sending stay unchanged.
3. **New script:** `scripts/07_play_live.py` (+ `make play-live`). It starts
   `ilPlay.ts --game <lobbyID> --host localhost:<port>`, feeds each tick to
   `PolicyAgent.act`, and writes back `{"intents": [...]}`.
4. Join the same lobby from your browser, start the game and watch.

**Fallback, if the local server is a hassle:** use `save_record` from
phase 5. It writes a standard game record, which the client may be able to
load in its replay viewer. Check this before relying on it.

**Done when:** the agent joins a local lobby, spawns, and visibly expands
and attacks in the browser.

## Phase 7 — Full coverage and documentation

- Extend `ActionDecoder` to the remaining `ACTION_TYPES` (alliances,
  donations, embargoes, `targetPlayer`, `upgrade_structure`,
  `move_warship`, cancels, `delete_unit`), one branch each, each covered by
  the round-trip test.
- Update `docs/rl_handoff.md`: the env is now the RL stage's entry point.
  Document the observation/action spaces, reward placeholder, and spawn
  heuristic.
- Add the new modules' READMEs (`src/openfront_il/env/`,
  `src/openfront_il/agent/`) in the same style as the existing layer
  READMEs.

## Risks to watch

| Risk | Symptom | Mitigation |
|---|---|---|
| Intents silently rejected by the engine | Agent "does nothing" | Round-trip test (phase 3); log decode failures; compare intent counts with engine-side unit/attack changes per step |
| Spawn phase out of distribution | Bad or no spawn | Heuristic spawn (phase 0) |
| Bot-vs-human distribution shift | Plays much worse than held-out metrics suggest | Expected. Measure it in phase 5 and report it; this motivates the RL stage |
| Map size varies per map | Shape errors in the gym spaces | Pad to a configured max; start with one map |
| JSON/stdio throughput | Slow episodes | Fine for evaluation and watching; optimize only for RL training |
