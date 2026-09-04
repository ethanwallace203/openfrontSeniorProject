# engine/bridge/

## Responsibility

Our own TypeScript code that drives the vendored engine
(`engine/openfront-ai/`) headlessly to produce the `data/` pipeline
artifacts. This is a thin, script-like layer — it has no game logic of its
own, only orchestration, serialization, and I/O around the real engine.

## Allowed to import / depend on

- The vendored engine at `../openfront-ai` (its public exports only — reach
  into internal/unexported modules only if the engine itself offers no
  public API for what's needed, and document why in a code comment at the
  import site).
- Node built-ins, `tsx` (dev runtime), and small utility deps declared in
  the root `package.json`.

## Must NOT import from

- Anything under `src/openfront_il/` (Python) — there is no cross-language
  import boundary; the contract is files in `data/`, nothing else.
- Nothing here should reach back into `configs/*.yaml` directly for
  hyperparameters that belong to training — this layer only cares about
  `data.yaml`-style fetch/replay configuration (which maps, which
  gameIds, rate limits), not model or training config.

## Files and their I/O contract

| File                | Reads                                                                  | Writes                                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fetchGame.ts`      | `GET {OPENFRONT_API_BASE}/public/game/{gameId}` per id from `gameIdSource.ts` | `data/raw/{gameId}.json` (never overwrites existing)                                                                                                                |
| `gameIdSource.ts`   | CLI args, a seed file, or `GET {OPENFRONT_API_BASE}/public/games` (paginated)      | list of candidate `gameId`s, in-memory only                                                                                                                                |
| `replayGame.ts`     | `data/raw/{gameId}.json`, boots `engine/openfront-ai`                  | `data/snapshots/{gameId}.jsonl` via `snapshotWriter.ts` (never overwrites existing; deletes a partial file if replay fails)                                                                                                                       |
| `snapshotWriter.ts` | called by `replayGame.ts` once for the header and once per snapshot tick | `data/snapshots/{gameId}.jsonl` (one JSON object per line)                                                                                                        |
| `types.ts`          | —                                                                      | shared TS types for the raw API shape and the snapshot schema; the source of truth mirrored (by hand, with a drift test) into `src/openfront_il/replay/snapshot_schema.py` |

## Snapshot format decision

**JSON Lines**, one object per tick, per game, at
`data/snapshots/{gameId}.jsonl`. The first line is a `SnapshotHeader`
(terrain, written once — it never changes tick to tick); every subsequent
line is a `TickSnapshot`.

Measured, not assumed: an early version base64-encoded the raw tile-state
buffer with no compression, and a single 470-tick _singleplayer_ game on a
real map (2904x1672 = ~4.85M tiles) produced a 620MB snapshot file. That's
not viable for a dataset of hundreds of multi-thousand-tick multiplayer
games. Both tile buffers (`SnapshotHeader.terrainGzipBase64` and
`TickSnapshot.tileStateGzipBase64`) are now gzipped before base64 encoding
— tile ownership is highly repetitive (most of a map is unclaimed/ocean at
any given tick), so this compresses well without adding a binary sidecar
format or a second file per game. Every other per-tick field (scalars,
diplomacy, legality mask, actions) stays plain JSON, per the observation
encoder's design principle in `src/openfront_il/features/README.md`: only
the large, high-dimensional tile grid is worth compressing; small exact
facts should stay exact and directly readable.

## Determinism

`replayGame.ts` must produce byte-identical snapshots for the same
`(gameId, engine commit)` pair across runs. Any source of nondeterminism
(wall-clock reads, `Math.random()` without a fixed seed, iteration over
unordered collections where order matters) is a bug in this layer, not an
acceptable limitation — file an issue against this directory rather than
working around it downstream.

### Resolved: the `purchasedTribeNames` desync

Verified against real, live-fetched games on the correct pinned engine
commit (see "Pinned" above). A Singleplayer/bots game
(`tests/fixtures/USR3SJiv.json`) replayed with all 5 hash checkpoints
matching from the start. A real Public 1v1 human game
(`tests/fixtures/vmVhD8B3.json`) initially **desynced at tick 10** — root
cause: `bootGame()` constructed `Executor` without its 4th argument
(purchased bot tribe names, `GameStartInfo.tribes`), which happened to be
absent from the first fixture but present (10 real names) in the second.
`TribeSpawner.spawnTribes()` only calls `this.random.shuffleArray(...)`
when `purchasedNames.length > 0` — its own source comment: "Guarded so
games without purchased names consume the PRNG exactly as before." Passing
an empty array where the live game passed 10 real names skipped that PRNG
draw, shifting every subsequent tribe's spawn tile and ID from that point
on. Fixed by threading `info.tribes?.map((t) => t.name)` through to
`Executor`'s constructor in `bootGame()` (`replayGame.ts`); the same fix
also added the previously-dropped `PlayerInfo.teamIndex` and
`Config.listed` arguments the current engine's own `GameRunner.init()`
passes, since both were the same class of bug (an argument this project's
boot sequence silently dropped) even though neither was confirmed to
affect hash parity for these two fixtures.

With the fix, `vmVhD8B3` (3,371 ticks, a real Public 1v1) replays with
**all 307 hash checkpoints matching**. `tests/integration.test.ts` pins
both fixtures as full-hash-parity regression tests.

The same debugging pass also found and fixed `loadFreshTerrain()` throwing
on `GameMapSize.Compact` — it was stubbed to reject that size entirely
rather than implement the coordinate/spawn-area scaling
`TerrainMapLoader.ts`'s own `loadTerrainMap()` does, before it was clear
how common Compact maps actually are among real games. `loadFreshTerrain()`
now mirrors that scaling directly (minus its layer-image loading, which is
a client-rendering concern this headless replay never touches).

**Stress-tested against 22 diverse real games** fetched live (1 to 109
players, Free-For-All and Team modes, ranked 1v1s, both map sizes):
**22/22 achieved full hash parity** — 223,386 total ticks, 22,165 hash
checkpoints verified, zero desyncs. This is strong evidence the fix
generalizes rather than happening to cover just the two committed
fixtures, though it isn't a guarantee against some _other_ dropped
argument surfacing on a game shape this sample didn't happen to include.

Lesson for future engine-commit bumps: `bootGame()`'s call sites
(`Config`, `PlayerInfo`, `Executor`, `createNationsForGame`,
`createGame`) should be diffed against the _current_ checked-out
`GameRunner.ts`'s `createGameRunner()`/`init()` — not against an older
reference implementation — since a missing constructor argument fails
silently (TypeScript won't complain about a parameter with a default
value) and only surfaces as a hash mismatch on whichever real games
happen to exercise the field that was dropped.

**A separate, unrelated scaling finding from the same stress test:** even
gzipped, per-tick tile-state buffers add up fast on large/long games — the
22-game sample produced 7.2GB of snapshots, with the largest single game
(a 3,101-snapshot, ~100-player FFA) at 1.4GB. `data/snapshots/` output is
not meant to be a permanent local artifact of routine testing; the two
fixtures in `tests/fixtures/` (14MB, 56MB) are the only ones worth keeping
around. This is a real capacity-planning number for
`src/openfront_il/features/dataset_builder.py` to reckon with once a
fetch run targets `configs/data.yaml`'s `targetGameCount: 100` — expect
tens of GB, not the low GB a naive per-fixture estimate would suggest, and
consider whether every 10-tick snapshot cadence is worth keeping at full
resolution versus subsampling further at featurization time.

## A `tsc --noEmit -p .` caveat

`replayGame.ts` imports engine-internal modules (`GameImpl.ts`, `Config.ts`,
and others) that themselves import from the vendored engine's `client/`
directory (e.g. `GameImpl.ts` imports `../../client/Utils` for a display
formatter). TypeScript resolves and type-checks anything transitively
imported by an included file regardless of `exclude`, so a project-wide
`npx tsc --noEmit -p tsconfig.json` will also surface pre-existing type
errors inside `engine/openfront-ai/openfront/src/client/**` (missing `lit`
types, DOM globals, etc.) — code that is reachable by the type checker but
never actually executed by this bridge. This is not something to fix here;
it's an artifact of consuming the vendored engine's source directly rather
than as a published, isolated package. `npm run lint` (ESLint) does not
have this problem — ESLint only reports diagnostics for the files given on
its command line even when its type-aware rules build a full program for
inference, so `engine/bridge/README.md`'s own `lint`/`format`/`test` npm
scripts stay clean. If you need to sanity-check `tsc` output yourself,
filter to `engine/bridge/`: `npx tsc --noEmit -p tsconfig.json | grep ^engine/bridge`.

## Style

TypeScript strict mode, explicit return types on every exported function,
no `any` without an inline comment justifying it, prettier + eslint (see
root `package.json`). Tests in `tests/` use `vitest` against tiny fixture
game JSON checked into `tests/fixtures/` (not `data/raw/`, which is
gitignored).
