# engine/

## Responsibility

This directory — and nothing outside it — is allowed to touch the real
OpenFront game engine. That means: booting it, feeding it a game record,
stepping its deterministic simulation, or reading any of its internal types
directly.

## Contents

- `openfront-ai/` — git submodule vendoring
  [`djmango/openfront-ai`](https://github.com/djmango/openfront-ai) (which
  itself vendors `openfrontio/OpenFrontIO`), pinned to a known-good commit.
  Treated as read-only third-party code: never edit files inside it. If a
  patch is ever unavoidable, fork upstream instead of editing the submodule
  in place.
- `bridge/` — our own TypeScript code that drives the vendored engine
  headlessly and writes its output to `data/`. See `bridge/README.md` for
  its contract.

## What may import from here

Only `engine/bridge/src/*`. Nothing in `src/openfront_il/` (Python) may
import from here, link against it, or call into it at runtime — the Python
side only ever reads the files `bridge/` has already written to `data/`.
`src/openfront_il/ingestion/orchestrator.py` is allowed to *invoke* the
Node CLI as a subprocess (see `ingestion/README.md`), but even that never
imports engine code directly — it shells out and checks exit codes.
`src/openfront_il/env/daemon_client.py` is the second such exception: it
runs `bridge/src/ilDaemon.ts` as a long-lived subprocess for live play and
talks to it over stdio (see `src/openfront_il/env/README.md`).

## Windows: `core.autocrlf` will corrupt the vendored binary map files

The vendored engine ships no `.gitattributes` marking `resources/maps/**/*.bin`
as binary. If your global git config has `core.autocrlf=true` (a common
Windows default/recommendation), checking out this submodule silently
converts lone `\n` bytes inside those binary terrain files to `\r\n`,
inflating their size and breaking `genTerrainFromBin`'s exact
`width*height` byte-count check with a confusing "buffer size incorrect"
error that gives no hint the real cause is a checkout-time corruption, not
a code bug. If you hit that error: set `core.autocrlf false` in **both**
`engine/openfront-ai` and `engine/openfront-ai/openfront`, then force a
real re-checkout (deleting a file first, or `git checkout-index -a -f`,
is required — `git checkout HEAD -- .` alone can no-op against git's
cached stat info and leave the corrupted bytes in place):

```bash
git -C engine/openfront-ai config core.autocrlf false
git -C engine/openfront-ai checkout-index -a -f
git -C engine/openfront-ai/openfront config core.autocrlf false
git -C engine/openfront-ai/openfront checkout-index -a -f
```

Verify with `git -C engine/openfront-ai/openfront cat-file -s HEAD:resources/maps/europe/map.bin`
(must read exactly `4855488`, i.e. `2904*1672`) against the actual
checked-out file size (`wc -c`) — they must match.

## Adding the submodule

```bash
git submodule add https://github.com/djmango/openfront-ai engine/openfront-ai
git submodule update --init --recursive
```

Pin to a specific commit deliberately (`git -C engine/openfront-ai checkout <sha>`)
and record the sha and the reason for any bump in this file's history —
replay determinism depends on the engine version never changing silently
underneath already-generated snapshots.

Pinned: `djmango/openfront-ai` @ `53f2e42eccccefa97ded2560e22afb9ae5e5956f`, whose
nested `openfront/` submodule (`djmango/OpenFrontIO`, a fork) is pinned at
`58c536b7a3c528c125890b40b1b213109b8b7014`.

## Two resolution gotchas when importing from the vendored engine

1. **Path aliases.** `engine/openfront-ai/openfront/src/**` uses bare
   `resources/*` and `src/*` import specifiers resolved via that project's
   own (separate) tsconfig. `tsx`, run from our project root, doesn't see
   that nested tsconfig — so our root `tsconfig.json` mirrors the same two
   `paths` entries, pointed at the submodule, exactly as
   `engine/openfront-ai`'s own root tsconfig does for its `datagen/`/`bridge/`.
   Only the vendored engine's *own* files should ever resolve through these
   aliases; our code always imports it via explicit relative paths.
2. **Runtime deps of `src/core` — and its `src/client` neighbors.** Despite
   being marketed as dependency-free, `openfront/src/core` imports `zod`,
   `dompurify`, `jose`, and `nanoid` as real npm packages. Worse, a few core
   files (`GameImpl.ts`, `Config.ts`, `AttackExecution.ts`, and others)
   import formatting/rendering helpers from `../../client/*` for display
   purposes — which, because ESM eagerly loads whatever a static import
   graph reaches, pulls in `colord`, `intl-messageformat`, and `lit` too,
   even though nothing we do ever renders anything. (One client file two
   hops in, `WorkerClient.ts`, has a Vite-only `import(...?worker&inline)`
   — but it's inside a function that's never called from a headless
   replay, so it never actually executes; only its enclosing module needs
   to *load*, which requires no Vite-specific handling.) Since the
   submodule has no `node_modules` of its own, all seven packages above are
   declared as direct dependencies in our own root `package.json`, pinned
   to the versions the vendored engine itself uses. **If the pinned commit
   is ever bumped, re-check `engine/openfront-ai/openfront/package.json`
   for version drift in these packages, and re-run the dependency-closure
   check below in case the new commit reaches something else.**

   To recompute the exact closure of bare npm imports reachable from
   `engine/bridge/src/*.ts` after a commit bump (rather than discovering
   missing packages one `MODULE_NOT_FOUND` at a time), walk the static
   import graph following relative and `resources/*`/`src/*` alias imports,
   and collect whatever bare specifiers remain — that's the exact
   dependency list this section documents.
