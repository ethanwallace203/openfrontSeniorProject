# data/

## Responsibility

Holds every artifact the pipeline produces, organized as **immutable
pipeline stages**. Each subdirectory is written by exactly one stage of the
pipeline (see [ARCHITECTURE.md](../ARCHITECTURE.md)) and read by the next.
Nothing in `data/` is hand-edited.

## Stages (in order)

| Dir | Written by | Read by | Format |
|---|---|---|---|
| `raw/` | `engine/bridge/src/fetchGame.ts` | `replayGame.ts` | one `{gameId}.json` per game, verbatim API response |
| `snapshots/` | `engine/bridge/src/replayGame.ts` (via `snapshotWriter.ts`) | `src/openfront_il/replay/loader.py` | one `{gameId}.jsonl` per game, per-tick state+action |
| `processed/` | `src/openfront_il/features/dataset_builder.py` | `src/openfront_il/training/bc_trainer.py` | `*.parquet`, one row per (state, action) example |

## Rules

- **Append-only / immutable once written.** No stage overwrites another
  stage's output, and no stage overwrites its own prior output for the same
  `gameId` — if you need to regenerate, delete the specific file(s) first
  and re-run; don't add silent overwrite logic to any writer.
- **Nothing in `data/` is committed to git** except `.gitkeep` placeholders
  (this is real, large, regenerable data). See the root `.gitignore`.
- **Every path here is defined once**, in
  `src/openfront_il/utils/paths.py` (Python side) and the equivalent
  constant in `engine/bridge/src/types.ts` (Node side). Do not hardcode
  `data/...` path strings anywhere else — import the constant instead.
- A file's presence in a stage directory is the only contract — don't infer
  pipeline progress from anything other than "does the expected output file
  for this `gameId` exist."

## Not this directory's job

- No transformation logic lives here — this directory holds data, not code.
- No cross-language format decisions get improvised ad hoc; the exact
  on-disk schema for `snapshots/` is documented in
  `src/openfront_il/replay/README.md` (Python's mirror of
  `engine/bridge/src/types.ts`) and any change must update both sides plus
  the drift-detection test described there.
