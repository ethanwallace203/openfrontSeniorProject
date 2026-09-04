# tests/

## Responsibility

Test layout mirrors `src/openfront_il/` 1:1 — for every module added under
`src/openfront_il/<package>/<module>.py`, there is a corresponding
`tests/unit/<package>/test_<module>.py` before that module is considered
done (see the project-wide rule: "every module in `src/` gets a matching
test in `tests/unit/` before moving to the next module").

TypeScript tests for `engine/bridge` live alongside it at
`engine/bridge/tests/` (a separate `vitest` suite, run via `npm test`), not
here — this directory is Python-only (`pytest`).

## Layout

```
tests/
├── conftest.py     # points OPENFRONT_DATA_DIR at an isolated temp dir for the whole run
├── fixtures/       # real, tiny, committed games shared across unit + integration tests
├── unit/           # one test module per src module, no I/O, no engine, no network
│   ├── ingestion/
│   ├── replay/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
└── integration/    # small end-to-end run on 1-2 tiny fixture games, marked slow
```

`tests/fixtures/` holds real, live-fetched games small enough to check in
(currently `USR3SJiv.json`/`.snapshots.jsonl`, ~1KB/14MB, a Singleplayer
game with bots; `vmVhD8B3.json`, ~43KB, a real Public 1v1) — the same
fixtures `engine/bridge/tests/fixtures/` uses on the TypeScript side,
duplicated per-language rather than read cross-directory so neither side's
tests depend on the other's file layout. `ingestion/` and `replay/`'s unit
tests both read from here (raw records and snapshot lines respectively)
since real data catches schema-drift bugs that hand-built fixtures can't.

## Rules

- `tests/unit/` tests never touch the network or the real engine — they
  use small in-memory fixtures (e.g. a hand-built `Snapshot` object) or the
  real, tiny files in `tests/fixtures/`. They also never touch the
  project's real `data/` directory: `tests/conftest.py` points
  `OPENFRONT_DATA_DIR` at an isolated temp directory for the whole test
  session before any test module imports `utils.paths`.
- `tests/integration/` tests are allowed to run a real, tiny end-to-end
  slice (1-2 fixture games small enough to check into the repo or cache
  once) through multiple pipeline stages. Mark these `@pytest.mark.slow`
  (registered in `pyproject.toml`) so `pytest -m "not slow"` gives a fast
  default loop, and CI can run the full suite separately.
- Integration tests must not depend on live network access to
  `api.openfront.io` — the fixture games are pre-fetched once and committed
  (or cached), not fetched at test time.
- Follow the same dependency-direction rules as `src/`: a unit test for
  `features/` should not need to import anything from `training/` or
  `evaluation/` to construct its fixtures.
