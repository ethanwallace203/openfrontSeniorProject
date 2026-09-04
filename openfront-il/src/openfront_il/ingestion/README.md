# ingestion/

## Responsibility

Bridges the Python pipeline to the Node fetch step. Invokes
`engine/bridge/src/fetchGame.ts` as a subprocess for a given list of
`gameId`s, checks its exit code, and validates that the raw files it was
supposed to produce actually landed in `data/raw/` and parse as the
expected shape.

This package does **not** call the OpenFront API itself and does **not**
touch the engine — it only shells out to the Node CLI and inspects its
output.

## Allowed to import / depend on

- Standard library (`subprocess`, `pathlib`, etc.), `pydantic`.
- `src/openfront_il/utils/` (paths, logging).

## Must NOT import from

- `features/`, `models/`, `training/`, `evaluation/` — this is the bottom
  of the dependency chain along with `replay/`.
- `engine/` code directly — only via subprocess invocation of its built CLI
  entry point, never a Python-side import of any TS/JS module.

## Contract

| File | Input | Output |
|---|---|---|
| `api_contract.py` | — | `pydantic` models mirroring the raw `GET /public/game/{gameId}` response shape (a sanity-check contract, not exhaustive — see the module docstring), used to validate `data/raw/{gameId}.json` after the Node step writes it |
| `orchestrator.py` | list of `gameId`s | `run_node_fetch()` invokes `npm run fetch -- <gameIds>`, retrying the whole subprocess invocation (not individual HTTP requests) up to `max_attempts` times if it exits nonzero; `validate_raw_files()` checks each expected raw file exists and validates against `api_contract.py`; `orchestrate_fetch()` composes both, skipping the Node step entirely for gameIds already in `data/raw/` |

`orchestrator.py` never re-implements HTTP fetching, rate limiting, or
retry-on-network-error logic — that all lives in `fetchGame.ts`. It only
retries the *subprocess invocation itself* if it exits nonzero, and only
validates files after the fact. Validation results are returned as a list
of `FetchResult(game_id, status, detail)` for every requested game — the
caller decides what to do with `"missing"`/`"invalid"` entries (log, retry,
skip); this module never raises on a single game's validation failure, so
one bad game in a batch doesn't abort the rest.

## Style

Same as the package-level conventions in `src/openfront_il/README.md`.
`api_contract.py` should be kept honest against the live API response
shape — if the API adds/removes a field, update this file and note it in
`docs/imitation_learning_plan.md`'s "what counts as expert data" section if
the change affects what skill signal is available.
