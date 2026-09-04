# utils/

## Responsibility

Only truly cross-cutting helpers used by multiple layers. Keep this folder
small and resist adding to it — if a helper is only used by one package,
it belongs in that package, not here.

## Allowed to import / depend on

- Standard library, `pyyaml` (for path/config constants).

## Must NOT import from

- Any other `src/openfront_il/` subpackage. `utils/` sits below everything
  else in the dependency graph — everything may import from it, it may
  import from nothing else in this project.

## Contract

| File | Provides |
|---|---|
| `logging.py` | one `get_logger(name)` factory wrapping stdlib `logging` with consistent structured formatting; used everywhere in `src/` instead of `print()` (scripts in `scripts/` may use `print()` for CLI-facing output) |
| `paths.py` | the single source of truth for `data/`'s layout: `RAW_DIR`, `REPLAYS_DIR`, `SNAPSHOTS_DIR`, `PROCESSED_DIR`, resolved from `OPENFRONT_DATA_DIR` (see `.env.example`) with a sane default. No other module constructs a `data/...` path string directly. |

## Style

Same as `src/openfront_il/README.md`. Anything added here should be
justified by at least two existing call sites in different subpackages —
otherwise it's premature abstraction.
