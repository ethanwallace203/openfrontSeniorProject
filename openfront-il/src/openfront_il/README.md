# src/openfront_il/

## Responsibility

The entire Python side of the pipeline: everything downstream of the raw
snapshot files that `engine/bridge` produces. This package is **pure
Python** — it defines featurization, model architectures, training loops,
and evaluation. It never runs or calls the game engine.

## Allowed to import / depend on

- Standard library, and the third-party deps declared in `pyproject.toml`
  (`pydantic`, `numpy`, `pandas`/`pyarrow`, `torch`, `pyyaml`, `tqdm`).
- Files under `data/` (via `utils/paths.py`, never a hardcoded path).
- Config under `configs/*.yaml` (via `training/config.py` or equivalent —
  never hardcode a hyperparameter in a module body).

## Must NOT import from

- `engine/` — no Node process is ever started from here except the single,
  explicit subprocess call in `ingestion/orchestrator.py`, and even that
  call never imports engine *code*, only invokes its CLI and checks the
  exit code / reads the files it wrote.
- Nothing outside this package's own layering rules (see below).

## Subpackages, in dependency order

```
ingestion  →  replay  →  features  →  models
                                         │
                                         ▼
                                    training  →  evaluation
```

Each arrow is one-directional. `features/` and `models/` never import from
`training/` or `evaluation/`. `ingestion/` and `replay/` never import from
`features/` or anything later. If you find yourself needing a reverse
import, the responsibility is in the wrong layer — stop and restructure
rather than adding the import (this is a hard rule from the project spec,
not a style preference).

| Subpackage | Responsibility |
|---|---|
| `ingestion/` | Orchestrates the Node fetch step as a subprocess; validates raw files landed |
| `replay/` | Reads `data/snapshots/`; owns the on-disk snapshot schema (Python mirror of the TS types) |
| `features/` | Pure functions: snapshot → (spatial tensor, flat vector, action labels, legality mask) |
| `models/` | Network definitions only: spatial encoder, policy network, checkpoint schema |
| `training/` | The behavior-cloning training loop, config, callbacks, seeding |
| `evaluation/` | Metrics and the held-out evaluation report |
| `utils/` | Small cross-cutting helpers only (logging, path constants) |

## Style

Type hints on every function signature. Google-style docstrings on every
public function/class. `pydantic` models for anything crossing an I/O
boundary (files, subprocess output); plain `dataclasses` for in-memory
config that never touches disk directly. `black` + `ruff` for
formatting/linting, `mypy --strict` for type checking. No bare `except:`.
No `print()` — use `utils/logging.py`.
