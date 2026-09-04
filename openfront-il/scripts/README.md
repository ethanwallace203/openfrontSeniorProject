# scripts/

## Responsibility

Thin CLI entry points only. Every script here should be a short amount of
argument parsing plus a call into `src/openfront_il/` (or, for the two
shell scripts, into `engine/bridge`) — no pipeline logic lives in this
directory itself.

## Allowed to import / depend on

- `src/openfront_il/*` (Python scripts), `npm`/`engine/bridge` CLI entry
  points (shell scripts).
- `configs/*.yaml` (to pass a config path through to the thing it's
  wiring up).

## Must NOT contain

- Any function longer than "parse args, call one function in `src/`,
  print/log the result." If a script grows real logic, that logic belongs
  in `src/openfront_il/` where it can be unit tested — move it there and
  leave the script as a thin wrapper.

## Contract

| Script | Wires up |
|---|---|
| `01_fetch_games.sh` | `engine/bridge/src/fetchGame.ts` via `npm run fetch` |
| `02_replay_games.sh` | `engine/bridge/src/replayGame.ts` via `npm run replay` |
| `03_build_dataset.py` | `src/openfront_il/features/dataset_builder.py` |
| `04_train_bc.py` | `src/openfront_il/training/bc_trainer.py` |
| `05_evaluate.py` | `src/openfront_il/evaluation/held_out_report.py` |

## Style

Python scripts here are the one place `print()` is allowed instead of
`utils/logging.py` (this is CLI-facing output, not library logging) — see
the exception noted in `src/openfront_il/README.md`. Shell scripts should
`set -euo pipefail` and fail loudly rather than swallowing a nonzero exit
from the Node side.
