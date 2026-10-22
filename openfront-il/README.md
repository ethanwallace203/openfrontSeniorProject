# openfront-il

A prototype for the **imitation-learning (behavior cloning) stage** of an RL
pipeline for [OpenFront.io](https://openfront.io). It is not a finished RL
agent — it is a clean, modular base that a later RL fine-tuning stage
(PPO / self-play) can build on.

This is a research prototype. Prioritize clarity and correctness of the
pipeline shape over performance or scale.

## Pipeline

```
fetch (Node)  ─▶  replay (Node, real engine)  ─▶  snapshots (JSONL/binary)
                                                        │
                                                        ▼
                                        featurize (Python) ─▶ processed tensors
                                                        │
                                                        ▼
                                        train (Python, behavior cloning)
                                                        │
                                                        ▼
                                        evaluate (Python) ─▶ held-out report
                                                        │
                                                        ▼
                                        checkpoint ─▶ handed off to RL stage
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data-flow diagram and the
reasoning behind the Node/Python split, and
[docs/imitation_learning_plan.md](docs/imitation_learning_plan.md) for the BC
methodology this pipeline implements.

## Why two languages

- **Node/TypeScript** is the only place that can run the actual OpenFront
  game engine (`engine/`) — the deterministic simulation only exists in
  TS/JS. This is a hard constraint, not a stylistic choice.
- **Python** (`src/openfront_il/`) does everything downstream of raw
  snapshots: featurization, modeling, training, evaluation.
- The two sides talk **only through files on disk** (JSON/JSONL/Parquet
  under `data/`). No RPC, no shared memory, no cross-language calls at
  runtime. See `data/README.md` for the stage contract. The one exception
  is live play (`src/openfront_il/env/`), which needs interactive
  reset/step and drives `engine/bridge/src/ilDaemon.ts` as a subprocess —
  see `src/openfront_il/env/README.md`.

## Quickstart

```bash
# 1. Node side: install the engine bridge deps and pull the vendored engine
npm install
git submodule update --init --recursive

# 2. Python side
pip install -e ".[dev]"

# 3. Run the pipeline stage by stage (see Makefile for exact targets)
make fetch          # engine/bridge/src/fetchGame.ts -> data/raw/
make replay          # engine/bridge/src/replayGame.ts -> data/snapshots/
make build-dataset   # src/openfront_il/features/dataset_builder.py -> data/processed/
make train           # src/openfront_il/training/bc_trainer.py -> checkpoint
make eval            # src/openfront_il/evaluation/held_out_report.py -> docs report
```

Each stage is independently runnable and testable — see `tests/README.md`.

## Repository layout

| Path | Responsibility |
|---|---|
| `engine/` | Only place allowed to touch the real OpenFront engine |
| `data/` | Immutable, staged pipeline artifacts (raw → snapshots → processed) |
| `src/openfront_il/` | Pure Python: ingestion orchestration, featurization, models, training, evaluation |
| `configs/` | All hyperparameters and run configuration (no hardcoded values in `src/`) |
| `scripts/` | Thin CLI entry points that wire up `src/` and `engine/bridge` |
| `tests/` | Unit + integration tests, mirrors `src/` layout |
| `docs/` | Methodology write-up and RL handoff spec |

## Status

Prototype under active development. See section-by-section READMEs in each
directory for the contract of that layer before adding code to it.
