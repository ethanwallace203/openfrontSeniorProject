# evaluation/

## Responsibility

Loads a trained checkpoint and a held-out slice of `data/processed/`
(games never seen during training — see `docs/imitation_learning_plan.md`
for the split strategy), computes per-head accuracy and legality-mask
violation rate, and writes a short markdown report.

## Allowed to import / depend on

- `src/openfront_il/models/` (to load the checkpoint and run inference),
  `src/openfront_il/features/` (to know head names / shapes).
- `src/openfront_il/utils/`.
- `numpy`, `torch`, `pandas`.

## Must NOT import from

- `training/` — evaluation never imports the training loop. It only needs
  a saved checkpoint (`models/checkpoint_format.py`) and the held-out
  data; it does not need to know how training happened.
- `ingestion/`, `replay/` — reads `data/processed/`, same as `training/`,
  never raw snapshots.

## Contract

| File | Responsibility |
|---|---|
| `metrics.py` | pure functions: `(predictions, labels, legality_mask) -> {accuracy per head, legality-violation rate, ...}` |
| `held_out_report.py` | loads a checkpoint + held-out split, runs inference, calls `metrics.py`, writes a markdown report to `docs/` (or a configurable output dir) |

## Success criteria this package reports against

Per `docs/imitation_learning_plan.md`, this prototype's success criterion
is **not** "does it win games" — it's "does it predict expert actions
above a baseline, and is the checkpoint loadable by a follow-on RL stage."
`held_out_report.py`'s output should make both of those legible: report a
trivial baseline (e.g. most-common-action-per-head) alongside the model's
accuracy, and include a checkpoint-schema-version line confirming it loaded
via the same `checkpoint_format.py` a later RL stage would use.

## Style

Same as `src/openfront_il/README.md`.
