# docs/

## Responsibility

Prose documentation that doesn't belong inline in code: the imitation
learning methodology spec (written *before* training code, and treated as
the spec that code implements against) and the RL handoff document.

## Contract

| File | Purpose |
|---|---|
| `imitation_learning_plan.md` | BC methodology: what counts as expert data, action-space taxonomy and loss weighting, class-imbalance handling, train/val/test split strategy, success criteria. Written before `training/` code, treated as a spec. |
| `rl_handoff.md` | Exactly what a follow-on RL stage needs from this prototype: checkpoint format, observation/action encoder interfaces, which parts of the policy network to reuse vs. reinitialize. |

Evaluation reports produced by
`src/openfront_il/evaluation/held_out_report.py` may also be written here
(or to a configurable output directory — see that package's README) as
dated markdown files, not overwriting the two spec documents above.
