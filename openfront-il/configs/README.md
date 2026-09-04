# configs/

## Responsibility

Every tunable value in the pipeline lives here, not hardcoded in `src/`.
This is a config-over-code project: if you find yourself writing a literal
learning rate, map name, date range, or architecture width inside a `.py`
file, it belongs in one of these YAML files instead, loaded through a typed
config class (see `src/openfront_il/training/config.py` for the pattern).

## Allowed to import / depend on

N/A — this directory holds data (YAML), not code. It is read by
`src/openfront_il/training/config.py` and by `scripts/*` entry points, and
by `engine/bridge` for `data.yaml`'s fetch/replay parameters.

## Contract

| File | Consumed by | Contents |
|---|---|---|
| `data.yaml` | `engine/bridge/src/fetchGame.ts`, `gameIdSource.ts` | which maps to include, date range, target game count, rate limit overrides |
| `model.yaml` | `src/openfront_il/models/` (via a config class, not read directly by `nn.Module`s) | architecture hyperparameters: spatial encoder channel widths, latent dim, policy head sizes |
| `training.yaml` | `src/openfront_il/training/config.py` | learning rate, batch size, epochs, seed, per-head loss weights, checkpoint dir, early-stopping patience |

## Rules

- No secrets here — API base URLs and rate limits are fine, credentials are
  not (there are none needed for the public OpenFront API today, but if
  that changes, secrets go in `.env`, never in a committed YAML file).
- A config file is data, not a place to leave comments about why a value
  was chosen — put reasoning in `docs/imitation_learning_plan.md` if it's
  a methodology decision, or a commit message if it's a one-off tuning
  choice.
