# Imitation Learning Plan

This is the spec `src/openfront_il/features/`, `models/`, and `training/`
implement against. Written before those layers are finalized, per the
project's own ordering rule; revised in place if a real implementation
detail forces a change (revisions should be called out, not silently
absorbed).

## 1. What counts as "expert" data

**All finished `Public` games, unfiltered by skill, for this prototype.**

What the API actually exposes (confirmed by inspecting real fetched
records — see `docs/API.md` reference in the vendored engine, and
`ingestion/api_contract.py`):

- `GameRecord.info.config.rankedType`: `"unranked" | "1v1"`. A `1v1` tag is
  the only explicit skill/competitive signal the public API surfaces
  per-game; there is no ELO or rank field on the game record itself.
- Per-player win/loss is derivable from `GameRecord.info.winner` (a
  `["player" | "team" | "nation", id, ...]` tuple) cross-referenced against
  `info.players[].clientID` — so "did this player win" is available, but
  not a skill *rating*.
- `GET /public/player/:playerId` (not used by this prototype's fetch
  path) exposes per-player stats/history, which could support a future
  skill filter (e.g. win-rate threshold) without needing the API to expose
  ELO directly. Not implemented now — flagged as a natural extension in
  `docs/rl_handoff.md` if BC accuracy turns out to need cleaner supervision.

**Decision:** don't filter by skill for this prototype. Reasoning:
imitation learning from a broad population still teaches the "grammar" of
the game (legal-ish, locally sensible actions), which is the prototype's
actual bar (see Section 5) — and skill-filtering would cut an already
small fetched sample further for a project that isn't trying to reach
strong play. `configs/data.yaml`'s `gameType: "Public"` filter is the only
filter applied.

## 2. Action space taxonomy

The engine's real intent union has ~22 variants (confirmed by inspecting
`Schemas.ts`'s `IntentSchema` and cross-checking against real fetched
game records). Modeling a fully general action space — arbitrary tile
targets among millions of tiles, arbitrary player targets among a
variable, per-game player count — is out of scope for a prototype; the
design below picks the tractable core and is explicit about what's cut.

### Modeled action types (the `action_type` head)

One discrete head over:

```
noop, spawn, attack, boat, build_unit, upgrade_structure, move_warship,
cancel_boat, cancel_attack, delete_unit, allianceRequest, allianceReject,
allianceExtension, breakAlliance, donate_gold, donate_troops, embargo,
embargo_all, targetPlayer
```

`noop` is added by `dataset_builder.py`, not present in the raw intent
data — see Section 3 for why it has to exist as a real class.

**Deliberately excluded from `action_type`:** `emoji`, `quick_chat` (social/
cosmetic, not a territorial or economic decision), `mark_disconnected`
(system-generated, never a player's actual choice), and the four
lobby-admin intents (`kick_player`, `toggle_pause`, `update_game_config`,
`toggle_game_start_timer` — pre-game/moderation, not in-game strategy).
This mirrors a design choice the reference project's own BC label
normalization made independently (dropping "emoji, quick chat, lobby
admin" from its modeled surface) — cited here because it's good evidence
the exclusion is reasonable, not because it was copied uncritically.

### Secondary head: `unit_type` (conditional)

One discrete head over the engine's 16 `UnitType` values plus `"none"`,
used only when `action_type` is `build_unit` or `upgrade_structure`;
`"none"` otherwise. Nukes (`Atom Bomb`, `Hydrogen Bomb`, `MIRV`) are not a
separate action type in the engine — they're `build_unit` targeted at an
enemy tile — so this head is also what distinguishes "build a Missile
Silo" from "launch an Atom Bomb."

### Tile and player targets: pointer heads, not fixed classifiers

Spawn/boat/build/move-warship carry a target tile
(`TILE_TARGET_ACTION_TYPES`); attack/alliance/donate/embargo/targetPlayer
carry a target player (`PLAYER_TARGET_ACTION_TYPES`). Neither can be a
fixed-size classification head — one class per tile is millions of
classes and map-size-dependent, and a per-game player roster has no
stable class index across games. So `action_encoder.py` stores both raw
on the label (`target_x`/`target_y` decoded from the intent's TileRef;
`target_player_small_id`, with an `attack` on unowned land labeled as
terra nullius), and `models/policy_network.py` predicts them with
*pointer* heads that score a variable number of choices:

- **`tile_target`**: one logit per cell of the spatial encoder's feature
  map (1/8 resolution with the default three conv layers), from a 1x1
  conv scored against a query from the trunk. The label is the cell
  containing `(target_x, target_y)`.
- **`target_player`**: one logit per candidate player (terra nullius plus
  every other living player, described by `PLAYER_FEATURES` — relation,
  size, troop advantage, shared border, who's attacking whom), scored the
  same way. The label is the target's index among the candidates.

Both are conditioned on the action type and trained only on rows whose
action type takes that argument, with per-argument legality masks
(`encode_tile_target_mask`, `encode_player_target_mask`). Intent target
fields hold engine **PlayerIDs** (not clientIDs — the two differ even for
humans), resolved through `PlayerScalars.id` (snapshot format v2). One
known gap: targets are cell-precise, not tile-precise (an 8x8-tile cell;
play-time code picks a tile within it).

### Legality masking, and its real limit

`action_encoder.encode_legality_mask()` derives an `action_type`-level
mask from `LegalityMask` (`src/openfront_il/replay/snapshot_schema.py`).
Coverage is uneven because the upstream mask (computed in
`engine/bridge/src/snapshotWriter.ts`, already shipped and tested) tracks
per-player-target and per-unit-type legality, not a signal for every
action type:

- Backed by a real field: `attack`/`boat` (`attackablePlayerSmallIDs` /
  `canExpand`), `build_unit`/`upgrade_structure`
  (`buildableUnitTypes`), `allianceRequest`
  (`allianceRequestablePlayerSmallIDs`), `allianceReject`
  (`allianceAcceptablePlayerSmallIDs`), `breakAlliance`
  (`breakableAlliancePlayerSmallIDs`), `donate_gold`/`donate_troops`
  (`donatableGoldPlayerSmallIDs`/`donatableTroopsPlayerSmallIDs`),
  `embargo`/`embargo_all` (`embargoablePlayerSmallIDs` /
  `canEmbargoAll`), `targetPlayer` (`targetablePlayerSmallIDs`),
  `delete_unit` (`canDeleteUnit`), `spawn` (derived from
  `spawnPhase` + not yet owning tiles — not in `LegalityMask` itself).
- **No tracked signal, defaults to legal:** `allianceExtension`,
  `move_warship`, `cancel_boat`, `cancel_attack`, `noop`. Loss on these
  heads is never masked to zero, which is strictly safer than
  incorrectly masking out an action the model should be free to predict.

This is not a "TODO, fix later" — it's the actual, permanent shape of the
legality signal available from a coarse, player/unit-type-level mask
(`types.ts`'s own documented design choice), and downstream consumers of
`encode_legality_mask()` should treat the un-backed heads' masks as
uninformative rather than authoritative.

## 3. Class imbalance handling

Not yet measured against the full fetched dataset (no `data/processed/`
exists yet as of this writing) — this section states the *plan*, to be
confirmed or revised once `dataset_builder.py` runs against a real batch
and produces real class frequencies.

**Expectation, not yet measured:** most snapshot windows for most players
contain zero intents (a 10-tick window is ~1 second of game time; players
don't act every second). `noop` is expected to dominate by a wide margin.

**Plan:**
1. Keep `noop` as a real class rather than dropping empty-action windows —
   dropping them would teach the model that *every* observed state calls
   for action, which is false and would make it wildly trigger-happy.
2. Downweight `noop` in the `action_type` head's cross-entropy loss
   (inverse-frequency or a capped inverse-frequency weight, decided once
   real frequencies are measured — an uncapped inverse-frequency weight on
   a potentially 90%+-`noop` distribution would overcorrect).
3. Do **not** downsample `noop` windows at the `dataset_builder.py` level —
   keep the featurization stage a faithful, lossless-where-possible
   representation (per `features/README.md`) and handle the imbalance in
   the loss function, where it's a one-line, reversible decision instead
   of a destructive, hard-to-audit resampling of the dataset on disk.
4. Revisit weighting once `bc_trainer.py` exists and per-head validation
   accuracy is measurable — this plan is not a substitute for looking at
   real numbers.

## 4. Train/val/test split strategy

**Split at the game level, never the tick level.** Ticks from the same
game are highly correlated (same map, same players, same match dynamics);
splitting at the tick level would leak information between train and
eval and inflate reported accuracy.

Plan: hash each `gameId` to a stable bucket (e.g.
`int(sha256(gameId).hexdigest(), 16) % 100`) and assign ranges to
train/val/test (e.g. 80/10/10). Deterministic and reproducible from the
gameId alone — no split-assignment file to keep in sync as new games are
fetched. Implemented in `training/`, not `features/`, per the existing
package boundary (`features/README.md`: split strategy is training's job).

## 5. Success criteria for this prototype

Explicitly **not** "does it win games." Per the original project brief,
this prototype's bar is:

1. **Does it predict expert actions above a trivial baseline?** Report
   per-head accuracy (`action_type`, `unit_type`) against a most-common-
   class baseline (predicting `noop` — or whatever the true plurality
   class turns out to be — for every example). Beating that baseline on
   the *non-noop* classes specifically matters more than raw accuracy,
   since a model that only ever predicts `noop` would score deceptively
   well on the imbalanced full distribution. The argument heads
   (`tile_target`, `target_player`) are scored on only the examples that
   take that argument, conditioned on the true action type, against a
   uniform-over-legal-choices baseline (their number of choices varies
   per example, so a most-common-class baseline doesn't apply).
2. **Is the resulting checkpoint loadable by a follow-on RL stage?**
   `models/checkpoint_format.py` defines the saved schema;
   `evaluation/held_out_report.py` should confirm a freshly loaded
   checkpoint reproduces the same held-out metrics as the one recorded at
   training time, as a basic save/load correctness check.
3. **Legality-violation rate**: how often the model's top prediction for
   `action_type` falls in a masked-illegal slot per Section 2's coverage
   (only meaningful for the action types with a real backing signal).
   Not a hard pass/fail bar — a diagnostic for whether the model is
   learning the game's actual constraints or just surface statistics.
