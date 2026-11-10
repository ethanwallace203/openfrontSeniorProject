"""The play loop: a trained BC policy controlling the human seat in a live,
headless game against nations and tribes.

Each step:

    snapshot -> encode_observation -> policy -> logits
             -> legality-masked action_type (+ unit_type for builds)
             -> target player / build tile  (temporary rules, see below)
             -> ActionDecoder.decode_action -> intent -> LiveGame.step

The observation goes through the exact ``encode_observation`` the training
rows were built from, and action_type is masked with the same
``encode_legality_mask`` the trainer's loss used, so the policy sees what
it was trained on.

**Temporary rules.** The policy only has ``action_type`` and ``unit_type``
heads. Until target-player / tile / amount heads exist, those parts of an
action come from simple rules here (``_pick_target``, ``_pick_build_tile``)
and amounts are left ``null`` (the engine's own default). Action types the
agent can't fully specify yet are masked out (``UNSUPPORTED_ACTIONS``), so
the policy's probability mass is spread over the ones it can.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openfront_il.agent.live_game import AGENT_CLIENT_ID, LiveGame
from openfront_il.features.action_decoder import ActionDecoder
from openfront_il.features.action_encoder import (
    ACTION_TYPES,
    UNIT_TYPES,
    EncodedAction,
    encode_legality_mask,
)
from openfront_il.features.observation_encoder import (
    SPATIAL_CHANNELS,
    encode_candidate_cells,
    encode_observation,
    encode_player_candidates,
)
from openfront_il.models.checkpoint_format import load_checkpoint
from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.replay.snapshot_schema import LegalityMask, SnapshotHeader, TickSnapshot
from openfront_il.utils import paths

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = paths.PROJECT_ROOT / "checkpoints" / "best.pt"

_ACTION_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}

# Action types that need something the agent can't produce yet: a unit/attack
# ID (no head or rule for picking one of the agent's units), a boat
# destination plus a required troop count, or a spawn tile (spawn is random).
UNSUPPORTED_ACTIONS = frozenset(
    {
        "spawn",
        "boat",
        "upgrade_structure",
        "move_warship",
        "cancel_boat",
        "cancel_attack",
        "delete_unit",
    }
)

# Action type -> the LegalityMask list of smallIDs it may target. Mirrors
# encode_legality_mask's backing signals.
TARGET_CANDIDATES = {
    "attack": "attackablePlayerSmallIDs",
    "allianceRequest": "allianceRequestablePlayerSmallIDs",
    "allianceReject": "allianceAcceptablePlayerSmallIDs",
    "allianceExtension": "breakableAlliancePlayerSmallIDs",
    "breakAlliance": "breakableAlliancePlayerSmallIDs",
    "donate_gold": "donatableGoldPlayerSmallIDs",
    "donate_troops": "donatableTroopsPlayerSmallIDs",
    "embargo": "embargoablePlayerSmallIDs",
    "targetPlayer": "targetablePlayerSmallIDs",
}

# smallID the decoder turns into targetID=null — i.e. "attack unclaimed land".
_EXPAND = 0

_NUKES = frozenset({"Atom Bomb", "Hydrogen Bomb", "MIRV"})
_NAVAL = frozenset({"Port", "Warship"})
_CH_LAND = SPATIAL_CHANNELS.index("is_land")
_CH_SHORE = SPATIAL_CHANNELS.index("is_shoreline")
_CH_MINE = SPATIAL_CHANNELS.index("owned_by_viewer")
_CH_ENEMY = SPATIAL_CHANNELS.index("owned_by_enemy")


@dataclass
class Decision:
    """What the agent chose on one step, for logging."""

    action: str
    intent: dict[str, Any] | None
    unit: str | None = None
    target_small_id: int | None = None
    tile: tuple[int, int] | None = None


@dataclass(frozen=True)
class ModelInputs:
    """Everything ``PolicyNetwork`` reads for one decision, from one
    player's perspective — built once per step and reused for both forward
    passes (action type, then the arguments conditioned on it).

    Built by exactly the encoders the training examples were built with
    (``example_loader.ExampleTable.example``), so the policy sees the same
    format it was trained on.
    """

    spatial: np.ndarray  # [C, H, W] uint8
    flat: np.ndarray  # [F] float32
    candidate_ids: np.ndarray  # [N] smallIDs: terra nullius, then other living players
    player_features: np.ndarray  # [N, P] float32, one row per candidate
    candidate_cells: np.ndarray  # [H', W'] int16: which candidate owns each cell, -1 none


@dataclass
class EpisodeResult:
    game_id: str
    map: str
    won: bool
    survived: bool
    final_tick: int
    steps: int
    peak_tiles: int
    final_tiles: int
    action_counts: Counter[str] = field(default_factory=Counter)
    intents_sent: int = 0
    record_path: Path | None = None


def _candidates(action: str, legality: LegalityMask) -> list[int]:
    ids: list[int] = list(getattr(legality, TARGET_CANDIDATES[action]))
    if action == "attack" and legality.canExpand:
        ids.append(_EXPAND)
    return ids


def _pick_target(
    action: str, legality: LegalityMask, tick: TickSnapshot, rng: np.random.Generator
) -> int | None:
    """Temporary rule until a target-player head exists. Attack: expand
    into unclaimed land while there is any, otherwise hit the weakest
    attackable neighbor (fewest troops). Everything else: a random legal
    candidate."""
    candidates = _candidates(action, legality)
    if not candidates:
        return None
    if action == "attack":
        if _EXPAND in candidates:
            return _EXPAND
        troops = {p.smallID: p.troops for p in tick.players}
        return min(candidates, key=lambda small_id: troops.get(small_id, float("inf")))
    return int(rng.choice(candidates))


def _pick_build_tile(
    unit: str, spatial: np.ndarray, rng: np.random.Generator
) -> tuple[int, int] | None:
    """Temporary rule until a tile head exists: a random tile that's at
    least plausible for the unit. Nukes target enemy land, ports/warships
    go on the agent's own shoreline, everything else on the agent's own
    land. The engine still validates placement and silently drops an
    invalid build."""
    if unit in _NUKES:
        allowed = spatial[_CH_ENEMY] > 0
    else:
        allowed = (spatial[_CH_MINE] > 0) & (spatial[_CH_LAND] > 0)
        if unit in _NAVAL:
            allowed &= spatial[_CH_SHORE] > 0
    ys, xs = np.nonzero(allowed)
    if len(xs) == 0:
        return None
    index = int(rng.integers(len(xs)))
    return int(xs[index]), int(ys[index])


class PolicyAgent:
    """Wraps a trained `PolicyNetwork` as an agent for one seat.

    Args:
        greedy: always take the highest-probability legal action. Off by
            default — a BC policy trained on noop-heavy data will greedily
            pick noop nearly every step; sampling shows its real mix.
        temperature: sampling temperature (only used when not greedy).
        seed: seeds both action sampling and the temporary rules.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        *,
        greedy: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.policy = policy.eval()
        self.greedy = greedy
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self._decoder: ActionDecoder | None = None
        self._header: SnapshotHeader | None = None

    @classmethod
    def from_checkpoint(cls, path: Path = DEFAULT_CHECKPOINT, **kwargs: Any) -> PolicyAgent:
        policy, _ = load_checkpoint(path)
        return cls(policy, **kwargs)

    @property
    def cell_factor(self) -> int:
        """Tiles per feature-map cell side — the size of one ``tile_target``
        / ``candidate_cells`` cell, fixed by the model's conv trunk."""
        return self.policy.spatial_encoder.downsample_factor

    def reset(self, header: SnapshotHeader) -> None:
        """Call once per game: the decoder is tied to the map's size."""
        self._header = header
        self._decoder = ActionDecoder(
            width=header.width,
            height=header.height,
            spatial_encoder_factor=self.cell_factor,
            greedy=self.greedy,
            temperature=self.temperature,
            rng=self.rng,
        )

    def build_inputs(self, tick: TickSnapshot, viewer_small_id: int) -> ModelInputs:
        """Step 1 of a decision: encode ``tick`` from the viewer's
        perspective into everything the policy reads."""
        if self._header is None:
            raise RuntimeError("build_inputs() called before reset()")
        header = self._header
        spatial, flat = encode_observation(header, tick, viewer_small_id)
        candidate_ids, player_features = encode_player_candidates(header, tick, viewer_small_id)
        candidate_cells = encode_candidate_cells(
            header, tick, viewer_small_id, candidate_ids, self.cell_factor
        )
        return ModelInputs(
            spatial=spatial,
            flat=flat,
            candidate_ids=candidate_ids,
            player_features=player_features,
            candidate_cells=candidate_cells,
        )

    def _run_policy(
        self, inputs: ModelInputs, action_type: int | None = None
    ) -> dict[str, np.ndarray]:
        """One forward pass on a batch of one -> each head's logits as numpy,
        batch dimension dropped. ``action_type`` conditions the argument
        heads (tile_target / target_player / amount) on that action; omitted,
        the model conditions them on its own top action."""
        with torch.no_grad():
            logits = self.policy(
                torch.from_numpy(inputs.spatial).unsqueeze(0),
                torch.from_numpy(inputs.flat).unsqueeze(0),
                torch.from_numpy(inputs.player_features).unsqueeze(0),
                torch.ones(1, len(inputs.candidate_ids), dtype=torch.bool),
                action_type=None if action_type is None else torch.tensor([action_type]),
                candidate_cells=torch.from_numpy(inputs.candidate_cells).long().unsqueeze(0),
            )
        return {head: value[0].float().numpy() for head, value in logits.items()}

    def act(self, tick: TickSnapshot, client_id: str = AGENT_CLIENT_ID) -> Decision:
        if self._decoder is None or self._header is None:
            raise RuntimeError("act() called before reset()")
        decoder = self._decoder

        me = next(p for p in tick.players if p.clientID == client_id)
        legality = tick.legal[client_id]

        inputs = self.build_inputs(tick, me.smallID)
        spatial = inputs.spatial
        logits = self._run_policy(inputs)
        action_logits = logits["action_type"]
        unit_logits = logits["unit_type"]

        mask = encode_legality_mask(
            legality, spawn_phase=tick.spawnPhase, viewer_num_tiles_owned=me.numTilesOwned
        )
        for name in UNSUPPORTED_ACTIONS:
            mask[_ACTION_INDEX[name]] = False
        for name in TARGET_CANDIDATES:
            if not _candidates(name, legality):
                mask[_ACTION_INDEX[name]] = False

        action_index = decoder.mask_choice(action_logits, mask)
        if action_index is None:
            return Decision(action="noop", intent=None)
        action = ACTION_TYPES[action_index]

        unit_index = 0
        unit: str | None = None
        tile: tuple[int, int] | None = None
        if action == "build_unit":
            chosen_unit = decoder.mask_choice(unit_logits, decoder.encode_unit_type_mask(legality))
            if chosen_unit is None:
                return Decision(action=action, intent=None)
            unit_index = chosen_unit
            unit = UNIT_TYPES[unit_index]
            tile = _pick_build_tile(unit, spatial, self.rng)
            if tile is None:
                return Decision(action=action, intent=None, unit=unit)

        target = None
        if action in TARGET_CANDIDATES:
            target = _pick_target(action, legality, tick, self.rng)

        encoded = EncodedAction(
            action_type=action_index,
            unit_type=unit_index,
            target_x=tile[0] if tile else None,
            target_y=tile[1] if tile else None,
            target_player_small_id=target,
        )
        intent = decoder.decode_action(encoded, tick=tick, viewer_small_id=me.smallID)
        return Decision(action=action, intent=intent, unit=unit, target_small_id=target, tile=tile)


def play_episode(
    agent: PolicyAgent,
    game: LiveGame,
    *,
    map: str = "Onion",  # noqa: A002 - matches the engine's own field name
    seed: str | None = None,
    bots: int = 4,
    difficulty: str = "Medium",
    max_steps: int = 5000,
    save: bool = True,
    save_path: Path | None = None,
) -> EpisodeResult:
    """Plays one full game: until the agent wins, is eliminated, or
    ``max_steps`` decisions have been made."""
    header, tick = game.reset(map=map, seed=seed, bots=bots, difficulty=difficulty)
    agent.reset(header)

    counts: Counter[str] = Counter()
    intents_sent = 0
    peak_tiles = game.agent.numTilesOwned if game.agent else 0
    steps = 0

    while not game.is_done and steps < max_steps:
        decision = agent.act(tick)
        counts[decision.action if decision.intent else "noop"] += 1
        intents = [decision.intent] if decision.intent else []
        intents_sent += len(intents)
        if decision.intent:
            logger.debug("tick %d: %s", tick.tick, decision.intent)

        tick = game.step(intents)
        steps += 1
        if game.agent is not None:
            peak_tiles = max(peak_tiles, game.agent.numTilesOwned)
        if steps % 100 == 0:
            logger.info(
                "step %d tick %d: tiles=%d troops=%.0f",
                steps,
                tick.tick,
                game.agent.numTilesOwned if game.agent else 0,
                game.agent.troops if game.agent else 0,
            )

    record_path = game.save(save_path) if save else None
    return EpisodeResult(
        game_id=header.gameId,
        map=header.map,
        won=game.agent_won,
        survived=game.agent_alive,
        final_tick=tick.tick,
        steps=steps,
        peak_tiles=peak_tiles,
        final_tiles=game.agent.numTilesOwned if game.agent else 0,
        action_counts=counts,
        intents_sent=intents_sent,
        record_path=record_path,
    )
