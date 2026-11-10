"""The play loop: a trained BC policy controlling the human seat in a live,
headless game against nations and tribes.

Each decision (``PolicyAgent.act``):

    1. snapshot -> build_inputs        (the same encoders training used)
    2. policy -> action_type logits -> action_type_mask -> choose_action_type
    3. argument_logits                 (second pass, conditioned on that action)
    4. unit_type / tile_target / target_player / amount -> ActionDecoder
       -> EncodedAction -> decode_action -> intent -> LiveGame.step

Every choice is legality-masked with the same masks training used
(``encode_legality_mask``, ``encode_tile_target_mask``,
``encode_player_target_mask``), so the policy sees what it was trained on
and never picks something the engine would reject. Action types whose
intent needs the ID of one of the agent's own units are never chosen
(``UNDECODABLE_ACTIONS``) — no head predicts one yet.
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
from openfront_il.features.action_decoder import NEEDS_UNIT_ID, ActionDecoder
from openfront_il.features.action_encoder import (
    ACTION_TYPES,
    AMOUNT_ACTION_TYPES,
    PLAYER_TARGET_ACTION_TYPES,
    TILE_TARGET_ACTION_TYPES,
    UNIT_TYPES,
    EncodedAction,
    encode_legality_mask,
    encode_player_target_mask,
    encode_tile_target_mask,
)
from openfront_il.features.observation_encoder import (
    SPATIAL_CHANNELS,
    decode_tile_state,
    encode_candidate_cells,
    encode_observation,
    encode_player_candidates,
)
from openfront_il.models.checkpoint_format import load_checkpoint
from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.replay.snapshot_schema import (
    LegalityMask,
    PlayerScalars,
    SnapshotHeader,
    TickSnapshot,
)
from openfront_il.utils import paths

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = paths.PROJECT_ROOT / "checkpoints" / "best.pt"

_ACTION_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}

# Action types the agent never chooses: those whose intent needs the ID of
# one of its own units/attacks (no head predicts one — see the decoder's
# NEEDS_UNIT_ID), and spawn (the agent spawns randomly; liveGame.ts handles
# it before the first decision).
UNDECODABLE_ACTIONS = NEEDS_UNIT_ID | {"spawn"}

# The heads PolicyNetwork conditions on the chosen action type (step 3);
# action_type and unit_type don't depend on it.
ARGUMENT_HEADS = ("tile_target", "target_player", "amount")

# The engine refuses a new transport while this many are already at sea
# (Config.boatMaxNumber() / TransportShipUtils.canBuildTransportShip).
MAX_BOATS_AT_SEA = 3

# How far (in tiles) from the clicked tile the engine searches for a shore
# tile of the clicked tile's owner to land on — SpatialQuery.
# closestReachableShore's maxDist. Beyond it the boat is refused.
BOAT_LANDING_RANGE = 50

# Low 12 bits of a tile-state value are its owner's smallID (GameMap.ts).
_PLAYER_ID_MASK = 0xFFF
_IS_SHORELINE = SPATIAL_CHANNELS.index("is_shoreline")
_OWNED_BY_VIEWER = SPATIAL_CHANNELS.index("owned_by_viewer")

# Action types that read at least one of ARGUMENT_HEADS (only these need step 3).
_TAKES_ARGUMENTS = TILE_TARGET_ACTION_TYPES | PLAYER_TARGET_ACTION_TYPES | AMOUNT_ACTION_TYPES


@dataclass
class Decision:
    """What the agent chose on one step, for logging. ``intent`` is ``None``
    when nothing was sent (noop, or no legal argument existed)."""

    action: str
    intent: dict[str, Any] | None
    unit: str | None = None
    target_small_id: int | None = None
    tile: tuple[int, int] | None = None
    amount: float | None = None


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


def player_target_mask(
    legality: LegalityMask, candidate_ids: np.ndarray, action_type: int
) -> np.ndarray:
    """Which of ``candidate_ids`` the agent may target with ``action_type``:
    training's ``encode_player_target_mask``, tightened for one case.
    Training leaves ``allianceExtension`` uninformative (every player
    "legal", since snapshots carry no signal for it); at play time only an
    existing ally can actually be extended, and those are exactly the
    breakable alliances."""
    mask = encode_player_target_mask(legality, candidate_ids, action_type)
    if ACTION_TYPES[action_type] == "allianceExtension":
        mask &= np.isin(candidate_ids, legality.breakableAlliancePlayerSmallIDs)
    return mask


def has_coast(spatial: np.ndarray) -> bool:
    """Whether the viewer owns any shoreline. The engine launches boats only
    from the attacker's own shore (SpatialQuery.closestReachableShore returns
    nothing for a landlocked player), so without one every boat is refused."""
    return bool(((spatial[_IS_SHORELINE] > 0) & (spatial[_OWNED_BY_VIEWER] > 0)).any())


def boat_tile_mask(spatial: np.ndarray, owner_id: np.ndarray) -> np.ndarray:
    """Play-time boat destinations: training's boat mask (land owned by
    neither the viewer nor an ally) narrowed to tiles within
    ``BOAT_LANDING_RANGE`` of a shore tile *of the same owner* — the engine
    lands on the clicked tile's owner's nearest shore and refuses the boat
    if there is none in range.

    Grown outward from each owner's shoreline, one tile per step, only into
    neighboring tiles of that same owner — so it's a subset of what the
    engine accepts (its search may also cross water or other owners' land),
    never a superset. ``owner_id`` is the ``(H, W)`` smallID map.
    """
    allowed = encode_tile_target_mask(spatial, _ACTION_INDEX["boat"], 0)
    reach: np.ndarray = allowed & (spatial[_IS_SHORELINE] > 0)
    same_vertical = owner_id[1:] == owner_id[:-1]
    same_horizontal = owner_id[:, 1:] == owner_id[:, :-1]
    for _ in range(BOAT_LANDING_RANGE):
        grown = reach.copy()
        grown[1:] |= reach[:-1] & same_vertical
        grown[:-1] |= reach[1:] & same_vertical
        grown[:, 1:] |= reach[:, :-1] & same_horizontal
        grown[:, :-1] |= reach[:, 1:] & same_horizontal
        grown &= allowed
        if np.array_equal(grown, reach):
            break
        reach = grown
    return reach


def buildable_unit_mask(
    decoder: ActionDecoder, legality: LegalityMask, spatial: np.ndarray
) -> np.ndarray:
    """``unit_type`` mask for ``build_unit``: units the agent can afford
    (``legality.buildableUnitTypes``) *and* that have at least one legal
    tile on the map right now. Step 2 uses ``.any()`` of this to decide if
    ``build_unit`` is choosable at all, step 4 to pick the unit — the same
    rule in both places, so step 4 can never pick a unit with nowhere to go."""
    build = _ACTION_INDEX["build_unit"]
    mask = decoder.encode_unit_type_mask(legality)
    for unit in np.flatnonzero(mask):
        if not encode_tile_target_mask(spatial, build, int(unit)).any():
            mask[unit] = False
    return mask


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

    def action_type_mask(
        self,
        inputs: ModelInputs,
        legality: LegalityMask,
        viewer: PlayerScalars,
        spawn_phase: bool,
        boats_at_sea: int = 0,
    ) -> np.ndarray:
        """Step 2a: which action types the agent may choose right now.

        Starts from the same ``encode_legality_mask`` training's loss used,
        then also removes every action type that couldn't become a valid
        intent this step — so the policy's probability is spread only over
        actions the agent can actually carry out, instead of being wasted on
        a choice that decodes to nothing:

        - ``UNDECODABLE_ACTIONS``: no head can fill in a required unit ID.
        - player-target actions with no legal candidate (``player_target_mask``).
        - ``boat`` with no coast to launch from (``has_coast``), no tile it
          could land on, or ``boats_at_sea`` already at the engine's cap
          (``MAX_BOATS_AT_SEA``).
        - ``build_unit`` when nothing buildable has a legal tile.
        """
        if self._decoder is None:
            raise RuntimeError("action_type_mask() called before reset()")
        mask = encode_legality_mask(
            legality, spawn_phase=spawn_phase, viewer_num_tiles_owned=viewer.numTilesOwned
        )
        for name in UNDECODABLE_ACTIONS:
            mask[_ACTION_INDEX[name]] = False

        for name in PLAYER_TARGET_ACTION_TYPES:
            index = _ACTION_INDEX[name]
            if mask[index] and not player_target_mask(legality, inputs.candidate_ids, index).any():
                mask[index] = False

        boat = _ACTION_INDEX["boat"]
        if mask[boat] and (
            boats_at_sea >= MAX_BOATS_AT_SEA
            or not has_coast(inputs.spatial)
            or not encode_tile_target_mask(inputs.spatial, boat, 0).any()
        ):
            mask[boat] = False

        build = _ACTION_INDEX["build_unit"]
        if mask[build] and not buildable_unit_mask(self._decoder, legality, inputs.spatial).any():
            mask[build] = False
        return mask

    def choose_action_type(
        self, logits: dict[str, np.ndarray], mask: np.ndarray
    ) -> int | None:
        """Step 2b: pick the action type (greedy, or sampled at
        ``temperature``) among ``mask``'s legal ones. ``None`` when nothing is
        legal — the agent then does nothing this step."""
        if self._decoder is None:
            raise RuntimeError("choose_action_type() called before reset()")
        return self._decoder.mask_choice(logits["action_type"], mask)

    def argument_logits(
        self, inputs: ModelInputs, first_pass: dict[str, np.ndarray], action_type: int
    ) -> dict[str, np.ndarray]:
        """Step 3: the argument heads' logits (``ARGUMENT_HEADS``) conditioned
        on the action type actually chosen in step 2.

        ``PolicyNetwork`` scores tiles, players and amounts *given* an action
        — where to send a boat is not where to build a City. With no
        ``action_type``, the first pass conditioned them on the model's own
        top action over all types; that is only the right conditioning when
        step 2 picked that same action (greedy, with the top action legal).
        Otherwise this runs a second forward pass conditioned on the chosen
        one. ``unit_type`` is read from the first pass either way — it isn't
        conditioned on the action.
        """
        if int(np.argmax(first_pass["action_type"])) == action_type:
            source = first_pass
        else:
            source = self._run_policy(inputs, action_type=action_type)
        return {head: source[head] for head in ARGUMENT_HEADS}

    def act(self, tick: TickSnapshot, client_id: str = AGENT_CLIENT_ID) -> Decision:
        """One full decision for ``client_id``: steps 1-4 (see the module
        docstring). Every argument comes from its head, masked by the same
        rule training used; if a needed argument has no legal choice the
        decision is "do nothing" rather than a broken intent."""
        if self._decoder is None or self._header is None:
            raise RuntimeError("act() called before reset()")
        decoder = self._decoder

        me = next(p for p in tick.players if p.clientID == client_id)
        legality = tick.legal[client_id]

        # Step 1: everything the model reads.
        inputs = self.build_inputs(tick, me.smallID)
        logits = self._run_policy(inputs)

        # Step 2: the action type.
        boats_at_sea = sum(
            1 for u in tick.units if u.ownerSmallID == me.smallID and u.type == "Transport"
        )
        mask = self.action_type_mask(inputs, legality, me, tick.spawnPhase, boats_at_sea)
        action_index = self.choose_action_type(logits, mask)
        if action_index is None:
            return Decision(action="noop", intent=None)
        action = ACTION_TYPES[action_index]
        if action == "noop":
            return Decision(action=action, intent=None)

        # Step 3: argument logits conditioned on that action (only if it has any).
        arguments = (
            self.argument_logits(inputs, logits, action_index)
            if action in _TAKES_ARGUMENTS
            else {}
        )

        # Step 4: each argument from its head.
        unit_index = UNIT_TYPES.index("none")
        unit: str | None = None
        if action == "build_unit":
            chosen_unit = decoder.mask_choice(
                logits["unit_type"], buildable_unit_mask(decoder, legality, inputs.spatial)
            )
            if chosen_unit is None:
                return Decision(action=action, intent=None)
            unit_index = chosen_unit
            unit = UNIT_TYPES[unit_index]

        tile: tuple[int, int] | None = None
        if action in TILE_TARGET_ACTION_TYPES:
            if action == "boat":
                owner_id = decode_tile_state(self._header, tick) & _PLAYER_ID_MASK
                tile_mask = boat_tile_mask(inputs.spatial, owner_id)
            else:
                tile_mask = encode_tile_target_mask(inputs.spatial, action_index, unit_index)
            tile = decoder.choose_tile(arguments["tile_target"], tile_mask)
            if tile is None:
                return Decision(action=action, intent=None, unit=unit)

        target: int | None = None
        if action in PLAYER_TARGET_ACTION_TYPES:
            target = decoder.choose_target_player(
                arguments["target_player"],
                inputs.candidate_ids,
                player_target_mask(legality, inputs.candidate_ids, action_index),
            )
            if target is None:
                return Decision(action=action, intent=None, unit=unit, tile=tile)

        amount: float | None = None
        if action in AMOUNT_ACTION_TYPES:
            amount = decoder.choose_amount(arguments["amount"], action_index, me)

        encoded = EncodedAction(
            action_type=action_index,
            unit_type=unit_index,
            target_x=tile[0] if tile else None,
            target_y=tile[1] if tile else None,
            target_player_small_id=target,
            amount=amount,
        )
        intent = decoder.decode_action(encoded, tick=tick, viewer_small_id=me.smallID)
        return Decision(
            action=action,
            intent=intent,
            unit=unit,
            target_small_id=target,
            tile=tile,
            amount=amount,
        )


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
