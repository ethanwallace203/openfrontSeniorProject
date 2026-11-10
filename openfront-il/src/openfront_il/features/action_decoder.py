from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openfront_il.features.action_encoder import (
   ACTION_TYPES,
   UNIT_TYPES,
   TILE_TARGET_ACTION_TYPES,
   EncodedAction,
   decode_amount_bucket,
)
from openfront_il.features.observation_encoder import TERRA_NULLIUS_SMALL_ID
from openfront_il.replay.snapshot_schema import PlayerScalars, LegalityMask, TickSnapshot

NEEDS_UNIT_ID = frozenset(
   {"upgrade_structure", "move_warship", "cancel_boat", "cancel_attack", "delete_unit"}
)
player_field = {"attack": "targetID", "donate_gold": "recipient",
                 "donate_troops": "recipient", "allianceRequest": "recipient", 
                 "allianceExtension": "recipient", "breakAlliance": "recipient", 
                 "allianceReject": "requestor", "embargo": "targetID", 
                 "targetPlayer": "target"}
amount_field = {"attack": "troops", "boat": "troops", "donate_gold": "gold",
                 "donate_troops": "troops"}

@dataclass(frozen=True)
class ActionDecoder:
   width: int
   height: int
   spatial_encoder_factor: int = 8
   greedy: bool = True
   temperature: float = 1.0
   rng: np.random.Generator = field(default_factory=np.random.default_rng)

   def decode_action(
      self, action: EncodedAction, *, tick: TickSnapshot, viewer_small_id: int
   ) -> dict[str, object] | None:
      name = ACTION_TYPES[action.action_type]
      if name == "noop" or name in NEEDS_UNIT_ID:
         return None
      intent: dict[str, object] = {"type": name}
      if name == "build_unit":
         unit = UNIT_TYPES[action.unit_type]
         if unit == "none":
            return None
         intent["unit"] = unit
      if name in player_field:
         small_id = action.target_player_small_id
         if small_id is None or small_id == viewer_small_id:
            return None
         player_id = self.player_id_for_small_id(tick, small_id)
         if player_id is None and not (name == "attack" and small_id == TERRA_NULLIUS_SMALL_ID):
            return None
         intent[player_field[name]] = player_id
      if name in amount_field:
         # attack/donate amounts are nullable in the engine schema (null =
         # the engine's default amount); boat's troops are required.
         if action.amount is None and name == "boat":
            return None
         intent[amount_field[name]] = action.amount
      if name in ("embargo", "embargo_all"):
         intent["action"] = "start"
      if name in TILE_TARGET_ACTION_TYPES:
         if action.target_x is None or action.target_y is None:
            return None
         # boat names its destination tile "dst"; every other tile intent uses "tile".
         intent["dst" if name == "boat" else "tile"] = self.tile_ref(action.target_x, action.target_y)
      return intent

   def mask_choice(self, logits: np.ndarray, mask: np.ndarray) -> int | None:
      logits = np.asarray(logits, dtype=np.float64)
      mask = np.asanyarray(mask, dtype=bool)
      if logits.ndim != 1 or mask.ndim != 1 or logits.shape[0] != mask.shape[0]:
         raise ValueError("logits and mask must be 1-D and the same length")
      if self.temperature <= 0:
         raise ValueError("temperature must be positive")
      if not mask.any():
         return None
      maskedLogits = np.where(mask, logits, float("-inf"))
      if self.greedy:
         choice = int(np.argmax(maskedLogits))
      else:
         scaledLogits = maskedLogits / self.temperature
         scaledLogits = scaledLogits - scaledLogits.max()
         processedLogits = (np.exp(scaledLogits)) / (np.sum(np.exp(scaledLogits)))
         choice = self.rng.choice(len(processedLogits), p=processedLogits)
      return int(choice)


   def encode_unit_type_mask(self, legality: LegalityMask) -> np.ndarray:
      mask: np.ndarray = np.isin(UNIT_TYPES, legality.buildableUnitTypes)
      mask[0] = False ##disable None since this is only used when something is being buit so it should never be None
      return mask

   def choose_amount(
      self, amount_logits: np.ndarray, action_type: int, viewer: PlayerScalars
   ) -> float | None:
      mask = np.ones(len(amount_logits), dtype=bool)
      bucket = self.mask_choice(amount_logits, mask)
      if bucket is None:
         return None
      return decode_amount_bucket(action_type, bucket, viewer_troops=viewer.troops, viewer_gold=float(viewer.gold))


   def choose_tile(
      self, tile_logits: np.ndarray, tile_mask: np.ndarray
   ) -> tuple[int, int] | None:
      logits = np.asarray(tile_logits, dtype=np.float64)
      mask = np.asanyarray(tile_mask, dtype=bool)
      if logits.ndim != 2 or mask.ndim != 2:
         raise ValueError("logits or mask is not 2D")
      if self.spatial_encoder_factor < 1:
         raise ValueError("spatial_encoder_factor must be at least 1")

      f = self.spatial_encoder_factor
      coarse_h, coarse_w = logits.shape

      if mask.shape[0] < coarse_h * f or mask.shape[1] < coarse_w * f:
         raise ValueError("width or height logit shape does not match spatial encoder")

      blocks = mask[: coarse_h * f, : coarse_w * f].reshape(coarse_h, f, coarse_w, f)
      cell_legal = blocks.any(axis=(1, 3))     

      index = self.mask_choice(logits.ravel(), cell_legal.ravel())
      if index is None:
         return None
      coarse_y = index // coarse_w
      coarse_x = index % coarse_w

      in_cell = blocks[coarse_y, :, coarse_x, :]
      ys, xs = np.nonzero(in_cell)
      center = (f - 1) / 2
      nearest = int(np.argmin((ys - center) ** 2 + (xs - center) ** 2))

      x = coarse_x * f + int(xs[nearest])
      y = coarse_y * f + int(ys[nearest])

      return (int(x), int(y))

   def choose_target_player(
      self, player_logits: np.ndarray, candidate_small_ids: np.ndarray, player_mask: np.ndarray
   ) -> int | None:
      if len(candidate_small_ids) != len(player_logits):
         raise ValueError("player logits and ID's dimensions do not align")
      index = self.mask_choice(player_logits, player_mask)
      if index is None:
         return None
      return int(candidate_small_ids[index])

   def tile_ref(self, x: int, y: int) -> int:
      if self.width <= 0:
         raise ValueError("width must be positive")
      if not (0 <= x < self.width) or not (0 <= y < self.height):
         raise ValueError(f"tile ({x}, {y}) is outside the {self.width}x{self.height} map")
      return int(y * self.width + x)

   @staticmethod
   def player_id_for_small_id(tick: TickSnapshot, small_id: int) -> str | None:
      if small_id == TERRA_NULLIUS_SMALL_ID:
         return None
      # Intents reference players by the engine's PlayerID (player.id),
      # not clientID — nations/bots don't even have a clientID.
      for player in tick.players:
         if player.smallID == small_id:
            return player.id
      return None
