from __future__ import annotations

import numpy as np

from openfront_il.features.action_encoder import (
   ACTION_TYPES,
   NOOP_ACTION,
   UNIT_TYPES,
   EncodedAction,
)

from dataclasses import dataclass, field

from openfront_il.features.observation_encoder import _TERRA_NULLIUS_SMALL_ID
from openfront_il.replay.snapshot_schema import SnapshotHeader, TickSnapshot



@dataclass(frozen=True)
class ActionDecoder():
   width: int
   height: int
   spatial_encoder_factor: int = 8
   greedy: bool = True
   temperature: float = 1.0
   rng: np.random.Generator = field(default_factory=np.random.default_rng)
   player_field = {"attack": "targetID", "donate_gold": "recipient", "donate_troops": "recipient", "allianceRequest": "recipient", "allianceExtension": "recipient", "breakAlliance": "recipient", "allianceReject": "requestor", "embargo": "targetID", "targetPlayer": "target"}
   amount_field = {"attack": "troops", "boat": "troops", "donate_gold": "gold", "donate_troops": "troops"}

   def decode_action(self, action: EncodedAction, *, tick, viewer_small_id) -> dict | None:
      name = ACTION_TYPES[action.action_type]
      if name == "noop":
         return None
      intent = {"type": name}

      if name in ("build_unit", "upgrade_structure"):
         unit = UNIT_TYPES[action.unit_type]
         if unit == "none":
            return None
         intent["unit"] = unit
      if name in self.player_field:
         small_id = action.target_player_small_id
         if small_id == viewer_small_id:
            return None
         player_id = self.player_id_for_small_id(tick, small_id)
         if player_id is None and name != "attack":
            return None
         intent[self.player_field[name]] = player_id
      if name in self.amount_field:
         # attack/donate amounts are nullable in the engine schema (null =
         # the engine's default amount); boat's troops are required.
         if action.amount is None and name == "boat":
            return None
         intent[self.amount_field[name]] = action.amount
      if name in ("embargo", "embargo_all"):
         intent["action"] = "start"
      if action.target_x is not None and action.target_y is not None:
         # boat names its destination tile "dst"; every other tile intent uses "tile".
         intent["dst" if name == "boat" else "tile"] = self.tile_ref(action.target_x, action.target_y)
      return intent

   def mask_choice(self, logits, mask) -> int | None:
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


   def encode_unit_type_mask(self, legality) -> np.ndarray:
      mask = np.isin(UNIT_TYPES, legality.buildableUnitTypes)
      mask[0] = False ##disable None since this is only used when something is being buit so it should never be None
      return mask



   def choose_tile(self, tile_logits, tile_mask) -> tuple[int, int] | None:
      logits = np.asarray(tile_logits, dtype=np.float64)
      mask = np.asanyarray(tile_mask, dtype=bool)
      if logits.shape != mask.shape:
         raise ValueError("logits and mask do not have the same shape")
      if logits.ndim != 2 or mask.ndim != 2:
         raise ValueError("logits or mask is not 2D")
      if self.spatial_encoder_factor < 1:
         raise ValueError("spatial_encoder_factor must be at least 1")

      coarse_h, coarse_w = logits.shape

      index = self.mask_choice(logits.ravel(), mask.ravel())
      if index is None:
         return None
      coarse_y = index // coarse_w
      coarse_x = index % coarse_w

      x = coarse_x * self.spatial_encoder_factor + self.spatial_encoder_factor // 2
      y = coarse_y * self.spatial_encoder_factor + self.spatial_encoder_factor // 2

      x = min(x, self.width - 1)
      y = min(y, self.height - 1)

      return (int(x), int(y))

   def choose_target_player(self, player_logits, candidate_small_ids, player_mask) -> int | None:
      if len(candidate_small_ids) != len(player_logits):
         raise ValueError("player logits and ID's dimensions do not align")
      index = self.mask_choice(player_logits, player_mask)
      if index is None:
         return None
      return int(candidate_small_ids[index])

   def tile_ref(self, x, y) -> int:
      if self.width <= 0:
         raise ValueError("width must be positive")
      if not (0 <= x < self.width) or y < 0:
         raise ValueError("x is out of bounds")
      return int(y * self.width + x)

   @staticmethod
   def player_id_for_small_id(tick, small_id) -> str | None:
      if small_id == _TERRA_NULLIUS_SMALL_ID:
         return None
      # Intents reference players by the engine's PlayerID (player.id),
      # not clientID — nations/bots don't even have a clientID.
      for player in tick.players:
         if player.smallID == small_id:
            return player.id
      return None
