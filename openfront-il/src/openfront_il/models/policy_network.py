"""``PolicyNetwork``: ``(spatial grid, flat facts, player candidates) ->
{head_name: logits}``.

Four heads, factorized the way an OpenFront intent is: *what* to do
(``action_type``, ``unit_type``), and the argument it needs — *where*
(``tile_target``) or *whom* (``target_player``). See
``docs/rl_handoff.md`` for which action types use which argument.

The two argument heads are pointer-style rather than fixed classifiers,
which is what keeps them independent of map size and player count:

- ``tile_target``: a 1x1 conv turns every cell of ``SpatialEncoder``'s
  feature map into a key vector, scored against a query from the trunk —
  one logit per cell, ``[B, H', W']``, for any ``(H', W')``.
- ``target_player``: a small MLP turns each candidate player's feature
  vector into a key, scored against a query the same way — one logit per
  candidate, ``[B, N]``, for any ``N``. Player identities are arbitrary
  per game, so scoring by *features* (size, troops, relation, shared
  border) is what generalizes; a fixed slot index would not.

Both queries are conditioned on the action type (an embedding added to
the trunk's output), since the right tile for a spawn is not the right
tile for a City. Training passes the true label (teacher forcing);
inference passes whatever it sampled from ``action_type``, or omits it to
condition on the argmax.

Legality masking is the caller's job, not this module's (it needs
``features/``'s per-action-type rules, which ``models/`` may not import).
The one mask applied here is structural: padded candidate slots
(``player_valid == False``) get the dtype's minimum logit.

Head sizes are passed in as plain integers, not imported from
``features/action_encoder.py`` — ``models/`` is not allowed to import from
``features/`` (see ``models/README.md``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from openfront_il.models.spatial_encoder import SpatialEncoder


class PolicyNetwork(nn.Module):
    """``(spatial [B, C, H, W], flat [B, F], player_features [B, N, P],
    player_valid [B, N], action_type [B] | None) -> {"action_type": [B,
    action_type_count], "unit_type": [B, unit_type_count], "tile_target":
    [B, H', W'], "target_player": [B, N], "amount": [B,
    amount_bucket_count]}`` logits.
    """

    def __init__(
        self,
        spatial_encoder: SpatialEncoder,
        flat_dim: int,
        action_type_count: int,
        unit_type_count: int,
        player_feature_dim: int,
        amount_bucket_count: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.flat_dim = flat_dim
        self.action_type_count = action_type_count
        self.unit_type_count = unit_type_count
        self.player_feature_dim = player_feature_dim
        self.amount_bucket_count = amount_bucket_count
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(spatial_encoder.latent_dim + flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_type_head = nn.Linear(hidden_dim, action_type_count)
        self.unit_type_head = nn.Linear(hidden_dim, unit_type_count)

        self.action_type_embedding = nn.Embedding(action_type_count, hidden_dim)
        self.tile_key = nn.Conv2d(spatial_encoder.feature_channels, hidden_dim, kernel_size=1)
        self.tile_query = nn.Linear(hidden_dim, hidden_dim)
        # Each candidate's key reads its scalar features *and* the feature map
        # averaged over the cells it owns (see _territory_features), so the
        # head can tell two players with identical numbers apart by what
        # their territory actually looks like.
        self.player_key = nn.Sequential(
            nn.Linear(player_feature_dim + spatial_encoder.feature_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.player_query = nn.Linear(hidden_dim, hidden_dim)
        # Reads the action-conditioned context like the two pointer heads:
        # how much to commit depends on what's being done (a boat vs. an
        # attack vs. a donation).
        self.amount_head = nn.Linear(hidden_dim, amount_bucket_count)

    def forward(
        self,
        spatial: Tensor,
        flat: Tensor,
        player_features: Tensor,
        player_valid: Tensor,
        action_type: Tensor | None = None,
        candidate_cells: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``candidate_cells`` ``[B, H', W']`` (long): which candidate owns
        each feature-map cell, ``-1`` for none — from
        ``observation_encoder.encode_candidate_cells``. Omitted, every
        candidate's territory features are zero (the head then scores on
        scalar features alone)."""
        if flat.dim() != 2 or flat.shape[1] != self.flat_dim:
            raise ValueError(f"expected flat shape [B, {self.flat_dim}], got {tuple(flat.shape)}")
        if player_features.dim() != 3 or player_features.shape[2] != self.player_feature_dim:
            raise ValueError(
                f"expected player_features shape [B, N, {self.player_feature_dim}], "
                f"got {tuple(player_features.shape)}"
            )
        if player_valid.shape != player_features.shape[:2]:
            raise ValueError(
                f"player_valid shape {tuple(player_valid.shape)} must match "
                f"player_features' [B, N] {tuple(player_features.shape[:2])}"
            )

        feature_map, latent = self.spatial_encoder(spatial)
        hidden = self.trunk(torch.cat([latent, flat], dim=-1))
        action_type_logits = self.action_type_head(hidden)

        if action_type is None:
            action_type = action_type_logits.argmax(dim=-1)
        context = hidden + self.action_type_embedding(action_type)
        scale = self.hidden_dim**-0.5

        tile_keys = self.tile_key(feature_map)
        tile_logits = torch.einsum("bdhw,bd->bhw", tile_keys, self.tile_query(context)) * scale

        territory = self._territory_features(feature_map, candidate_cells, player_features.shape[1])
        player_keys = self.player_key(
            torch.cat([player_features, territory.to(player_features.dtype)], dim=-1)
        )
        player_logits = torch.einsum("bnd,bd->bn", player_keys, self.player_query(context)) * scale
        player_logits = player_logits.masked_fill(
            ~player_valid, torch.finfo(player_logits.dtype).min
        )

        return {
            "action_type": action_type_logits,
            "unit_type": self.unit_type_head(hidden),
            "tile_target": tile_logits,
            "target_player": player_logits,
            "amount": self.amount_head(context),
        }

    def _territory_features(
        self, feature_map: Tensor, candidate_cells: Tensor | None, candidate_count: int
    ) -> Tensor:
        """``[B, N, feature_channels]``: the mean of ``feature_map`` over each
        candidate's cells (zero for a candidate owning no cell, or when
        ``candidate_cells`` is omitted). One ``scatter_add`` over the cells,
        so the cost doesn't grow with the number of candidates."""
        batch, channels, cell_height, cell_width = feature_map.shape
        territory = feature_map.new_zeros(batch, candidate_count, channels, dtype=torch.float32)
        if candidate_cells is None or candidate_count == 0:
            return territory

        # Padded batches can leave the cell grid a different size than the
        # feature map; align it (extra cells are dropped, missing ones are
        # "no owner").
        cells = candidate_cells[:, :cell_height, :cell_width].long()
        missing_rows = cell_height - cells.shape[1]
        missing_columns = cell_width - cells.shape[2]
        if missing_rows > 0 or missing_columns > 0:
            cells = torch.nn.functional.pad(cells, (0, missing_columns, 0, missing_rows), value=-1)

        owner = cells.flatten(1)  # [B, H'*W']
        owned = (owner >= 0) & (owner < candidate_count)
        index = owner.clamp(0, max(candidate_count - 1, 0))
        # float32 for the sums even under autocast: summing many bf16 cells loses precision.
        values = feature_map.flatten(2).transpose(1, 2).float() * owned.unsqueeze(-1)
        territory.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, channels), values)
        counts = feature_map.new_zeros(batch, candidate_count, dtype=torch.float32)
        counts.scatter_add_(1, index, owned.float())
        return territory / counts.clamp(min=1.0).unsqueeze(-1)
