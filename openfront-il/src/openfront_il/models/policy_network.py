"""``PolicyNetwork``: ``(spatial latent, flat facts) -> {head_name: logits}``.

Head sizes (``action_type_count``, ``unit_type_count``) are passed in as
plain integers, not imported from ``features/action_encoder.py`` —
``models/`` is not allowed to import from ``features/`` (see
``models/README.md``). ``training/`` is what actually connects
``action_encoder.ACTION_TYPES``/``UNIT_TYPES`` lengths to these
constructor arguments; this module only needs to agree on the head
*names* (``"action_type"``, ``"unit_type"``) as a documented convention.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from openfront_il.models.spatial_encoder import SpatialEncoder


class PolicyNetwork(nn.Module):
    """``(spatial [B, C, H, W], flat [B, F]) -> {"action_type": logits
    [B, action_type_count], "unit_type": logits [B, unit_type_count]}``.
    """

    def __init__(
        self,
        spatial_encoder: SpatialEncoder,
        flat_dim: int,
        action_type_count: int,
        unit_type_count: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.flat_dim = flat_dim
        self.action_type_count = action_type_count
        self.unit_type_count = unit_type_count
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(spatial_encoder.latent_dim + flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_type_head = nn.Linear(hidden_dim, action_type_count)
        self.unit_type_head = nn.Linear(hidden_dim, unit_type_count)

    def forward(self, spatial: Tensor, flat: Tensor) -> dict[str, Tensor]:
        if flat.dim() != 2 or flat.shape[1] != self.flat_dim:
            raise ValueError(f"expected flat shape [B, {self.flat_dim}], got {tuple(flat.shape)}")

        latent = self.spatial_encoder(spatial)
        combined = torch.cat([latent, flat], dim=-1)
        hidden = self.trunk(combined)
        return {
            "action_type": self.action_type_head(hidden),
            "unit_type": self.unit_type_head(hidden),
        }
