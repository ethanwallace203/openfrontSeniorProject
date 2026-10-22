"""``SpatialEncoder``: a small convnet over the tile grid that returns
both a spatial feature map (for heads that must answer "where") and a
pooled latent vector (for heads that only need "what").

This is the "compress only the large, high-dimensional state" half of the
observation design principle in ``features/README.md`` — the flat,
per-player facts bypass this entirely and reach ``PolicyNetwork`` raw (see
``policy_network.py``).

The feature map is kept, not just the pooled vector, because the pooled
vector alone has no notion of location: ``AdaptiveAvgPool2d`` averages
every cell together, so a policy built only on it can say "build a City"
but never *where*. ``PolicyNetwork``'s tile-target head reads the feature
map directly, one logit per cell, which is also what forces these conv
filters to learn location-specific features during behavior cloning.

Real OpenFront maps vary hugely in size (a small map versus 2904x1672 for
Europe), so this module never assumes a fixed ``(H, W)``: the feature map
is whatever ``(H / downsample_factor, W / downsample_factor)`` the conv
trunk produces, and the pooled vector is fixed-size regardless. That says
nothing about whether the *learned features* transfer across map sizes —
that's an empirical question for training, not something this module's
shape-generality claims.
"""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn


class SpatialEncoder(nn.Module):
    """``spatial tensor [B, C, H, W] (float, or uint8 0/1) -> (feature map [B, conv_channels[-1],
    H', W'], latent [B, latent_dim])``, where ``H' = H // downsample_factor``
    and ``W' = W // downsample_factor``.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 128,
        conv_channels: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels

        layers: list[nn.Module] = []
        previous_channels = in_channels
        for channels in conv_channels:
            layers.append(nn.Conv2d(previous_channels, channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            previous_channels = channels
        self.conv = nn.Sequential(*layers)
        self.feature_channels = previous_channels

        # Collapses any (H, W) down to (1, 1) so the latent's size never
        # depends on the input map's dimensions.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(previous_channels, latent_dim)

    @property
    def downsample_factor(self) -> int:
        """How many input tiles one feature-map cell spans per side (one
        ``MaxPool2d(2)`` per conv layer)."""
        return int(2 ** len(self.conv_channels))

    def downsample_mask(self, tile_mask: Tensor) -> Tensor:
        """``[B, H, W]`` bool tile-level mask -> ``[B, H', W']`` bool
        cell-level mask aligned with ``forward()``'s feature map: a cell is
        ``True`` if *any* tile inside it is.

        A single max-pool with kernel/stride ``downsample_factor`` floors
        to exactly the same ``(H', W')`` as the conv trunk's repeated
        ``MaxPool2d(2)`` (``floor(floor(H/2)/2) == floor(H/4)``), so the two
        always line up cell for cell.
        """
        factor = self.downsample_factor
        pooled = functional.max_pool2d(tile_mask.unsqueeze(1).float(), factor, factor)
        return pooled.squeeze(1) > 0

    def cell_index(
        self, target_x: Tensor, target_y: Tensor, cell_height: int, cell_width: int
    ) -> Tensor:
        """Tile coordinates -> flat index into a ``[H', W']`` feature map
        (row-major, matching ``tile_logits.flatten(1)``).

        Clamped to the last row/column: the trunk's floor-division drops a
        map's last ``H % downsample_factor`` rows (and likewise columns),
        so a target on those edge tiles maps to the nearest kept cell
        rather than off the grid.
        """
        factor = self.downsample_factor
        row = (target_y // factor).clamp(max=cell_height - 1)
        column = (target_x // factor).clamp(max=cell_width - 1)
        index: Tensor = row * cell_width + column
        return index

    def forward(self, spatial: Tensor) -> tuple[Tensor, Tensor]:
        if spatial.dim() != 4:
            raise ValueError(f"expected a [B, C, H, W] tensor, got shape {tuple(spatial.shape)}")
        if spatial.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {spatial.shape[1]}")
        # features/ encodes the 0/1 channels as uint8 to keep host memory
        # and host->GPU copies 4x smaller; cast here, on the model's device.
        if not spatial.is_floating_point():
            spatial = spatial.float()
        feature_map = self.conv(spatial)
        pooled = self.pool(feature_map).flatten(1)
        return feature_map, self.project(pooled)
