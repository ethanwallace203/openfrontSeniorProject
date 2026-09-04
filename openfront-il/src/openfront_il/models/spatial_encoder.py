"""``SpatialEncoder``: a small convnet that compresses the large,
high-dimensional tile grid into a fixed-size latent vector.

This is the "compress only the large, high-dimensional state" half of the
observation design principle in ``features/README.md`` — the flat,
per-player facts bypass this entirely and reach ``PolicyNetwork`` raw (see
``policy_network.py``).

Real OpenFront maps vary hugely in size (a small map versus 2904x1672 for
Europe), so this module never assumes a fixed ``(H, W)``: an adaptive
pooling layer collapses whatever spatial size the convolutional trunk
produces down to one vector per channel before the final linear
projection, so a checkpoint trained on one map size still runs (structurally)
on another. It says nothing about whether the *learned features* transfer
across map sizes — that's an empirical question for training, not
something this module's shape-generality claims.
"""

from __future__ import annotations

from torch import Tensor, nn


class SpatialEncoder(nn.Module):
    """``spatial tensor [B, C, H, W] -> latent [B, latent_dim]``."""

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

        # Collapses any (H, W) down to (1, 1) so the encoder's output size
        # never depends on the input map's dimensions.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(previous_channels, latent_dim)

    def forward(self, spatial: Tensor) -> Tensor:
        if spatial.dim() != 4:
            raise ValueError(f"expected a [B, C, H, W] tensor, got shape {tuple(spatial.shape)}")
        if spatial.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {spatial.shape[1]}")
        features = self.conv(spatial)
        pooled = self.pool(features).flatten(1)
        return self.project(pooled)  # type: ignore[no-any-return]
