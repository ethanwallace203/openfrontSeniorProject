import pytest
import torch

from openfront_il.models.spatial_encoder import SpatialEncoder


class TestSpatialEncoder:
    def test_forward_shape(self) -> None:
        encoder = SpatialEncoder(in_channels=3, latent_dim=16, conv_channels=(4, 8))
        dummy = torch.randn(2, 3, 16, 16)

        latent = encoder(dummy)

        assert latent.shape == (2, 16)

    def test_output_size_independent_of_spatial_dims(self) -> None:
        """Real maps range from small to 2904x1672 (Europe) — the encoder
        must produce the same-shaped latent regardless of input H, W."""
        encoder = SpatialEncoder(in_channels=3, latent_dim=16, conv_channels=(4, 8))

        small = encoder(torch.randn(1, 3, 16, 16))
        large = encoder(torch.randn(1, 3, 64, 48))

        assert small.shape == large.shape == (1, 16)

    def test_single_batch_item(self) -> None:
        encoder = SpatialEncoder(in_channels=7, latent_dim=32)
        latent = encoder(torch.randn(1, 7, 32, 32))
        assert latent.shape == (1, 32)

    def test_wrong_channel_count_raises(self) -> None:
        encoder = SpatialEncoder(in_channels=7, latent_dim=32)
        with pytest.raises(ValueError, match="expected 7 input channels"):
            encoder(torch.randn(1, 3, 32, 32))

    def test_wrong_ndim_raises(self) -> None:
        encoder = SpatialEncoder(in_channels=7, latent_dim=32)
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            encoder(torch.randn(7, 32, 32))

    def test_gradients_flow(self) -> None:
        encoder = SpatialEncoder(in_channels=3, latent_dim=8)
        dummy = torch.randn(2, 3, 16, 16, requires_grad=True)

        latent = encoder(dummy)
        latent.sum().backward()

        assert dummy.grad is not None
        assert any(p.grad is not None for p in encoder.parameters())
