import pytest
import torch

from openfront_il.models.spatial_encoder import SpatialEncoder


class TestSpatialEncoder:
    def test_forward_shape(self) -> None:
        encoder = SpatialEncoder(in_channels=3, latent_dim=16, conv_channels=(4, 8))
        dummy = torch.randn(2, 3, 16, 16)

        feature_map, latent = encoder(dummy)

        assert latent.shape == (2, 16)
        # Two MaxPool2d(2) layers -> 1/4 resolution, last conv's channels.
        assert feature_map.shape == (2, 8, 4, 4)

    def test_latent_size_independent_of_spatial_dims(self) -> None:
        """Real maps range from small to 2904x1672 (Europe) — the encoder
        must produce the same-shaped latent regardless of input H, W."""
        encoder = SpatialEncoder(in_channels=3, latent_dim=16, conv_channels=(4, 8))

        _, small = encoder(torch.randn(1, 3, 16, 16))
        _, large = encoder(torch.randn(1, 3, 64, 48))

        assert small.shape == large.shape == (1, 16)

    def test_feature_map_keeps_spatial_layout(self) -> None:
        """The feature map is what the tile_target head reads — it must
        scale with the map, not collapse to a vector."""
        encoder = SpatialEncoder(in_channels=3, latent_dim=16, conv_channels=(4, 8))

        feature_map, _ = encoder(torch.randn(1, 3, 64, 48))

        assert feature_map.shape == (1, 8, 16, 12)

    def test_uint8_input_matches_float_input(self) -> None:
        """features/ sends 0/1 channels as uint8; the encoder must treat
        them exactly like the equivalent floats."""
        encoder = SpatialEncoder(in_channels=3, latent_dim=8, conv_channels=(4, 8))
        binary = torch.randint(0, 2, (2, 3, 16, 16), dtype=torch.uint8)

        from_uint8 = encoder(binary)
        from_float = encoder(binary.float())

        for got, expected in zip(from_uint8, from_float, strict=True):
            torch.testing.assert_close(got, expected)

    def test_single_batch_item(self) -> None:
        encoder = SpatialEncoder(in_channels=7, latent_dim=32)
        _, latent = encoder(torch.randn(1, 7, 32, 32))
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

        feature_map, latent = encoder(dummy)
        (feature_map.sum() + latent.sum()).backward()

        assert dummy.grad is not None
        assert any(p.grad is not None for p in encoder.parameters())


class TestDownsampleFactor:
    def test_is_two_per_conv_layer(self) -> None:
        assert SpatialEncoder(in_channels=1, conv_channels=(4,)).downsample_factor == 2
        assert SpatialEncoder(in_channels=1, conv_channels=(4, 8, 16)).downsample_factor == 8


class TestDownsampleMask:
    def test_matches_feature_map_shape_for_uneven_sizes(self) -> None:
        encoder = SpatialEncoder(in_channels=1, conv_channels=(2, 2, 2))
        spatial = torch.randn(1, 1, 37, 29)

        feature_map, _ = encoder(spatial)
        cell_mask = encoder.downsample_mask(torch.ones(1, 37, 29, dtype=torch.bool))

        assert cell_mask.shape == feature_map.shape[0:1] + feature_map.shape[2:]

    def test_cell_is_legal_if_any_tile_in_it_is(self) -> None:
        encoder = SpatialEncoder(in_channels=1, conv_channels=(2, 2))  # factor 4
        tile_mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        tile_mask[0, 5, 2] = True  # inside cell (row 1, column 0)

        cell_mask = encoder.downsample_mask(tile_mask)

        expected = torch.tensor([[[False, False], [True, False]]])
        assert torch.equal(cell_mask, expected)


class TestCellIndex:
    def test_row_major_index_of_containing_cell(self) -> None:
        encoder = SpatialEncoder(in_channels=1, conv_channels=(2, 2))  # factor 4

        index = encoder.cell_index(torch.tensor([9]), torch.tensor([5]), 3, 4)

        # Tile (x=9, y=5) -> cell (row 1, column 2) -> 1 * 4 + 2.
        assert index.tolist() == [6]

    def test_clamps_edge_tiles_dropped_by_floor_division(self) -> None:
        encoder = SpatialEncoder(in_channels=1, conv_channels=(2, 2))  # factor 4
        # A 10x10 map gives a 2x2 feature map; tile (9, 9) would be cell (2, 2).

        index = encoder.cell_index(torch.tensor([9]), torch.tensor([9]), 2, 2)

        assert index.tolist() == [3]
