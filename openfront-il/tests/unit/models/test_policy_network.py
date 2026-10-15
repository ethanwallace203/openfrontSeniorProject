import pytest
import torch

from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.models.spatial_encoder import SpatialEncoder

PLAYER_FEATURE_DIM = 5


def _make_policy(
    flat_dim: int = 12, action_type_count: int = 19, unit_type_count: int = 17
) -> PolicyNetwork:
    encoder = SpatialEncoder(in_channels=7, latent_dim=16, conv_channels=(4, 8))
    return PolicyNetwork(
        spatial_encoder=encoder,
        flat_dim=flat_dim,
        action_type_count=action_type_count,
        unit_type_count=unit_type_count,
        player_feature_dim=PLAYER_FEATURE_DIM,
    )


def _inputs(
    batch_size: int = 4, candidates: int = 3, height: int = 16, width: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, 7, height, width),
        torch.randn(batch_size, 12),
        torch.randn(batch_size, candidates, PLAYER_FEATURE_DIM),
        torch.ones(batch_size, candidates, dtype=torch.bool),
    )


class TestPolicyNetwork:
    def test_forward_returns_all_heads_with_correct_shapes(self) -> None:
        policy = _make_policy()

        logits = policy(*_inputs(batch_size=4, candidates=3))

        assert set(logits.keys()) == {"action_type", "unit_type", "tile_target", "target_player"}
        assert logits["action_type"].shape == (4, 19)
        assert logits["unit_type"].shape == (4, 17)
        # conv_channels=(4, 8) -> 1/4 resolution: one logit per feature-map cell.
        assert logits["tile_target"].shape == (4, 4, 4)
        assert logits["target_player"].shape == (4, 3)

    def test_tile_target_scales_with_map_size(self) -> None:
        policy = _make_policy()
        logits = policy(*_inputs(batch_size=1, height=32, width=24))
        assert logits["tile_target"].shape == (1, 8, 6)

    def test_target_player_handles_any_candidate_count(self) -> None:
        policy = _make_policy()
        assert policy(*_inputs(candidates=1))["target_player"].shape == (4, 1)
        assert policy(*_inputs(candidates=9))["target_player"].shape == (4, 9)

    def test_padded_candidates_never_win(self) -> None:
        policy = _make_policy()
        spatial, flat, player_features, player_valid = _inputs(batch_size=2, candidates=4)
        player_valid[:, 2:] = False

        logits = policy(spatial, flat, player_features, player_valid)

        probabilities = logits["target_player"].softmax(dim=-1)
        assert torch.all(probabilities[:, 2:] == 0.0)
        assert torch.all(torch.isfinite(logits["target_player"]))

    def test_argument_heads_are_conditioned_on_action_type(self) -> None:
        policy = _make_policy()
        policy.eval()
        inputs = _inputs(batch_size=1)

        with torch.no_grad():
            as_spawn = policy(*inputs, action_type=torch.tensor([1]))
            as_build = policy(*inputs, action_type=torch.tensor([4]))

        assert not torch.allclose(as_spawn["tile_target"], as_build["tile_target"])
        assert not torch.allclose(as_spawn["target_player"], as_build["target_player"])
        # The action_type head itself never depends on the conditioning.
        torch.testing.assert_close(as_spawn["action_type"], as_build["action_type"])

    def test_omitted_action_type_conditions_on_argmax(self) -> None:
        policy = _make_policy()
        policy.eval()
        inputs = _inputs(batch_size=2)

        with torch.no_grad():
            implicit = policy(*inputs)
            explicit = policy(*inputs, action_type=implicit["action_type"].argmax(dim=-1))

        torch.testing.assert_close(implicit["tile_target"], explicit["tile_target"])

    def test_batch_size_one(self) -> None:
        policy = _make_policy()
        logits = policy(*_inputs(batch_size=1))
        assert logits["action_type"].shape == (1, 19)

    def test_flat_dim_mismatch_raises(self) -> None:
        policy = _make_policy(flat_dim=12)
        spatial, _, player_features, player_valid = _inputs(batch_size=2)
        with pytest.raises(ValueError, match="flat"):
            policy(spatial, torch.randn(2, 5), player_features, player_valid)

    def test_player_feature_dim_mismatch_raises(self) -> None:
        policy = _make_policy()
        spatial, flat, _, player_valid = _inputs(batch_size=2, candidates=3)
        with pytest.raises(ValueError, match="player_features"):
            policy(spatial, flat, torch.randn(2, 3, PLAYER_FEATURE_DIM + 1), player_valid)

    def test_player_valid_shape_mismatch_raises(self) -> None:
        policy = _make_policy()
        spatial, flat, player_features, _ = _inputs(batch_size=2, candidates=3)
        with pytest.raises(ValueError, match="player_valid"):
            policy(spatial, flat, player_features, torch.ones(2, 4, dtype=torch.bool))

    def test_gradients_flow_through_all_heads(self) -> None:
        policy = _make_policy()

        logits = policy(*_inputs(batch_size=2))
        loss = sum(head.sum() for head in logits.values())
        loss.backward()

        assert any(p.grad is not None for p in policy.spatial_encoder.parameters())
        assert policy.action_type_head.weight.grad is not None
        assert policy.unit_type_head.weight.grad is not None
        assert policy.tile_key.weight.grad is not None
        assert policy.player_query.weight.grad is not None

    def test_tile_head_trains_the_spatial_encoder(self) -> None:
        """The point of the tile head: its loss alone must reach the conv
        filters, so behavior cloning teaches them *where*, not just *what*."""
        policy = _make_policy()

        logits = policy(*_inputs(batch_size=2))
        logits["tile_target"].sum().backward()

        first_conv = policy.spatial_encoder.conv[0]
        assert first_conv.weight.grad is not None
        assert first_conv.weight.grad.abs().sum() > 0

    def test_head_sizes_are_configurable_without_importing_features(self) -> None:
        """models/ must not import from features/ — head sizes are plain
        constructor arguments, not action_encoder.ACTION_TYPES/UNIT_TYPES."""
        policy = _make_policy(action_type_count=3, unit_type_count=2)
        logits = policy(*_inputs(batch_size=1))
        assert logits["action_type"].shape == (1, 3)
        assert logits["unit_type"].shape == (1, 2)
