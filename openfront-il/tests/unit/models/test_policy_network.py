import pytest
import torch

from openfront_il.models.policy_network import PolicyNetwork
from openfront_il.models.spatial_encoder import SpatialEncoder


def _make_policy(
    flat_dim: int = 12, action_type_count: int = 19, unit_type_count: int = 17
) -> PolicyNetwork:
    encoder = SpatialEncoder(in_channels=7, latent_dim=16, conv_channels=(4, 8))
    return PolicyNetwork(
        spatial_encoder=encoder,
        flat_dim=flat_dim,
        action_type_count=action_type_count,
        unit_type_count=unit_type_count,
    )


class TestPolicyNetwork:
    def test_forward_returns_both_heads_with_correct_shapes(self) -> None:
        policy = _make_policy()
        spatial = torch.randn(4, 7, 16, 16)
        flat = torch.randn(4, 12)

        logits = policy(spatial, flat)

        assert set(logits.keys()) == {"action_type", "unit_type"}
        assert logits["action_type"].shape == (4, 19)
        assert logits["unit_type"].shape == (4, 17)

    def test_batch_size_one(self) -> None:
        policy = _make_policy()
        logits = policy(torch.randn(1, 7, 16, 16), torch.randn(1, 12))
        assert logits["action_type"].shape == (1, 19)

    def test_flat_dim_mismatch_raises(self) -> None:
        policy = _make_policy(flat_dim=12)
        with pytest.raises(ValueError, match="flat"):
            policy(torch.randn(2, 7, 16, 16), torch.randn(2, 5))

    def test_gradients_flow_through_both_heads(self) -> None:
        policy = _make_policy()
        spatial = torch.randn(2, 7, 16, 16)
        flat = torch.randn(2, 12)

        logits = policy(spatial, flat)
        loss = logits["action_type"].sum() + logits["unit_type"].sum()
        loss.backward()

        assert any(p.grad is not None for p in policy.spatial_encoder.parameters())
        assert policy.action_type_head.weight.grad is not None
        assert policy.unit_type_head.weight.grad is not None

    def test_head_sizes_are_configurable_without_importing_features(self) -> None:
        """models/ must not import from features/ — head sizes are plain
        constructor arguments, not action_encoder.ACTION_TYPES/UNIT_TYPES."""
        policy = _make_policy(action_type_count=3, unit_type_count=2)
        logits = policy(torch.randn(1, 7, 16, 16), torch.randn(1, 12))
        assert logits["action_type"].shape == (1, 3)
        assert logits["unit_type"].shape == (1, 2)
