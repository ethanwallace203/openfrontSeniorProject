from pathlib import Path

import torch

from openfront_il.models.checkpoint_format import (
    FORMAT_VERSION,
    CheckpointMetadata,
    PolicyNetworkConfig,
    SpatialEncoderConfig,
    build_policy_network,
    load_checkpoint,
    save_checkpoint,
)


def _inputs(batch_size: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch_size, 7, 16, 16),
        torch.randn(batch_size, 12),
        torch.randn(batch_size, 3, 5),
        torch.ones(batch_size, 3, dtype=torch.bool),
    )


def _metadata(**overrides: object) -> CheckpointMetadata:
    defaults: dict[str, object] = dict(
        spatial_encoder_config=SpatialEncoderConfig(
            in_channels=7, latent_dim=16, conv_channels=(4, 8)
        ),
        policy_network_config=PolicyNetworkConfig(
            flat_dim=12, action_type_count=19, unit_type_count=17, player_feature_dim=5
        ),
        training_config={"lr": 1e-3, "epochs": 5},
        seed=1337,
    )
    defaults.update(overrides)
    return CheckpointMetadata.model_validate(defaults)


class TestBuildPolicyNetwork:
    def test_builds_matching_architecture(self) -> None:
        metadata = _metadata()
        policy = build_policy_network(metadata)

        logits = policy(*_inputs(2))

        assert logits["action_type"].shape == (2, 19)
        assert logits["unit_type"].shape == (2, 17)
        assert logits["tile_target"].shape == (2, 4, 4)
        assert logits["target_player"].shape == (2, 3)


class TestSaveAndLoadCheckpoint:
    def test_roundtrip_preserves_metadata(self, tmp_path: Path) -> None:
        metadata = _metadata()
        policy = build_policy_network(metadata)
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(checkpoint_path, policy, metadata)
        _, loaded_metadata = load_checkpoint(checkpoint_path)

        assert loaded_metadata == metadata

    def test_roundtrip_preserves_weights_and_forward_pass(self, tmp_path: Path) -> None:
        metadata = _metadata()
        policy = build_policy_network(metadata)
        policy.eval()
        checkpoint_path = tmp_path / "checkpoint.pt"
        inputs = _inputs(3)

        with torch.no_grad():
            original_logits = policy(*inputs)

        save_checkpoint(checkpoint_path, policy, metadata)
        loaded_policy, _ = load_checkpoint(checkpoint_path)
        loaded_policy.eval()

        with torch.no_grad():
            reloaded_logits = loaded_policy(*inputs)

        for head in ("action_type", "unit_type", "tile_target", "target_player"):
            torch.testing.assert_close(original_logits[head], reloaded_logits[head])

    def test_format_version_mismatch_raises(self, tmp_path: Path) -> None:
        metadata = _metadata()
        policy = build_policy_network(metadata)
        checkpoint_path = tmp_path / "checkpoint.pt"
        raw_metadata = metadata.model_dump()
        raw_metadata["format_version"] = FORMAT_VERSION + 999
        torch.save(
            {"metadata": raw_metadata, "model_state_dict": policy.state_dict()}, checkpoint_path
        )

        try:
            load_checkpoint(checkpoint_path)
        except ValueError as error:
            assert "format_version" in str(error)
            return
        raise AssertionError("expected ValueError")


class TestFormatVersion:
    def test_is_2_for_the_argument_heads(self) -> None:
        """v1 checkpoints have no tile_target/target_player heads — bumping
        this is what makes load_checkpoint refuse them instead of failing
        on a missing state-dict key."""
        assert FORMAT_VERSION == 2
