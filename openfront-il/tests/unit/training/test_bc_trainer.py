import shutil
from pathlib import Path

import pytest
import torch

from openfront_il.features import dataset_builder
from openfront_il.features.action_encoder import ACTION_TYPES, UNIT_TYPES
from openfront_il.features.observation_encoder import FLAT_FEATURES, SPATIAL_CHANNELS
from openfront_il.models.checkpoint_format import load_checkpoint
from openfront_il.training import bc_trainer
from openfront_il.training.config import ModelConfig, TrainingConfig
from openfront_il.utils import paths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


class TestCollateExamples:
    def _item(self, height: int, width: int) -> dict[str, torch.Tensor]:
        return {
            "spatial": torch.ones(len(SPATIAL_CHANNELS), height, width),
            "flat": torch.randn(len(FLAT_FEATURES)),
            "action_type": torch.tensor(1, dtype=torch.long),
            "unit_type": torch.tensor(2, dtype=torch.long),
            "action_type_legal": torch.ones(len(ACTION_TYPES), dtype=torch.bool),
        }

    def test_pads_different_map_sizes_to_batch_max(self) -> None:
        batch = [self._item(4, 6), self._item(10, 3)]

        collated = bc_trainer.collate_examples(batch)

        assert collated["spatial"].shape == (2, len(SPATIAL_CHANNELS), 10, 6)

    def test_padding_is_zero_not_repeated_content(self) -> None:
        batch = [self._item(4, 4), self._item(8, 8)]

        collated = bc_trainer.collate_examples(batch)

        # The smaller example's padded region (beyond its original 4x4) must be zero.
        small_padded = collated["spatial"][0]
        assert torch.all(small_padded[:, :4, :4] == 1.0)
        assert torch.all(small_padded[:, 4:, :] == 0.0)
        assert torch.all(small_padded[:, :, 4:] == 0.0)

    def test_same_size_batch_is_unchanged(self) -> None:
        batch = [self._item(5, 5), self._item(5, 5)]

        collated = bc_trainer.collate_examples(batch)

        assert collated["spatial"].shape == (2, len(SPATIAL_CHANNELS), 5, 5)
        assert torch.all(collated["spatial"] == 1.0)

    def test_stacks_non_spatial_fields_normally(self) -> None:
        batch = [self._item(4, 4), self._item(4, 4)]

        collated = bc_trainer.collate_examples(batch)

        assert collated["flat"].shape == (2, len(FLAT_FEATURES))
        assert collated["action_type"].shape == (2,)
        assert collated["unit_type"].shape == (2,)
        assert collated["action_type_legal"].shape == (2, len(ACTION_TYPES))


class TestGameSplitBucket:
    def test_deterministic(self) -> None:
        assert bc_trainer.game_split_bucket("abc123") == bc_trainer.game_split_bucket("abc123")

    def test_in_range(self) -> None:
        for game_id in ("abc123", "xyz789", "USR3SJiv", "vmVhD8B3"):
            bucket = bc_trainer.game_split_bucket(game_id)
            assert 0 <= bucket < 100


class TestSplitGameIds:
    def test_every_game_assigned_exactly_once(self) -> None:
        game_ids = [f"game{i}" for i in range(50)]
        train, val, test = bc_trainer.split_game_ids(game_ids, val_fraction=0.2, test_fraction=0.2)
        assert sorted(train + val + test) == sorted(game_ids)
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)

    def test_stable_across_calls_with_more_games(self) -> None:
        """Adding a new game must not reshuffle existing games' splits —
        see docs/imitation_learning_plan.md Section 4."""
        base_ids = [f"game{i}" for i in range(20)]
        train1, val1, test1 = bc_trainer.split_game_ids(
            base_ids, val_fraction=0.2, test_fraction=0.2
        )

        extended_ids = [*base_ids, "a-brand-new-game"]
        train2, val2, test2 = bc_trainer.split_game_ids(
            extended_ids, val_fraction=0.2, test_fraction=0.2
        )

        assert set(train1) <= set(train2)
        assert set(val1) <= set(val2)
        assert set(test1) <= set(test2)

    def test_zero_val_and_test_puts_everything_in_train(self) -> None:
        game_ids = ["a", "b", "c"]
        train, val, test = bc_trainer.split_game_ids(game_ids, val_fraction=0.0, test_fraction=0.0)
        assert sorted(train) == sorted(game_ids)
        assert val == []
        assert test == []


class TestMaskedCrossEntropy:
    def test_matches_plain_cross_entropy_when_all_legal(self) -> None:
        logits = torch.randn(4, 5)
        target = torch.tensor([0, 1, 2, 3])
        legal = torch.ones(4, 5, dtype=torch.bool)

        loss = bc_trainer.masked_cross_entropy(logits, target, legal)
        expected = torch.nn.functional.cross_entropy(logits, target)

        torch.testing.assert_close(loss, expected)

    def test_does_not_produce_nan_when_true_label_is_masked_illegal(self) -> None:
        logits = torch.randn(2, 3)
        target = torch.tensor([0, 1])
        # Mark every class illegal, including the true labels.
        legal = torch.zeros(2, 3, dtype=torch.bool)

        loss = bc_trainer.masked_cross_entropy(logits, target, legal)

        assert torch.isfinite(loss)

    def test_illegal_classes_get_near_zero_probability(self) -> None:
        logits = torch.tensor([[10.0, 10.0, 10.0]])
        target = torch.tensor([0])
        legal = torch.tensor([[True, False, False]])

        # A high loss here would mean the illegal classes (also logit=10)
        # are still competing for probability mass.
        loss = bc_trainer.masked_cross_entropy(logits, target, legal)

        assert loss.item() < 0.01


class TestBuildModelForConfig:
    def test_shapes_match_features_constants(self) -> None:
        policy, metadata = bc_trainer.build_model_for_config(ModelConfig())

        assert metadata.spatial_encoder_config.in_channels == len(SPATIAL_CHANNELS)
        assert metadata.policy_network_config.flat_dim == len(FLAT_FEATURES)
        assert metadata.policy_network_config.action_type_count == len(ACTION_TYPES)
        assert metadata.policy_network_config.unit_type_count == len(UNIT_TYPES)

        spatial = torch.randn(2, len(SPATIAL_CHANNELS), 16, 16)
        flat = torch.randn(2, len(FLAT_FEATURES))
        logits = policy(spatial, flat)
        assert logits["action_type"].shape == (2, len(ACTION_TYPES))
        assert logits["unit_type"].shape == (2, len(UNIT_TYPES))


class TestTrainEndToEnd:
    @pytest.fixture(autouse=True)
    def _clean_data_dirs(self) -> None:
        yield
        shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)
        shutil.rmtree(paths.PROCESSED_DIR, ignore_errors=True)

    def _build_processed_fixture(self) -> None:
        paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
        dataset_builder.build_dataset_for_game("USR3SJiv")

    def test_trains_and_writes_a_loadable_checkpoint(self, tmp_path: Path) -> None:
        self._build_processed_fixture()
        training_config = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            epochs=2,
            seed=0,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            early_stopping_patience=10,
            # A single small game can't support a real val/test split.
            val_fraction=0.0,
            test_fraction=0.0,
        )
        model_config = ModelConfig()
        model_config.spatial_encoder.conv_channels = [4, 8]
        model_config.spatial_encoder.latent_dim = 8
        model_config.policy_network.hidden_dim = 8

        checkpoint_path = bc_trainer.train(training_config, model_config)

        assert checkpoint_path.exists()
        policy, metadata = load_checkpoint(checkpoint_path)
        assert metadata.seed == 0
        assert metadata.training_config["epochs"] == 2

        # The loaded policy must still run inference.
        spatial = torch.randn(1, len(SPATIAL_CHANNELS), 16, 16)
        flat = torch.randn(1, len(FLAT_FEATURES))
        policy.eval()
        with torch.no_grad():
            logits = policy(spatial, flat)
        assert logits["action_type"].shape == (1, len(ACTION_TYPES))

    def test_raises_with_no_processed_games(self) -> None:
        training_config = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            epochs=1,
            seed=0,
            checkpoint_dir="unused",
            early_stopping_patience=1,
        )
        try:
            bc_trainer.train(training_config, ModelConfig())
        except RuntimeError as error:
            assert "no processed games" in str(error)
            return
        raise AssertionError("expected RuntimeError")
