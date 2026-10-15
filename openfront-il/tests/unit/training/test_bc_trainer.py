import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from openfront_il.features import dataset_builder
from openfront_il.features.action_encoder import ACTION_TYPES, UNIT_TYPES
from openfront_il.features.observation_encoder import (
    FLAT_FEATURES,
    PLAYER_FEATURES,
    SPATIAL_CHANNELS,
)
from openfront_il.models.checkpoint_format import load_checkpoint
from openfront_il.training import bc_trainer
from openfront_il.training.config import ModelConfig, TrainingConfig
from openfront_il.utils import paths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


class TestCollateExamples:
    def _item(self, height: int, width: int, candidates: int = 3) -> dict[str, torch.Tensor]:
        return _item(height, width, candidates)

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
        assert collated["target_x"].shape == (2,)
        assert collated["target_player"].shape == (2,)

    def test_pads_tile_legal_with_false(self) -> None:
        batch = [self._item(4, 4), self._item(8, 8)]

        collated = bc_trainer.collate_examples(batch)

        assert collated["tile_legal"].shape == (2, 8, 8)
        assert torch.all(collated["tile_legal"][0, :4, :4])
        assert not collated["tile_legal"][0, 4:, :].any()
        assert not collated["tile_legal"][0, :, 4:].any()

    def test_pads_player_candidates_and_marks_them_invalid(self) -> None:
        batch = [self._item(4, 4, candidates=2), self._item(4, 4, candidates=5)]

        collated = bc_trainer.collate_examples(batch)

        assert collated["player_features"].shape == (2, 5, len(PLAYER_FEATURES))
        assert collated["player_valid"].tolist() == [
            [True, True, False, False, False],
            [True] * 5,
        ]
        assert not collated["player_legal"][0, 2:].any()


def _item(
    height: int,
    width: int,
    candidates: int = 3,
    *,
    action_type: str = "spawn",
    target_xy: tuple[int, int] = (-1, -1),
    target_player: int = -1,
) -> dict[str, torch.Tensor]:
    return {
        "spatial": torch.ones(len(SPATIAL_CHANNELS), height, width),
        "flat": torch.randn(len(FLAT_FEATURES)),
        "action_type": torch.tensor(ACTION_TYPES.index(action_type), dtype=torch.long),
        "unit_type": torch.tensor(0, dtype=torch.long),
        "action_type_legal": torch.ones(len(ACTION_TYPES), dtype=torch.bool),
        "target_x": torch.tensor(target_xy[0], dtype=torch.long),
        "target_y": torch.tensor(target_xy[1], dtype=torch.long),
        "tile_legal": torch.ones(height, width, dtype=torch.bool),
        "player_features": torch.randn(candidates, len(PLAYER_FEATURES)),
        "player_legal": torch.ones(candidates, dtype=torch.bool),
        "target_player": torch.tensor(target_player, dtype=torch.long),
    }


def _tiny_policy() -> bc_trainer.PolicyNetwork:
    model_config = ModelConfig()
    model_config.spatial_encoder.conv_channels = [4, 8]
    model_config.spatial_encoder.latent_dim = 8
    model_config.policy_network.hidden_dim = 16
    policy, _ = bc_trainer.build_model_for_config(model_config)
    return policy


def _training_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = dict(
        learning_rate=0.01,
        batch_size=4,
        epochs=1,
        seed=0,
        checkpoint_dir="unused",
        early_stopping_patience=1,
    )
    defaults.update(overrides)
    return TrainingConfig.model_validate(defaults)


class TestArgumentLosses:
    def _loss(self, batch: dict[str, torch.Tensor], **weights: float) -> float:
        torch.manual_seed(0)
        policy = _tiny_policy()
        loss, _ = bc_trainer._compute_loss(
            policy, batch, _training_config(**weights), torch.ones(len(ACTION_TYPES))
        )
        assert torch.isfinite(loss)
        return loss.item()

    def test_rows_without_an_argument_contribute_no_argument_loss(self) -> None:
        """A noop has no tile and no player: the argument heads' weights
        must not change the loss at all when no row takes an argument."""
        batch = bc_trainer.collate_examples(
            [_item(16, 16, action_type="noop"), _item(16, 16, action_type="noop")]
        )

        baseline = self._loss(batch, tile_target_loss_weight=0.0, target_player_loss_weight=0.0)
        weighted = self._loss(batch, tile_target_loss_weight=100.0, target_player_loss_weight=100.0)

        assert weighted == pytest.approx(baseline)

    def test_tile_loss_applies_to_rows_with_a_tile_target(self) -> None:
        batch = bc_trainer.collate_examples(
            [
                _item(16, 16, action_type="spawn", target_xy=(3, 9)),
                _item(16, 16, action_type="noop"),
            ]
        )

        without = self._loss(batch, tile_target_loss_weight=0.0)
        with_tile = self._loss(batch, tile_target_loss_weight=1.0)

        assert with_tile > without

    def test_player_loss_applies_to_rows_with_a_player_target(self) -> None:
        batch = bc_trainer.collate_examples(
            [_item(16, 16, action_type="attack", target_player=1), _item(16, 16, candidates=5)]
        )

        without = self._loss(batch, target_player_loss_weight=0.0)
        with_player = self._loss(batch, target_player_loss_weight=1.0)

        assert with_player > without

    def test_tile_head_can_learn_a_location(self) -> None:
        """End-to-end check that the spatial pathway is trainable: on a
        board whose only distinguishing feature is one owned patch, the
        tile head should learn to point at that patch's cell."""
        torch.manual_seed(0)
        policy = _tiny_policy()
        items = []
        for corner in [(1, 1), (13, 13), (1, 13), (13, 1)] * 2:
            item = _item(16, 16, action_type="build_unit", target_xy=corner)
            item["spatial"] = torch.zeros(len(SPATIAL_CHANNELS), 16, 16)
            x, y = corner
            item["spatial"][
                SPATIAL_CHANNELS.index("owned_by_viewer"), y - 1 : y + 2, x - 1 : x + 2
            ] = 1.0
            items.append(item)
        batch = bc_trainer.collate_examples(items)
        config = _training_config(learning_rate=0.01)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        weight = torch.ones(len(ACTION_TYPES))

        for _ in range(300):
            optimizer.zero_grad()
            loss, _ = bc_trainer._compute_loss(policy, batch, config, weight)
            loss.backward()
            optimizer.step()

        policy.eval()
        _, metrics = bc_trainer._evaluate(policy, [batch], config, weight)
        assert metrics["tile_target_accuracy"] == 1.0


class _StubTable:
    """Just what MapGroupedBatchSampler reads from an ExampleTable."""

    def __init__(self, games: dict[str, tuple[tuple[int, int], list[str]]]) -> None:
        self._shapes = {game: shape for game, (shape, _) in games.items()}
        self.rows = pd.DataFrame(
            [
                {"game_id": game, "action_type": ACTION_TYPES.index(action)}
                for game, (_, actions) in games.items()
                for action in actions
            ]
        )

    def map_shape(self, game_id: str) -> tuple[int, int]:
        return self._shapes[game_id]


def _stub_table() -> _StubTable:
    # Two games share a map size; a third has a different one.
    return _StubTable(
        {
            "a": ((10, 20), ["noop"] * 50 + ["attack"] * 10),
            "b": ((10, 20), ["noop"] * 50 + ["build_unit"] * 5),
            "c": ((30, 15), ["noop"] * 100 + ["boat"] * 7),
        }
    )


def _sampler(table: _StubTable, **overrides: object) -> bc_trainer.MapGroupedBatchSampler:
    options: dict[str, object] = dict(batch_size=8, noop_keep_fraction=1.0, shuffle=True, seed=0)
    options.update(overrides)
    return bc_trainer.MapGroupedBatchSampler(table, **options)  # type: ignore[arg-type]


class TestMapGroupedBatchSampler:
    def _shape_of(self, table: _StubTable, index: int) -> tuple[int, int]:
        return table.map_shape(table.rows.iloc[index]["game_id"])

    def test_every_batch_has_a_single_map_size(self) -> None:
        table = _stub_table()
        for batch in _sampler(table, noop_keep_fraction=0.3):
            assert len({self._shape_of(table, i) for i in batch}) == 1
            assert 1 <= len(batch) <= 8

    def test_keep_fraction_one_yields_every_example_exactly_once(self) -> None:
        table = _stub_table()
        indices = [i for batch in _sampler(table) for i in batch]
        assert sorted(indices) == list(range(len(table.rows)))

    def test_always_keeps_every_real_action(self) -> None:
        table = _stub_table()
        non_noop = set(np.flatnonzero(table.rows["action_type"] != ACTION_TYPES.index("noop")))

        kept = {i for batch in _sampler(table, noop_keep_fraction=0.05) for i in batch}

        assert non_noop <= kept

    def test_subsamples_noops_near_the_keep_fraction(self) -> None:
        table = _StubTable({"a": ((10, 10), ["noop"] * 10_000)})

        kept = sum(len(batch) for batch in _sampler(table, noop_keep_fraction=0.1))

        assert 850 < kept < 1150

    def test_training_resamples_each_epoch(self) -> None:
        sampler = _sampler(_stub_table(), noop_keep_fraction=0.3)
        first, second = list(sampler), list(sampler)
        assert first != second

    def test_validation_is_identical_every_epoch(self) -> None:
        sampler = _sampler(_stub_table(), noop_keep_fraction=0.3, shuffle=False)
        assert list(sampler) == list(sampler)

    def test_len_matches_batch_count_without_randomness(self) -> None:
        sampler = _sampler(_stub_table(), shuffle=False)
        assert len(sampler) == len(list(sampler))

    def test_len_estimates_batch_count_with_subsampling(self) -> None:
        sampler = _sampler(_stub_table(), noop_keep_fraction=0.3)
        assert abs(len(sampler) - len(list(sampler))) <= 3

    @pytest.mark.parametrize("keep", [0.0, -0.1, 1.5])
    def test_rejects_invalid_keep_fraction(self, keep: float) -> None:
        with pytest.raises(ValueError, match="noop_keep_fraction"):
            _sampler(_stub_table(), noop_keep_fraction=keep)

    def test_batches_from_real_table_need_no_padding(self, tmp_path: Path) -> None:
        """End to end on the fixture: collated batches are exactly the map's size."""
        paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
        dataset_builder.build_dataset_for_game("USR3SJiv")
        try:
            dataset = bc_trainer.BCDataset(["USR3SJiv"])
            loader = bc_trainer._make_loader(
                dataset,
                _training_config(batch_size=4, noop_keep_fraction=0.5),
                shuffle=True,
                device=torch.device("cpu"),
                persistent=False,
            )
            height, width = dataset.table.map_shape("USR3SJiv")
            for batch in loader:
                assert batch["spatial"].shape[2:] == (height, width)
            dataset.close()
        finally:
            shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)
            shutil.rmtree(paths.PROCESSED_DIR, ignore_errors=True)


class TestMakeLoader:
    def test_validation_workers_are_not_persistent(self, tmp_path: Path) -> None:
        paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
        dataset_builder.build_dataset_for_game("USR3SJiv")
        try:
            dataset = bc_trainer.BCDataset(["USR3SJiv"])
            config = _training_config(num_workers=2)
            device = torch.device("cpu")
            train_loader = bc_trainer._make_loader(
                dataset, config, shuffle=True, device=device, persistent=True
            )
            val_loader = bc_trainer._make_loader(
                dataset, config, shuffle=False, device=device, persistent=False
            )
            assert train_loader.persistent_workers
            assert not val_loader.persistent_workers
            dataset.close()
        finally:
            shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)
            shutil.rmtree(paths.PROCESSED_DIR, ignore_errors=True)


class TestResolveDevice:
    def test_cpu(self) -> None:
        assert bc_trainer.resolve_device("cpu") == torch.device("cpu")

    def test_auto_picks_cuda_only_when_available(self) -> None:
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert bc_trainer.resolve_device("auto").type == expected

    @pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a GPU")
    def test_explicit_cuda_without_gpu_raises(self) -> None:
        with pytest.raises(RuntimeError, match="cuda"):
            bc_trainer.resolve_device("cuda")


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
        assert metadata.policy_network_config.player_feature_dim == len(PLAYER_FEATURES)

        spatial = torch.randn(2, len(SPATIAL_CHANNELS), 16, 16)
        flat = torch.randn(2, len(FLAT_FEATURES))
        player_features = torch.randn(2, 3, len(PLAYER_FEATURES))
        player_valid = torch.ones(2, 3, dtype=torch.bool)
        logits = policy(spatial, flat, player_features, player_valid)
        assert logits["action_type"].shape == (2, len(ACTION_TYPES))
        assert logits["unit_type"].shape == (2, len(UNIT_TYPES))
        assert logits["target_player"].shape == (2, 3)


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
        player_features = torch.randn(1, 2, len(PLAYER_FEATURES))
        player_valid = torch.ones(1, 2, dtype=torch.bool)
        policy.eval()
        with torch.no_grad():
            logits = policy(spatial, flat, player_features, player_valid)
        assert logits["action_type"].shape == (1, len(ACTION_TYPES))
        assert logits["tile_target"].shape == (1, 4, 4)

    def _train_tiny(self, tmp_path: Path, **hardware: object) -> Path:
        self._build_processed_fixture()
        training_config = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            epochs=1,
            seed=0,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            early_stopping_patience=10,
            val_fraction=0.0,
            test_fraction=0.0,
            **hardware,
        )
        model_config = ModelConfig()
        model_config.spatial_encoder.conv_channels = [4, 8]
        model_config.spatial_encoder.latent_dim = 8
        model_config.policy_network.hidden_dim = 8
        return bc_trainer.train(training_config, model_config)

    def test_trains_with_background_workers(self, tmp_path: Path) -> None:
        """Workers must be able to rebuild examples from the forked
        ExampleTable and hand back collated batches."""
        checkpoint_path = self._train_tiny(tmp_path, device="cpu", num_workers=2)
        assert checkpoint_path.exists()

    def test_trains_with_noop_subsampling_and_progress_logging(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO"):
            checkpoint_path = self._train_tiny(
                tmp_path, device="cpu", noop_keep_fraction=0.5, log_every_batches=1
            )

        assert checkpoint_path.exists()
        assert any("epoch ETA" in record.getMessage() for record in caplog.records)

    def test_mixed_precision_is_ignored_on_cpu(self, tmp_path: Path) -> None:
        checkpoint_path = self._train_tiny(tmp_path, device="cpu", mixed_precision=True)
        assert checkpoint_path.exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
    def test_trains_on_gpu_with_bf16_and_loads_on_cpu(self, tmp_path: Path) -> None:
        checkpoint_path = self._train_tiny(
            tmp_path, device="cuda", num_workers=2, mixed_precision=True
        )

        # load_checkpoint maps to CPU, so a GPU-trained checkpoint must
        # still run on a machine without one.
        policy, _ = load_checkpoint(checkpoint_path)
        assert next(policy.parameters()).device.type == "cpu"
        logits = policy(
            torch.randn(1, len(SPATIAL_CHANNELS), 16, 16),
            torch.randn(1, len(FLAT_FEATURES)),
            torch.randn(1, 2, len(PLAYER_FEATURES)),
            torch.ones(1, 2, dtype=torch.bool),
        )
        assert torch.isfinite(logits["action_type"]).all()

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
