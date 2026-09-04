import shutil
from pathlib import Path

import pytest

from openfront_il.evaluation import held_out_report
from openfront_il.features import dataset_builder
from openfront_il.features.action_encoder import ACTION_TYPES, UNIT_TYPES
from openfront_il.models.checkpoint_format import (
    CheckpointMetadata,
    PolicyNetworkConfig,
    SpatialEncoderConfig,
    build_policy_network,
    save_checkpoint,
)
from openfront_il.utils import paths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


@pytest.fixture
def _processed_fixture() -> None:
    paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
    dataset_builder.build_dataset_for_game("USR3SJiv")
    yield
    shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)
    shutil.rmtree(paths.PROCESSED_DIR, ignore_errors=True)


def _tiny_checkpoint(tmp_path: Path) -> Path:
    from openfront_il.features.observation_encoder import FLAT_FEATURES, SPATIAL_CHANNELS

    metadata = CheckpointMetadata(
        spatial_encoder_config=SpatialEncoderConfig(
            in_channels=len(SPATIAL_CHANNELS), latent_dim=8, conv_channels=(4, 8)
        ),
        policy_network_config=PolicyNetworkConfig(
            flat_dim=len(FLAT_FEATURES),
            action_type_count=len(ACTION_TYPES),
            unit_type_count=len(UNIT_TYPES),
            hidden_dim=8,
        ),
    )
    policy = build_policy_network(metadata)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, policy, metadata)
    return checkpoint_path


class TestRunInference:
    def test_produces_one_prediction_per_example(
        self, tmp_path: Path, _processed_fixture: None
    ) -> None:
        checkpoint_path = _tiny_checkpoint(tmp_path)

        results, metadata = held_out_report.run_inference(checkpoint_path, ["USR3SJiv"])

        assert len(results.action_type_predictions) == len(results.action_type_labels)
        assert len(results.action_type_predictions) > 0
        assert results.action_type_legality_masks.shape == (
            len(results.action_type_predictions),
            len(ACTION_TYPES),
        )
        assert metadata.format_version == 1


class TestComputeMetrics:
    def test_returns_sane_ranges(self, tmp_path: Path, _processed_fixture: None) -> None:
        checkpoint_path = _tiny_checkpoint(tmp_path)
        results, metadata = held_out_report.run_inference(checkpoint_path, ["USR3SJiv"])

        result = held_out_report.compute_metrics(checkpoint_path, metadata.format_version, results)

        assert result.num_examples == len(results.action_type_labels)
        for value in (
            result.action_type_accuracy,
            result.action_type_baseline_accuracy,
            result.action_type_non_noop_accuracy,
            result.action_type_legality_violation_rate,
            result.unit_type_accuracy,
            result.unit_type_baseline_accuracy,
        ):
            assert 0.0 <= value <= 1.0


class TestEvaluateAndReport:
    def test_writes_a_markdown_report(self, tmp_path: Path, _processed_fixture: None) -> None:
        checkpoint_path = _tiny_checkpoint(tmp_path)
        output_path = tmp_path / "reports" / "held_out.md"

        result = held_out_report.evaluate_and_report(checkpoint_path, ["USR3SJiv"], output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "action_type" in content
        assert "unit_type" in content
        assert str(result.num_examples) in content
        assert "format_version=1" in content
