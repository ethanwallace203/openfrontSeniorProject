import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openfront_il.ingestion import orchestrator
from openfront_il.utils import paths

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture(autouse=True)
def _clean_raw_dir() -> None:
    if paths.RAW_DIR.exists():
        shutil.rmtree(paths.RAW_DIR)
    yield
    if paths.RAW_DIR.exists():
        shutil.rmtree(paths.RAW_DIR)


class TestValidateRawFiles:
    def test_missing_file(self) -> None:
        [result] = orchestrator.validate_raw_files(["doesNotExist"])
        assert result.status == "missing"

    def test_validated_file(self) -> None:
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES_DIR / "USR3SJiv.json", paths.raw_game_path("USR3SJiv"))

        [result] = orchestrator.validate_raw_files(["USR3SJiv"])

        assert result.status == "validated"
        assert result.detail is None

    def test_invalid_file(self) -> None:
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        paths.raw_game_path("garbage").write_text('{"not": "a game record"}', encoding="utf-8")

        [result] = orchestrator.validate_raw_files(["garbage"])

        assert result.status == "invalid"
        assert result.detail is not None

    def test_malformed_json(self) -> None:
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        paths.raw_game_path("broken").write_text("{not even json", encoding="utf-8")

        [result] = orchestrator.validate_raw_files(["broken"])

        assert result.status == "invalid"


class TestRunNodeFetch:
    def test_noop_for_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = MagicMock()
        monkeypatch.setattr(subprocess, "run", mock_run)

        orchestrator.run_node_fetch([])

        mock_run.assert_not_called()

    def test_succeeds_on_first_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = MagicMock(return_value=None)
        monkeypatch.setattr(subprocess, "run", mock_run)

        orchestrator.run_node_fetch(["abc123"])

        assert mock_run.call_count == 1
        args = mock_run.call_args.args[0]
        assert args[-1] == "abc123"

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = MagicMock(
            side_effect=[
                subprocess.CalledProcessError(1, "npm"),
                subprocess.CalledProcessError(1, "npm"),
                None,
            ]
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        orchestrator.run_node_fetch(["abc123"], max_attempts=3)

        assert mock_run.call_count == 3

    def test_raises_after_exhausting_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "npm"))
        monkeypatch.setattr(subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            orchestrator.run_node_fetch(["abc123"], max_attempts=2)

        assert mock_run.call_count == 2


class TestOrchestrateFetch:
    def test_skips_fetch_for_already_present_games(self, monkeypatch: pytest.MonkeyPatch) -> None:
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES_DIR / "USR3SJiv.json", paths.raw_game_path("USR3SJiv"))
        mock_fetch = MagicMock()
        monkeypatch.setattr(orchestrator, "run_node_fetch", mock_fetch)

        results = orchestrator.orchestrate_fetch(["USR3SJiv"])

        mock_fetch.assert_not_called()
        assert results == [orchestrator.FetchResult("USR3SJiv", "validated")]

    def test_fetches_only_missing_games(self, monkeypatch: pytest.MonkeyPatch) -> None:
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES_DIR / "USR3SJiv.json", paths.raw_game_path("USR3SJiv"))

        def fake_fetch(game_ids: list[str], **_kwargs: object) -> None:
            assert game_ids == ["newGame1"]
            shutil.copy(FIXTURES_DIR / "vmVhD8B3.json", paths.raw_game_path("newGame1"))

        monkeypatch.setattr(orchestrator, "run_node_fetch", fake_fetch)

        results = orchestrator.orchestrate_fetch(["USR3SJiv", "newGame1"])

        assert {r.game_id: r.status for r in results} == {
            "USR3SJiv": "validated",
            "newGame1": "validated",
        }
