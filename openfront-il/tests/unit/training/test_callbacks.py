import pytest

from openfront_il.training.callbacks import _format_duration, log_progress


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0m00s"),
            (59.9, "0m59s"),
            (61, "1m01s"),
            (3600, "1h00m"),
            (5 * 3600 + 7 * 60, "5h07m"),
        ],
    )
    def test_formats(self, seconds: float, expected: str) -> None:
        assert _format_duration(seconds) == expected


class TestLogProgress:
    def test_reports_progress_rate_and_eta(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            # 100 of 400 batches in 10 minutes -> 30 more minutes.
            log_progress(2, 100, 400, 1.2345, 800, 600.0)

        message = caplog.records[-1].getMessage()
        assert "epoch 2: batch 100/~400 (25%)" in message
        assert "running_loss=1.2345" in message
        assert "1.3 ex/s" in message
        assert "epoch ETA 30m00s" in message
