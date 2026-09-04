import numpy as np

from openfront_il.evaluation import metrics


class TestAccuracy:
    def test_all_correct(self) -> None:
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        assert metrics.accuracy(predictions, labels) == 1.0

    def test_all_wrong(self) -> None:
        predictions = np.array([1, 2, 0])
        labels = np.array([0, 1, 2])
        assert metrics.accuracy(predictions, labels) == 0.0

    def test_partial(self) -> None:
        predictions = np.array([0, 1, 1, 1])
        labels = np.array([0, 1, 2, 3])
        assert metrics.accuracy(predictions, labels) == 0.5

    def test_empty_is_zero_not_nan(self) -> None:
        result = metrics.accuracy(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        assert result == 0.0


class TestMostCommonClassBaselineAccuracy:
    def test_finds_plurality_class(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 2])
        assert metrics.most_common_class_baseline_accuracy(labels, num_classes=3) == 0.5

    def test_uniform_distribution(self) -> None:
        labels = np.array([0, 1, 2, 3])
        assert metrics.most_common_class_baseline_accuracy(labels, num_classes=4) == 0.25

    def test_empty_is_zero(self) -> None:
        assert metrics.most_common_class_baseline_accuracy(np.array([], dtype=np.int64), 5) == 0.0


class TestNonExcludedClassAccuracy:
    def test_excludes_noop_examples_from_both_numerator_and_denominator(self) -> None:
        # noop=0. Only indices 1,2 count; prediction matches at index 1 only.
        predictions = np.array([0, 5, 6])
        labels = np.array([0, 5, 7])
        result = metrics.non_excluded_class_accuracy(predictions, labels, excluded_class=0)
        assert result == 0.5

    def test_all_excluded_gives_zero(self) -> None:
        predictions = np.array([0, 0])
        labels = np.array([0, 0])
        assert metrics.non_excluded_class_accuracy(predictions, labels, excluded_class=0) == 0.0


class TestLegalityViolationRate:
    def test_no_violations(self) -> None:
        predictions = np.array([0, 1])
        legality = np.array([[True, False], [False, True]])
        assert metrics.legality_violation_rate(predictions, legality) == 0.0

    def test_all_violations(self) -> None:
        predictions = np.array([1, 0])
        legality = np.array([[True, False], [False, True]])
        assert metrics.legality_violation_rate(predictions, legality) == 1.0

    def test_partial_violations(self) -> None:
        predictions = np.array([0, 0])
        legality = np.array([[True, False], [False, True]])
        assert metrics.legality_violation_rate(predictions, legality) == 0.5

    def test_empty_is_zero(self) -> None:
        result = metrics.legality_violation_rate(
            np.array([], dtype=np.int64), np.zeros((0, 3), dtype=bool)
        )
        assert result == 0.0
