import pytest

from evaluation.metrics import hit_at_k, mrr, precision_at_k, recall_at_k


@pytest.mark.parametrize(
    ("expected", "retrieved", "expected_hit", "expected_recall", "expected_mrr"),
    [
        (["a"], ["b", "a", "c"], 1.0, 1.0, 0.5),
        (["a", "c"], ["a", "b", "c"], 1.0, 1.0, 1.0),
        (["a"], ["b", "c"], 0.0, 0.0, 0.0),
    ],
)
def test_metrics(expected, retrieved, expected_hit, expected_recall, expected_mrr):
    assert hit_at_k(expected, retrieved, 3) == expected_hit
    assert recall_at_k(expected, retrieved, 3) == expected_recall
    assert mrr(expected, retrieved) == expected_mrr
    assert 0 <= precision_at_k(expected, retrieved, 3) <= 1
