import pytest

from evaluation.failure_classifier import FailureType, classify_failure


@pytest.mark.parametrize(
    ("retrieved", "answer", "expected"),
    [(["wrong"], "任意回答", FailureType.RETRIEVAL_FAILURE), (["right"], "不完整回答", FailureType.GENERATION_FAILURE), (["right"], "正确答案 8年", FailureType.PASS)],
)
def test_classify_failure(retrieved, answer, expected):
    assert classify_failure(["right"], retrieved, answer, ["8年"]) == expected


def test_classify_system_error():
    assert classify_failure(["right"], [], None, [], system_error=True) == FailureType.SYSTEM_ERROR


def test_classify_refusal_failure():
    assert classify_failure([], ["other"], "这是我确定的答案", [], knowledge_expected=False) == FailureType.REFUSAL_FAILURE


def test_classify_hallucination():
    contexts = [{"content": "电池质保8年", "score": 0.9}]
    assert classify_failure(["right"], ["right"], "电池质保终身", [], contexts=contexts) == FailureType.HALLUCINATION
