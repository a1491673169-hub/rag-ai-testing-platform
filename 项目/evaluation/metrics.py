from typing import Iterable


def hit_at_k(expected_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    expected = set(expected_ids)
    return float(bool(expected.intersection(list(retrieved_ids)[:k])))


def recall_at_k(expected_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    expected = set(expected_ids)
    if not expected:
        return 0.0
    found = expected.intersection(list(retrieved_ids)[:k])
    return len(found) / len(expected)


def mrr(expected_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float:
    expected = set(expected_ids)
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in expected:
            return 1.0 / rank
    return 0.0


def precision_at_k(expected_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    retrieved = list(retrieved_ids)[:k]
    if not retrieved:
        return 0.0
    return len(set(expected_ids).intersection(retrieved)) / len(retrieved)
