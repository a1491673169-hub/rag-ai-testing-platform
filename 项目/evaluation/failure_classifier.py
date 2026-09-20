from enum import Enum
from typing import Iterable


class FailureType(str, Enum):
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    HALLUCINATION = "HALLUCINATION"
    REFUSAL_FAILURE = "REFUSAL_FAILURE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    PASS = "PASS"


def classify_failure(
    reference_document_ids: Iterable[str],
    retrieved_document_ids: Iterable[str] | None,
    answer: str | None,
    expected_keywords: Iterable[str],
    system_error: bool = False,
    contexts: list[dict] | None = None,
    knowledge_expected: bool = True,
) -> FailureType:
    if system_error:
        return FailureType.SYSTEM_ERROR
    retrieved = set(retrieved_document_ids or [])
    if not set(reference_document_ids).intersection(retrieved):
        if not knowledge_expected and answer and "没有足够信息" not in answer and "无法确认" not in answer:
            return FailureType.REFUSAL_FAILURE
        return FailureType.RETRIEVAL_FAILURE
    if contexts and answer:
        supported = " ".join(context.get("content", "") for context in contexts)
        answer_text = answer.replace("根据知识库信息：", "").strip()
        parts = [part.strip() for part in answer_text.split("；") if part.strip()]
        if parts and not all(part in supported for part in parts):
            return FailureType.HALLUCINATION
    if not answer or not all(keyword in answer for keyword in expected_keywords):
        return FailureType.GENERATION_FAILURE
    return FailureType.PASS
