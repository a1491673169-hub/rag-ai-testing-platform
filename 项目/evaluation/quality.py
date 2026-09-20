from typing import Iterable


def keyword_coverage(answer: str, expected_keywords: Iterable[str]) -> float:
    keywords = list(expected_keywords)
    if not keywords:
        return 1.0
    return sum(keyword in answer for keyword in keywords) / len(keywords)


def answer_correctness(answer: str, ground_truth: str, expected_keywords: Iterable[str]) -> float:
    """规则基线：同时参考期望关键词和答案文本，不伪装成语义模型评分。"""
    coverage = keyword_coverage(answer, expected_keywords)
    if not ground_truth:
        return coverage
    return coverage if coverage < 1.0 else 1.0


def refusal_accuracy(answer: str, reference_document_ids: Iterable[str]) -> float:
    if list(reference_document_ids):
        return 1.0
    refusal_words = ("没有足够信息", "无法从当前知识库确认", "无法确认")
    return float(any(word in answer for word in refusal_words))


def faithfulness_score(answer: str, contexts: list[dict]) -> float:
    """Rule-based Faithfulness Baseline：回答中的上下文事实必须来自 Context。"""
    if not answer:
        return 0.0
    if not contexts or all(context.get("score", 0) <= 0 for context in contexts):
        return float("没有足够信息" in answer or "无法确认" in answer)
    supported_text = " ".join(context.get("content", "") for context in contexts)
    meaningful = answer.replace("根据知识库信息：", "").strip()
    parts = [part.strip() for part in meaningful.split("；") if part.strip()]
    return float(bool(parts) and all(part in supported_text for part in parts))
