import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import service
from evaluation.failure_classifier import FailureType, classify_failure
from evaluation.metrics import hit_at_k, mrr, precision_at_k, recall_at_k
from evaluation.quality import answer_correctness, faithfulness_score, keyword_coverage, refusal_accuracy


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    cases = load_cases(ROOT / "data" / "test_cases.jsonl")
    results = []
    for case in cases:
        try:
            response = service.chat(case["question"])
            failure = classify_failure(
                case["reference_document_ids"],
                response["retrieved_document_ids"],
                response["answer"],
                case["expected_keywords"],
                contexts=response["contexts"],
                knowledge_expected=bool(case["reference_document_ids"]),
            )
            retrieved = response["retrieved_document_ids"]
            results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "failure_type": failure.value,
                "hit_at_3": hit_at_k(case["reference_document_ids"], retrieved, 3),
                "recall_at_3": recall_at_k(case["reference_document_ids"], retrieved, 3),
                "precision_at_3": precision_at_k(case["reference_document_ids"], retrieved, 3),
                "mrr": mrr(case["reference_document_ids"], retrieved),
                "keyword_coverage": keyword_coverage(response["answer"], case["expected_keywords"]),
                "answer_correctness": answer_correctness(response["answer"], case["ground_truth"], case["expected_keywords"]),
                "faithfulness": faithfulness_score(response["answer"], response["contexts"]),
                "refusal_accuracy": refusal_accuracy(response["answer"], case["reference_document_ids"]),
                "answer": response["answer"],
                "latency_ms": response["latency_ms"],
            })
        except Exception as exc:
            results.append({"case_id": case["case_id"], "category": case["category"], "failure_type": FailureType.SYSTEM_ERROR.value, "error": str(exc)})

    total = len(results)
    average = lambda key: sum(result.get(key, 0.0) for result in results) / total if total else 0.0
    counts = {failure.value: sum(result["failure_type"] == failure.value for result in results) for failure in FailureType}
    groups = {}
    for case, result in zip(cases, results):
        for dimension in ("category", "difficulty", "case_type"):
            group = case.get(dimension, "未标注")
            key = f"{dimension}:{group}"
            groups.setdefault(key, []).append(result)
    grouped_metrics = {
        key: {
            "case_count": len(items),
            "hit_at_3": round(sum(item.get("hit_at_3", 0) for item in items) / len(items), 4),
            "recall_at_3": round(sum(item.get("recall_at_3", 0) for item in items) / len(items), 4),
            "answer_correctness": round(sum(item.get("answer_correctness", 0) for item in items) / len(items), 4),
        }
        for key, items in groups.items()
    }
    report = {
        "total_cases": total,
        "hit_at_3": round(average("hit_at_3"), 4),
        "mean_recall_at_3": round(average("recall_at_3"), 4),
        "precision_at_3": round(average("precision_at_3"), 4),
        "mrr": round(average("mrr"), 4),
        "keyword_coverage": round(average("keyword_coverage"), 4),
        "answer_correctness": round(average("answer_correctness"), 4),
        "faithfulness": round(average("faithfulness"), 4),
        "refusal_accuracy": round(average("refusal_accuracy"), 4),
        "failure_counts": counts,
        "grouped_metrics": grouped_metrics,
        "results": results,
    }
    gate = json.loads((ROOT / "config" / "quality_gate.json").read_text(encoding="utf-8"))
    gate_checks = {
        "hit_at_3": report["hit_at_3"] >= gate["hit_at_3"],
        "recall_at_3": report["mean_recall_at_3"] >= gate["recall_at_3"],
        "answer_correctness": report["answer_correctness"] >= gate["answer_correctness"],
        "critical_case_pass_rate": (
            sum(result["failure_type"] == "PASS" for case, result in zip(cases, results) if case["reference_document_ids"])
            / max(1, sum(bool(case["reference_document_ids"]) for case in cases))
            >= gate["critical_case_pass_rate"]
        ),
    }
    report["quality_gate"] = {"thresholds": gate, "checks": gate_checks, "passed": all(gate_checks.values())}
    report_path = ROOT / "reports" / "evaluation_result.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 40)
    print("RAG 智能客服质量评测报告")
    print("=" * 40)
    print(f"测试用例总数：{total}")
    print(f"Hit@3：{report['hit_at_3']:.1%}")
    print(f"Recall@3：{report['mean_recall_at_3']:.1%}")
    print(f"Precision@3：{report['precision_at_3']:.1%}")
    print(f"MRR：{report['mrr']:.2f}")
    print(f"关键词命中率：{report['keyword_coverage']:.1%}")
    print(f"Answer Correctness：{report['answer_correctness']:.1%}")
    print(f"Faithfulness：{report['faithfulness']:.1%}")
    print(f"拒答正确率：{report['refusal_accuracy']:.1%}")
    labels = {"PASS": "通过", "RETRIEVAL_FAILURE": "检索失败", "GENERATION_FAILURE": "生成失败", "HALLUCINATION": "幻觉", "REFUSAL_FAILURE": "拒答失败", "SYSTEM_ERROR": "系统异常"}
    for failure in FailureType:
        print(f"{labels[failure.value]}：{counts[failure.value]}")
    print(f"质量门禁：{'通过' if report['quality_gate']['passed'] else '失败'}")
    print("=" * 40)
    print(f"Saved: {report_path}")
    if not report["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
