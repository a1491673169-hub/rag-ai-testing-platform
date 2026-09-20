import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import service
from evaluation.failure_classifier import classify_failure
from evaluation.metrics import hit_at_k, mrr, precision_at_k, recall_at_k
from evaluation.quality import answer_correctness, faithfulness_score


def load_cases():
    return [json.loads(line) for line in (ROOT / "data" / "test_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def run(top_k: int):
    results = {}
    for case in load_cases():
        response = service.chat(case["question"])
        ids = response["retrieved_document_ids"][:top_k]
        results[case["case_id"]] = {
            "status": classify_failure(case["reference_document_ids"], ids, response["answer"], case["expected_keywords"], contexts=response["contexts"], knowledge_expected=bool(case["reference_document_ids"])).value,
            "hit_at_3": hit_at_k(case["reference_document_ids"], ids, 3),
            "recall_at_3": recall_at_k(case["reference_document_ids"], ids, 3),
            "precision_at_3": precision_at_k(case["reference_document_ids"], ids, 3),
            "mrr": mrr(case["reference_document_ids"], ids),
            "answer_correctness": answer_correctness(response["answer"], case["ground_truth"], case["expected_keywords"]),
            "faithfulness": faithfulness_score(response["answer"], response["contexts"]),
        }
    return results


def summarize(results):
    keys = ["hit_at_3", "recall_at_3", "precision_at_3", "mrr", "answer_correctness", "faithfulness"]
    return {key: round(sum(item[key] for item in results.values()) / len(results), 4) for key in keys}


def main():
    baseline = run(3)
    candidate = run(5)
    regressions = [case_id for case_id in baseline if baseline[case_id]["status"] == "PASS" and candidate[case_id]["status"] != "PASS"]
    report = {"baseline": {"top_k": 3, "metrics": summarize(baseline)}, "candidate": {"top_k": 5, "metrics": summarize(candidate)}, "regression_cases": regressions, "baseline_cases": baseline, "candidate_cases": candidate}
    path = ROOT / "reports" / "regression_result.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 50)
    print("RAG 版本回归报告")
    print("=" * 50)
    print("指标                 基线 Top-K=3    候选 Top-K=5")
    for key in summarize(baseline):
        print(f"{key:<20} {summarize(baseline)[key]:.2%}          {summarize(candidate)[key]:.2%}")
    print(f"回归用例数：{len(regressions)}")
    print(f"报告文件：{path}")


if __name__ == "__main__":
    main()
