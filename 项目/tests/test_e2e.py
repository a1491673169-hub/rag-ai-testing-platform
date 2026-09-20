import json
from pathlib import Path

import pytest


def load_cases():
    path = Path(__file__).resolve().parent.parent / "data" / "test_cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["case_id"])
def test_rag_e2e_case(client, case):
    response = client.post("/api/chat", json={"query": case["question"]})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert isinstance(body["retrieved_document_ids"], list)
