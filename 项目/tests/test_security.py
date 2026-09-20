import json
from pathlib import Path

import pytest

from rag.generator import MockGenerator


def load_security_cases():
    path = Path(__file__).resolve().parent.parent / "data" / "security_test_cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", load_security_cases(), ids=lambda case: case["case_id"])
def test_mock_generator_rejects_security_instruction(case):
    answer = MockGenerator().generate(case["question"], [])
    assert "不能" in answer
    assert all(keyword in answer for keyword in case["expected_keywords"])
