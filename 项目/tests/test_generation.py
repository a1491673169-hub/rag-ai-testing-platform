from rag.generator import MockGenerator


def test_generator_answer_is_not_empty(rag_service):
    contexts = rag_service.retrieve("动力电池质保多久")
    assert MockGenerator().generate("动力电池质保多久", contexts)


def test_generator_contains_expected_keywords(rag_service):
    response = rag_service.chat("动力电池可以保修几年")
    assert "8年" in response["answer"]
    assert "16万公里" in response["answer"]

