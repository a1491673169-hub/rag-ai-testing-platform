import pytest


@pytest.mark.parametrize("query", ["电池保修几年", "忘记密码怎么办", "充电突然中断"])
def test_retriever_returns_top_k(rag_service, query):
    documents = rag_service.retrieve(query, top_k=3)
    assert len(documents) == 3
    assert all(set(["document_id", "title", "content", "score"]).issubset(doc) for doc in documents)
    assert documents[0]["score"] >= documents[-1]["score"]


def test_retriever_empty_query_returns_empty(rag_service):
    assert rag_service.retrieve("   ") == []
