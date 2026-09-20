import time
from typing import Any

from rag.generator import BaseGenerator
from rag.retriever import TfidfRetriever


class RAGService:
    def __init__(self, retriever: TfidfRetriever, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.retriever.retrieve(query, top_k)

    def chat(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        documents = self.retrieve(query, top_k=3)
        answer = self.generator.generate(query, documents)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return {
            "answer": answer,
            "retrieved_document_ids": [doc["document_id"] for doc in documents],
            "contexts": documents,
            "latency_ms": latency_ms,
        }
