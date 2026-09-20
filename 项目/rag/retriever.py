import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Small, deterministic retriever used to make the RAG test chain explainable."""

    def __init__(self, knowledge_base_path: str | Path):
        self.knowledge_base_path = Path(knowledge_base_path)
        with self.knowledge_base_path.open("r", encoding="utf-8") as file:
            self.documents: list[dict[str, Any]] = json.load(file)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self.document_matrix = self.vectorizer.fit_transform(
            [self._document_text(document) for document in self.documents]
        )

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        return " ".join(
            [document["title"], document["content"], document["category"]]
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        top_k = max(1, min(top_k, len(self.documents)))
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix)[0]
        ranked_indexes = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [
            {
                "document_id": self.documents[index]["document_id"],
                "title": self.documents[index]["title"],
                "content": self.documents[index]["content"],
                "score": round(float(scores[index]), 6),
            }
            for index in ranked_indexes
        ]
