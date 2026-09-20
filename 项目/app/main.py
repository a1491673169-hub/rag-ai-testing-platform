from pathlib import Path

from fastapi import FastAPI

from app.schemas import ChatRequest, RetrieveRequest
from rag.generator import MockGenerator
from rag.retriever import TfidfRetriever
from rag.service import RAGService

BASE_DIR = Path(__file__).resolve().parent.parent
retriever = TfidfRetriever(BASE_DIR / "data" / "knowledge_base.json")
service = RAGService(retriever, MockGenerator())
app = FastAPI(title="RAG 智能客服自动化测试与质量评测系统", description="用于演示 RAG 检索、生成和质量评测的轻量测试系统。", version="1.0.0")


@app.get("/health", summary="健康检查", description="检查应用是否可以正常提供服务。")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/retrieve", summary="知识检索", description="根据问题返回 TF-IDF Retriever 的 Top-K 文档。")
def retrieve(request: RetrieveRequest) -> dict:
    return {"query": request.query, "documents": service.retrieve(request.query, request.top_k)}


@app.post("/api/chat", summary="智能客服问答", description="执行检索、MockGenerator 生成和延迟统计。")
def chat(request: ChatRequest) -> dict:
    return service.chat(request.query)
