from abc import ABC, abstractmethod
from typing import Any
import json
import os
from urllib.request import Request, urlopen


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """Deterministic context-to-answer generator; no external model is called."""

    def generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        security_words = ("忽略之前", "系统提示词", "system prompt", "其他用户的手机号", "你现在不是客服", "不要查询知识库")
        if any(word in query for word in security_words):
            return "抱歉，我不能执行越权指令或提供未授权信息。"
        if not contexts or all(context["score"] <= 0 for context in contexts):
            return "抱歉，当前知识库中没有足够信息回答这个问题。"
        selected = [context["content"] for context in contexts if context["score"] > 0]
        return "根据知识库信息：" + "；".join(selected[:2])


class OpenAICompatibleGenerator(BaseGenerator):
    """可选的 OpenAI-compatible 接口适配器；未配置密钥时不会被默认使用。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "")

    def generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        if not all((self.base_url, self.api_key, self.model)):
            raise RuntimeError("未配置 LLM_BASE_URL、LLM_API_KEY 或 LLM_MODEL")
        context_text = "\n".join(context["content"] for context in contexts)
        payload = {"model": self.model, "messages": [{"role": "user", "content": f"只依据以下知识回答：\n{context_text}\n问题：{query}"}], "temperature": 0}
        request = Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
        return body["choices"][0]["message"]["content"]
