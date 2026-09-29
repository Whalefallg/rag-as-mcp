"""Ollama LLM 实现 (src/libs/llm/ollama_llm.py)"""
import os
from typing import List

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_llm

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


@register_llm("ollama")
class OllamaLLM(BaseLLM):
    """Ollama 本地模型实现（兼容 OpenAI 格式）"""

    def __init__(self, model: str = "llama3", base_url: str = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        self._validate_messages(messages)
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请先安装 openai 依赖：pip install openai")

        try:
            # Ollama 兼容 OpenAI 格式，base_url 指向本地端口
            client = OpenAI(
                api_key="ollama",  # Ollama 不需要真实 key，但 SDK 要求非空
                base_url=f"{self._base_url}/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
            choice = response.choices[0]
            return ChatResponse(
                content=choice.message.content,
                model=self.model,
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama API 调用失败 [{self.model}]，请确认 Ollama 已启动（{self._base_url}）: {e}"
            ) from e
