"""
Ollama LLM 实现 (src/libs/llm/ollama_llm.py)
=============================================
为什么需要这个文件：
  隐私敏感场景或无网络环境需要完全本地部署。Ollama 兼容 OpenAI 接口格式，
  OllamaLLM 只替换 base_url，复用 OpenAI SDK，实现成本极低。
  适合在本机跑 llama3/qwen2 等开源模型，零 API 费用。

本文件实现基于 Ollama 本地服务的 LLM 调用。

类说明:
  - OllamaLLM : 继承 BaseLLM，调用本地运行的 Ollama HTTP API。
                Ollama 也兼容 OpenAI 的接口格式（/v1/chat/completions），
                所以复用 OpenAI SDK，只替换 base_url 指向本地端口即可。
                通过 @register_llm("ollama") 自动注册到 LLMFactory，
                settings.yaml 中设置 llm.provider: ollama 时工厂自动创建此实例。
                适合完全离线、隐私敏感场景，无 API 调用费用。
                使用前需本地安装并启动 Ollama：https://ollama.ai
"""
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
