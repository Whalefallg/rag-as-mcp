"""DeepSeek LLM 实现 (src/libs/llm/deepseek_llm.py)"""
import os
from typing import List

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_llm

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@register_llm("deepseek")
class DeepSeekLLM(BaseLLM):
    """DeepSeek API 实现（兼容 OpenAI 格式）"""

    def __init__(self, model: str = "deepseek-chat", api_key: str = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        self._validate_messages(messages)
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请先安装 openai 依赖：pip install openai")

        try:
            client = OpenAI(api_key=self._api_key, base_url=DEEPSEEK_BASE_URL)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            return ChatResponse(
                content=choice.message.content,
                model=response.model,
                usage=usage,
            )
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 调用失败 [{self.model}]: {e}") from e
