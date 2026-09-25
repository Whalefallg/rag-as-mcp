"""
OpenAI-Compatible LLM 实现 (src/libs/llm/openai_llm.py)
=========================================================
为什么需要这个文件：
  OpenAI 是最常用的 LLM 供应商，gpt-4o/gpt-4-turbo 是 RAG 系统的主力模型。
  封装官方 SDK 的网络错误/认证失败，统一转为 RuntimeError，
  让上层代码不需要处理各种 OpenAI SDK 特有的异常类型。

本文件实现基于 OpenAI 官方 API 的 LLM 调用。

类说明:
  - OpenAILLM : 继承 BaseLLM，调用 OpenAI 官方接口（api.openai.com）。
                支持 gpt-4o / gpt-4-turbo / gpt-3.5-turbo 等所有 Chat Completion 模型。
                通过 @register_llm("openai") 自动注册到 LLMFactory，
                settings.yaml 中设置 llm.provider: openai 时工厂自动创建此实例。
                内部使用 openai 官方 SDK，对网络错误、认证失败等统一转换为 RuntimeError。
"""
import os
from typing import List

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_llm


@register_llm("openai")
class OpenAILLM(BaseLLM):
    """OpenAI 官方 API 实现"""

    def __init__(self, model: str, api_key: str = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        self._validate_messages(messages)
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请先安装 openai 依赖：pip install openai")

        try:
            client = OpenAI(api_key=self._api_key)
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
            raise RuntimeError(f"OpenAI API 调用失败 [{self.model}]: {e}") from e
