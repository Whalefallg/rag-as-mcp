"""Azure OpenAI LLM 实现 (src/libs/llm/azure_llm.py)"""
import os
from typing import List

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_llm


@register_llm("azure")
class AzureLLM(BaseLLM):
    """Azure OpenAI 服务实现"""

    def __init__(
        self,
        model: str,
        api_key: str = None,
        azure_endpoint: str = None,
        api_version: str = "2024-02-01",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._api_version = api_version

    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        self._validate_messages(messages)
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise RuntimeError("请先安装 openai 依赖：pip install openai")

        try:
            client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
            response = client.chat.completions.create(
                model=self.model,  # Azure 中对应 deployment_name
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
            raise RuntimeError(f"Azure OpenAI API 调用失败 [{self.model}]: {e}") from e
