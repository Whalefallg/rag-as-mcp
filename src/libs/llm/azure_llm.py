"""
Azure OpenAI LLM 实现 (src/libs/llm/azure_llm.py)
===================================================
为什么需要这个文件：
  企业环境中常用 Azure OpenAI 而非直连 OpenAI——
  数据不出租户、有 SLA 保障、支持私有网络部署。
  AzureLLM 封装 Azure SDK 的 endpoint/deployment_name/api_version 差异，
  让上层代码感受不到与 OpenAILLM 的区别。

本文件实现基于 Azure OpenAI 服务的 LLM 调用。

类说明:
  - AzureLLM : 继承 BaseLLM，调用 Azure OpenAI 端点。
               与 OpenAILLM 的区别：需要额外的 azure_endpoint 和 api_version，
               model 参数对应 Azure 里的 deployment_name（部署名称，不是模型名）。
               通过 @register_llm("azure") 自动注册到 LLMFactory，
               settings.yaml 中设置 llm.provider: azure 时工厂自动创建此实例。
               适合企业合规场景，数据不出 Azure 租户。
"""
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
