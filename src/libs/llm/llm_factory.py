"""
LLM 工厂 (src/libs/llm/llm_factory.py)
========================================
为什么需要这个文件：
  工厂模式让「选择哪个 LLM 实现」的决策集中在一处，由 settings.yaml 驱动。
  register_llm / register_vision_llm 装饰器让每个实现文件自我注册。

  关键点：
    由于项目采用装饰器注册，工厂模块在加载时必须主动导入内置实现模块，
    否则注册表会为空，create_llm()/create_vision_llm() 会报 provider 未注册。
"""
from typing import Dict, Type

from src.libs.llm.base_llm import BaseLLM
from src.core.settings import Settings

_LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {}
_VISION_LLM_REGISTRY: Dict[str, Type] = {}


def register_llm(provider: str):
    """装饰器：将 LLM 实现类注册到工厂，用法：@register_llm("azure")"""
    def decorator(cls: Type[BaseLLM]):
        _LLM_REGISTRY[provider] = cls
        return cls
    return decorator


def register_vision_llm(provider: str):
    """装饰器：将 Vision LLM 实现类注册到工厂，用法：@register_vision_llm("azure")"""
    def decorator(cls):
        _VISION_LLM_REGISTRY[provider] = cls
        return cls
    return decorator


# 导入内置 LLM / Vision LLM 实现，触发自动注册。
from src.libs.llm import (  # noqa: F401,E402
    openai_llm,
    azure_llm,
    azure_vision_llm,
    ollama_llm,
    deepseek_llm,
)


def create_llm(settings: Settings) -> BaseLLM:
    """
    根据 settings.llm.provider 创建对应的 LLM 实例。

    Raises:
        ValueError: provider 未注册时抛出，错误信息包含已支持的列表。
    """
    provider = settings.llm.provider
    if provider not in _LLM_REGISTRY:
        supported = ", ".join(_LLM_REGISTRY.keys()) or "（暂无已注册的 provider）"
        raise ValueError(f"不支持的 LLM provider: '{provider}'。已注册的 provider: {supported}")

    kwargs = {"model": settings.llm.model}
    if settings.llm.api_key:
        kwargs["api_key"] = settings.llm.api_key
    if provider == "azure" and settings.llm.azure_endpoint:
        kwargs["azure_endpoint"] = settings.llm.azure_endpoint

    return _LLM_REGISTRY[provider](**kwargs)


def create_vision_llm(settings: Settings):
    """
    根据配置创建 Vision LLM 实例。

    优先读取 settings.raw_config 中的 vision_llm 配置段；
    若不存在，则回退到 settings.llm 配置（假设同一个 provider 也支持 Vision）。

    Raises:
        ValueError: provider 未注册时抛出。
    """
    raw = settings.raw_config or {}
    vision_cfg = raw.get("vision_llm", {})

    provider = vision_cfg.get("provider", settings.llm.provider)
    model = vision_cfg.get("model", settings.llm.model)
    api_key = vision_cfg.get("api_key", settings.llm.api_key)
    azure_endpoint = vision_cfg.get("azure_endpoint", settings.llm.azure_endpoint)

    if provider not in _VISION_LLM_REGISTRY:
        supported = ", ".join(_VISION_LLM_REGISTRY.keys()) or "（暂无已注册的 Vision provider）"
        raise ValueError(
            f"不支持的 Vision LLM provider: '{provider}'。已注册的 provider: {supported}"
        )

    kwargs = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if provider == "azure" and azure_endpoint:
        kwargs["azure_endpoint"] = azure_endpoint

    return _VISION_LLM_REGISTRY[provider](**kwargs)


def get_supported_providers() -> list:
    """返回当前已注册的所有普通 LLM provider 名称。"""
    return list(_LLM_REGISTRY.keys())


def get_supported_vision_providers() -> list:
    """返回当前已注册的所有 Vision LLM provider 名称。"""
    return list(_VISION_LLM_REGISTRY.keys())
