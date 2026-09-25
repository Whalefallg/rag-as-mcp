"""
Splitter 工厂 (src/libs/splitter/splitter_factory.py)
======================================================
为什么需要这个文件：
  与其他工厂对称，确保 chunk_size/chunk_overlap 从 settings
  正确注入到具体实现，不会被默认值覆盖或静默丢失。

  关键点：
    splitter 的具体实现类（如 RecursiveSplitter）依赖装饰器
    `@register_splitter("recursive")` 自注册。
    因此工厂模块在加载时必须主动导入这些内置实现，
    否则注册装饰器永远不会执行，注册表会是空的。
"""
from typing import Dict, Type

from src.libs.splitter.base_splitter import BaseSplitter
from src.core.settings import Settings

_SPLITTER_REGISTRY: Dict[str, Type[BaseSplitter]] = {}


def register_splitter(method: str):
    """装饰器：将 Splitter 实现类注册到工厂，用法：@register_splitter("recursive")"""
    def decorator(cls: Type[BaseSplitter]):
        _SPLITTER_REGISTRY[method] = cls
        return cls
    return decorator


# 导入内置 splitter 实现，触发 @register_splitter 自动注册。
# 必须放在 register_splitter 定义之后，否则装饰器尚不可用。
from src.libs.splitter import recursive_splitter  # noqa: F401,E402


def create_splitter(settings: Settings) -> BaseSplitter:
    """
    根据 settings.splitter.method 创建对应的 Splitter 实例。

    Raises:
        ValueError: method 未注册时抛出。
    """
    method = settings.splitter.method
    if method not in _SPLITTER_REGISTRY:
        supported = ", ".join(_SPLITTER_REGISTRY.keys())
        raise ValueError(f"不支持的 Splitter method: '{method}'。已注册的方法: {supported}")

    return _SPLITTER_REGISTRY[method](
        chunk_size=settings.splitter.chunk_size,
        chunk_overlap=settings.splitter.chunk_overlap,
    )


def get_supported_methods() -> list:
    """返回当前已注册的所有 Splitter 策略名称。"""
    return list(_SPLITTER_REGISTRY.keys())
