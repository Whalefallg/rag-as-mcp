"""Splitter 工厂 (src/libs/splitter/splitter_factory.py)"""
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
