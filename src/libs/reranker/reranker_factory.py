"""
Reranker 工厂 (src/libs/reranker/reranker_factory.py)
======================================================
为什么需要这个文件：
  NoneReranker 在模块加载时自动注册（不依赖装饰器），
  保证 backend=none 始终可用，即使所有其他 reranker 实现都没有 import。
  其他后端用 register_reranker 装饰器自我注册，保持一致的扩展模式。

  关键点：
    对于 cross_encoder / llm 等装饰器注册的实现，工厂模块加载时必须
    主动导入内置实现模块，触发注册逻辑。
"""
from typing import Dict, Type

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.core.settings import Settings

_RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "none": NoneReranker,
}


def register_reranker(backend: str):
    """装饰器：将 Reranker 实现类注册到工厂，用法：@register_reranker("cross_encoder")"""
    def decorator(cls: Type[BaseReranker]):
        _RERANKER_REGISTRY[backend] = cls
        return cls
    return decorator


# 导入内置 reranker 实现，触发自动注册。
from src.libs.reranker import cross_encoder_reranker, llm_reranker  # noqa: F401,E402


def create_reranker(settings: Settings) -> BaseReranker:
    """
    根据 settings.rerank.backend 创建对应的 Reranker 实例。

    Raises:
        ValueError: backend 未注册时抛出。
    """
    backend = settings.rerank.backend
    if backend not in _RERANKER_REGISTRY:
        supported = ", ".join(_RERANKER_REGISTRY.keys())
        raise ValueError(f"不支持的 Reranker backend: '{backend}'。已注册的后端: {supported}")

    if backend == "none":
        return NoneReranker()

    kwargs = {}
    if settings.rerank.model:
        kwargs["model"] = settings.rerank.model
    return _RERANKER_REGISTRY[backend](**kwargs)


def get_supported_backends() -> list:
    """返回当前已注册的所有 Reranker 后端名称。"""
    return list(_RERANKER_REGISTRY.keys())
