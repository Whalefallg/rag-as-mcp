"""VectorStore 工厂 (src/libs/vector_store/vector_store_factory.py)"""
from typing import Dict, Type

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.core.settings import Settings

_VECTOR_STORE_REGISTRY: Dict[str, Type[BaseVectorStore]] = {}


def register_vector_store(backend: str):
    """装饰器：将 VectorStore 实现类注册到工厂，用法：@register_vector_store("chroma")"""
    def decorator(cls: Type[BaseVectorStore]):
        _VECTOR_STORE_REGISTRY[backend] = cls
        return cls
    return decorator


# 导入内置 vector store 实现，触发自动注册。
from src.libs.vector_store import chroma_store  # noqa: F401,E402


def create_vector_store(settings: Settings) -> BaseVectorStore:
    """
    根据 settings.vector_store.backend 创建对应的 VectorStore 实例。

    Raises:
        ValueError: backend 未注册时抛出。
    """
    backend = settings.vector_store.backend
    if backend not in _VECTOR_STORE_REGISTRY:
        supported = ", ".join(_VECTOR_STORE_REGISTRY.keys())
        raise ValueError(f"不支持的 VectorStore backend: '{backend}'。已注册的后端: {supported}")

    return _VECTOR_STORE_REGISTRY[backend](persist_path=settings.vector_store.persist_path)


def get_supported_backends() -> list:
    """返回当前已注册的所有后端名称。"""
    return list(_VECTOR_STORE_REGISTRY.keys())
