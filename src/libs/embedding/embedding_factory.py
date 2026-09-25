from typing import Dict, Type
from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import Settings

_EMBEDDING_REGISTRY: Dict[str, Type[BaseEmbedding]] = {}

def register_embedding(provider: str):
    def decorator(cls):
        _EMBEDDING_REGISTRY[provider] = cls
        return cls
    return decorator

from src.libs.embedding import openai_embedding, azure_embedding, ollama_embedding  # noqa: F401,E402

def create_embedding(settings: Settings) -> BaseEmbedding:
    provider = settings.embedding.provider
    if provider not in _EMBEDDING_REGISTRY:
        supported = ", ".join(_EMBEDDING_REGISTRY.keys())
        raise ValueError(f"不支持的 Embedding provider: '{provider}'。已注册的 provider: {supported}")
    kwargs = {"model": settings.embedding.model}
    if settings.embedding.api_key:
        kwargs["api_key"] = settings.embedding.api_key
    base_url = getattr(settings.embedding, "base_url", None)
    if base_url:
        kwargs["base_url"] = base_url
    return _EMBEDDING_REGISTRY[provider](**kwargs)

def get_supported_providers() -> list:
    return list(_EMBEDDING_REGISTRY.keys())
