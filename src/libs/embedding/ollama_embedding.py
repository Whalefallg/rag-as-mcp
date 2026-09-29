"""Ollama Embedding 实现 (src/libs/embedding/ollama_embedding.py)"""
import os
from typing import List

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_embedding

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


@register_embedding("ollama")
class OllamaEmbedding(BaseEmbedding):
    """Ollama 本地 Embedding 模型实现"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)

    def embed(self, texts: List[str], trace=None) -> List[List[float]]:
        self._validate_texts(texts)
        try:
            import requests
        except ImportError:
            raise RuntimeError("请先安装 requests 依赖：pip install requests")

        try:
            # Ollama /api/embed 支持批量输入
            url = f"{self._base_url}/api/embed"
            response = requests.post(
                url,
                json={"model": self.model, "input": texts},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            # 返回格式：{"embeddings": [[...], [...]]}
            return data["embeddings"]
        except Exception as e:
            raise RuntimeError(
                f"Ollama Embedding 调用失败 [{self.model}]，"
                f"请确认 Ollama 已启动（{self._base_url}）: {e}"
            ) from e
