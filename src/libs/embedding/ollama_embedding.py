"""
Ollama Embedding 实现 (src/libs/embedding/ollama_embedding.py)
=============================================================
为什么需要这个文件：
  本地部署 RAG 时用 nomic-embed-text/mxbai-embed-large 等开源模型，
  零 API 成本，数据不出本机。Ollama Embedding 用 /api/embed 接口，
  与 OllamaLLM 不同（后者用 /v1/chat/completions），注意区分。

本文件实现基于 Ollama 本地服务的文本向量化。

类说明:
  - OllamaEmbedding : 继承 BaseEmbedding，调用本地 Ollama 的 /api/embed 端点。
                      注意：Ollama Embedding 使用自己的 /api/embed 接口，
                      不是 OpenAI 兼容格式，需要直接用 requests 发 HTTP 请求。
                      通过 @register_embedding("ollama") 自动注册到 EmbeddingFactory，
                      settings.yaml 中设置 embedding.provider: ollama 时工厂自动创建此实例。
                      支持 nomic-embed-text / mxbai-embed-large 等本地 Embedding 模型，
                      适合完全离线部署场景。
"""
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
