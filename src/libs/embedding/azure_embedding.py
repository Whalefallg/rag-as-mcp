"""Azure Embedding 实现 (src/libs/embedding/azure_embedding.py)"""
import os
from typing import List

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_embedding

DEFAULT_BATCH_SIZE = 100


@register_embedding("azure")
class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding 服务实现"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
        azure_endpoint: str = None,
        api_version: str = "2024-02-01",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._api_version = api_version
        self._batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)

    def embed(self, texts: List[str], trace=None) -> List[List[float]]:
        self._validate_texts(texts)
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
            results = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i: i + self._batch_size]
                response = client.embeddings.create(model=self.model, input=batch)
                results.extend([item.embedding for item in response.data])
            return results
        except Exception as e:
            raise RuntimeError(f"Azure Embedding API 调用失败 [{self.model}]: {e}") from e
