import os
from typing import List
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_embedding

DEFAULT_BATCH_SIZE = 100

@register_embedding("openai")
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model="text-embedding-3-small", api_key=None, base_url=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url
        self._batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)

    def embed(self, texts: List[str], trace=None) -> List[List[float]]:
        self._validate_texts(texts)
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请先安装 openai 依赖：pip install openai")
        try:
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            client = OpenAI(**client_kwargs)
            results = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i: i + self._batch_size]
                response = client.embeddings.create(model=self.model, input=batch)
                results.extend(item.embedding for item in response.data)
            return results
        except Exception as e:
            raise RuntimeError(f"OpenAI Embedding API 调用失败 [{self.model}]: {e}") from e
