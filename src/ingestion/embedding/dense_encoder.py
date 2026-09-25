"""
DenseEncoder 实现 (src/ingestion/embedding/dense_encoder.py)
============================================================
为什么需要这个文件：
  向量检索的「向量」就是这里生成的。DenseEncoder 把每个 chunk 文本
  送给 Embedding 模型（OpenAI/Azure/Ollama），得到高维浮点向量，
  存入 chunk.metadata["dense_vector"]。
  Ingestion 写入的向量和 Retrieval 查询的向量必须来自同一个模型，
  DenseEncoder 和 DenseRetriever 都通过 EmbeddingFactory 创建，
  只要 settings.embedding.provider 一致，两边向量空间就保证相同。

本文件实现 Chunk 的稠密向量编码器。

类说明:
  - DenseEncoder : 调用 libs.embedding（BaseEmbedding）对 Chunk 列表批量向量化。
                   输入 List[Chunk]，输出 List[Chunk]（每个 Chunk.metadata 附加 dense_vector）。
                   通过 EmbeddingFactory 根据 settings 创建具体实现，上层不感知 Provider 差异。
                   维度一致性：同一批次所有向量维度相同，由底层 Embedding 模型保证。
"""
from typing import List, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.libs.embedding.embedding_factory import create_embedding


class DenseEncoder:
    """Chunk 稠密向量编码器"""

    def __init__(self, settings: Settings):
        self._embedding = create_embedding(settings)

    def encode(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        批量对 Chunk 文本向量化，将向量写入 Chunk.metadata["dense_vector"]。

        Args:
            chunks: 待编码的 Chunk 列表。
            trace: 追踪上下文（可选）。
        Returns:
            附加了 dense_vector 的 Chunk 列表，数量与输入一致。
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self._embedding.embed(texts, trace=trace)

        result = []
        for chunk, vector in zip(chunks, vectors):
            result.append(Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                index=chunk.index,
                metadata={**chunk.metadata, "dense_vector": vector},
            ))
        return result
