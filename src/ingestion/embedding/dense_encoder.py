"""DenseEncoder 实现 (src/ingestion/embedding/dense_encoder.py)"""
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
