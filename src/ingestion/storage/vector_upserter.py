"""
VectorUpserter 实现 (src/ingestion/storage/vector_upserter.py)
==============================================================
为什么需要这个文件：
  直接调用 VectorStore.upsert() 需要调用方自己生成 ID、处理重复写入。
  VectorUpserter 把这两件事封装好：用 (source_path + chunk_index + content_hash) 生成
  确定性 ID——同内容多次写入 ID 相同，VectorStore 的 upsert 语义自动覆盖旧记录，
  不产生重复向量。这是整个 Ingestion Pipeline 幂等性的最后一道保障。

本文件实现向量存储的幂等写入器。

类说明:
  - VectorUpserter : 接收 DenseEncoder 编码后的 Chunk 列表，生成确定性 chunk_id，
                     调用 BaseVectorStore.upsert() 将向量和 metadata 写入向量数据库。
                     幂等性：同一 Chunk（source_path + chunk_index + content_hash 相同）
                             重复写入只更新，不产生重复记录，由 VectorStore upsert 语义保证。
                     chunk_id 生成规则：hash(source_path + chunk_index + content_hash[:8])[:16]
                     — 内容变更时 id 变更（触发覆盖），内容不变时 id 稳定（跳过重复写入）。
"""
import hashlib
from typing import List, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.libs.vector_store.vector_store_factory import create_vector_store


def _stable_chunk_id(source_path: str, chunk_index: int, content_hash: str) -> str:
    """生成确定性 chunk_id（16 字符 hex）"""
    raw = f"{source_path}|{chunk_index}|{content_hash[:8]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class VectorUpserter:
    """向量存储幂等写入器"""

    def __init__(self, settings: Settings):
        self._store = create_vector_store(settings)

    def upsert(
        self,
        chunks: List[Chunk],
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> int:
        """
        批量幂等写入 Chunk 向量到 VectorStore。

        Args:
            chunks: 已完成 DenseEncoder 的 Chunk 列表（metadata 含 dense_vector）。
            collection: 目标 collection 名称。
            trace: 追踪上下文（可选）。
        Returns:
            实际写入的记录数量。
        """
        if not chunks:
            return 0

        records = []
        for chunk in chunks:
            # DocumentChunker 已保证 Chunk.id 的确定性；这里不再生成第二套 ID。
            record_id = chunk.id

            # 过滤 metadata，不将向量本身放入 metadata（向量单独存）
            meta = {
                k: v for k, v in chunk.metadata.items()
                if k not in ("dense_vector", "sparse_vector", "images")
                and isinstance(v, (str, int, float, bool))
            }
            meta["collection"] = collection
            meta["chunk_id"] = chunk.id

            records.append({
                "id": record_id,
                "text": chunk.text,
                "metadata": meta,
                "dense_vector": chunk.metadata.get("dense_vector", []),
            })

        self._store.upsert(records, trace=trace)
        return len(records)
