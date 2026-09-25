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
import json
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
                if k not in ("dense_vector", "sparse_vector", "images", "image_refs")
                and isinstance(v, (str, int, float, bool))
            }
            image_refs = chunk.metadata.get("image_refs")
            if isinstance(image_refs, list) and image_refs:
                # Chroma metadata 不接受 list，使用 JSON 字符串持久化。
                meta["image_refs"] = json.dumps(
                    [str(image_id) for image_id in image_refs],
                    ensure_ascii=False,
                )
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

    def list_source_ids(
        self,
        source_path: str,
        collection: str = "default",
    ) -> List[str]:
        """返回指定文档当前已存在的向量记录 ID。"""
        getter = getattr(self._store, "get_by_metadata", None)
        if not callable(getter):
            raise RuntimeError(
                "当前 VectorStore 不支持按 metadata 枚举文档记录，"
                "无法安全执行文档替换"
            )
        items = getter(
            filters={"source_path": source_path},
            collection=collection,
            limit=50000,
        )
        ids: List[str] = []
        for item in items:
            chunk_id = getattr(item, "chunk_id", None)
            if chunk_id is None and isinstance(item, dict):
                chunk_id = item.get("id") or item.get("chunk_id")
            if chunk_id:
                ids.append(str(chunk_id))
        return ids

    def delete_ids(
        self,
        ids: List[str],
        collection: str = "default",
    ) -> int:
        """精确删除指定 collection 中的一组向量记录。"""
        deleter = getattr(self._store, "delete_by_ids", None)
        if not callable(deleter):
            raise RuntimeError(
                "当前 VectorStore 不支持按 ID 删除，无法清理 stale chunks"
            )
        return int(deleter(ids, collection_name=collection))
