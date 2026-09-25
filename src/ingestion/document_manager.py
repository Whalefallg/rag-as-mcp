"""
DocumentManager (src/ingestion/document_manager.py)
====================================================
为什么需要这个文件：
  文档的"删除"操作跨越四个存储：ChromaDB、BM25 索引、ImageStorage、FileIntegrity。
  如果各存储各自删除，任一步失败就会造成数据不一致（Chroma 删了但 BM25 还在）。
  DocumentManager 把四个存储的协调删除封装成一次原子性操作（尽力而为），
  提供明确的成功/失败报告。

  与 DataService 的区别：
    - DataService：只读，供 Dashboard 数据浏览使用
    - DocumentManager：读写，管理文档生命周期（list/delete/stats）

  调用方：
    - Dashboard ingestion_manager.py（删除按钮）
    - scripts/ingest.py --delete 参数（命令行删除）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentInfo:
    source_path: str
    collection: str
    chunk_count: int
    image_count: int = 0
    title: str = ""


@dataclass
class DeleteResult:
    success: bool
    source_path: str
    deleted_chunks: int = 0
    deleted_images: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionStats:
    collection: str
    document_count: int
    chunk_count: int
    image_count: int = 0


class DocumentManager:
    """
    文档生命周期管理器：协调 Chroma / BM25 / ImageStorage / FileIntegrity 的增删查。

    Usage:
        dm = DocumentManager.from_settings(settings)
        docs = dm.list_documents("default")
        result = dm.delete_document("data/documents/default/guide.pdf", "default")
    """

    def __init__(
        self,
        chroma_store,
        bm25_indexer,
        image_storage,
        file_integrity,
    ) -> None:
        self._chroma = chroma_store
        self._bm25 = bm25_indexer
        self._images = image_storage
        self._integrity = file_integrity

    @classmethod
    def from_settings(cls, settings) -> "DocumentManager":
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import ImageStorage
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker

        return cls(
            chroma_store=ChromaStore(settings),
            bm25_indexer=BM25Indexer(),
            image_storage=ImageStorage(),
            file_integrity=SQLiteIntegrityChecker(),
        )

    # ──────────────────────────────────────────────────────────────────────
    # 查询接口
    # ──────────────────────────────────────────────────────────────────────

    def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        """
        列出已摄取的文档（按 source_path 聚合）。

        Args:
            collection: 指定集合；None 时列出所有集合的文档。
        Returns:
            DocumentInfo 列表，按 source_path 排序。
        """
        try:
            collections = self._get_collections(collection)
            docs: Dict[str, DocumentInfo] = {}

            for col_name in collections:
                items = self._chroma.get_by_metadata(
                    filters={}, collection=col_name, limit=50000
                )
                for item in items:
                    src = item.metadata.get("source_path", "unknown")
                    if src not in docs:
                        docs[src] = DocumentInfo(
                            source_path=src,
                            collection=col_name,
                            chunk_count=0,
                            title=item.metadata.get("title", _basename(src)),
                        )
                    docs[src].chunk_count += 1
            return sorted(docs.values(), key=lambda d: d.source_path)
        except Exception:
            return []

    def get_collection_stats(self, collection: Optional[str] = None) -> List[CollectionStats]:
        """返回各集合的统计信息"""
        stats = []
        for col_name in self._get_collections(collection):
            try:
                items = self._chroma.get_by_metadata(
                    filters={}, collection=col_name, limit=50000
                )
                sources = {i.metadata.get("source_path") for i in items}
                stats.append(CollectionStats(
                    collection=col_name,
                    document_count=len(sources),
                    chunk_count=len(items),
                ))
            except Exception:
                stats.append(CollectionStats(
                    collection=col_name, document_count=0, chunk_count=0
                ))
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # 删除接口
    # ──────────────────────────────────────────────────────────────────────

    def delete_document(
        self, source_path: str, collection: str = "default"
    ) -> DeleteResult:
        """
        从四个存储中协调删除指定文档的所有数据。

        删除顺序（故意从最好恢复到最难恢复）：
          1. ChromaDB chunk 向量（可重新摄取）
          2. BM25 索引（可重建）
          3. ImageStorage 图片（可重新摄取）
          4. FileIntegrity 记录（最后删，否则重试会被跳过）
        """
        result = DeleteResult(source_path=source_path, success=False)
        errors = []

        # 1. Chroma
        try:
            deleted_chunks = self._chroma.delete_by_metadata(
                filters={"source_path": source_path},
                collection=collection,
            )
            result.deleted_chunks = deleted_chunks
            result.details["chroma"] = f"deleted {deleted_chunks} chunks"
        except Exception as exc:
            errors.append(f"chroma: {exc}")

        # 2. BM25
        try:
            self._bm25.remove_document(source_path)
            result.details["bm25"] = "ok"
        except Exception as exc:
            errors.append(f"bm25: {exc}")

        # 3. ImageStorage
        try:
            deleted_images = self._images.delete_by_source(source_path)
            result.deleted_images = deleted_images
            result.details["images"] = f"deleted {deleted_images} images"
        except Exception as exc:
            errors.append(f"images: {exc}")

        # 4. FileIntegrity（最后删）
        try:
            self._integrity.remove_record(source_path)
            result.details["integrity"] = "ok"
        except Exception as exc:
            errors.append(f"integrity: {exc}")

        if errors:
            result.error = "; ".join(errors)
            result.success = False
        else:
            result.success = True
        return result

    # ──────────────────────────────────────────────────────────────────────

    def _get_collections(self, collection: Optional[str]) -> List[str]:
        if collection:
            return [collection]
        try:
            return self._chroma.list_collections()
        except Exception:
            return ["default"]


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
