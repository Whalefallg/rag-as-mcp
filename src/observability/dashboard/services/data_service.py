"""DataService (src/observability/dashboard/services/data_service.py)"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkInfo:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentInfo:
    source_path: str
    collection: str
    chunk_count: int
    image_count: int = 0
    title: str = ""
    ingested_at: str = ""


class DataService:
    """
    Dashboard 数据读取服务：封装 ChromaDB 和 ImageStorage 的读操作。

    Usage:
        svc = DataService(settings)
        docs = svc.list_documents("default")
        chunks = svc.get_chunks("data/documents/default/guide.pdf", "default")
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    def list_collections(self) -> List[str]:
        """返回所有已知集合名称"""
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=self._settings.vector_store.persist_path
            )
            return [c.name for c in client.list_collections()]
        except Exception:
            return ["default"]

    def list_documents(self, collection: str = "default") -> List[DocumentInfo]:
        """从 ChromaDB 读取指定集合的文档列表（按 source_path 聚合）"""
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=self._settings.vector_store.persist_path
            )
            col = client.get_collection(collection)
            items = col.get(include=["metadatas"], limit=50000)
            metadatas = items.get("metadatas") or []

            # 按 source_path 聚合
            aggregated: Dict[str, DocumentInfo] = {}
            for m in metadatas:
                src = m.get("source_path", "unknown")
                if src not in aggregated:
                    aggregated[src] = DocumentInfo(
                        source_path=src,
                        collection=collection,
                        chunk_count=0,
                        title=m.get("title", _basename(src)),
                        ingested_at=m.get("ingested_at", ""),
                    )
                aggregated[src].chunk_count += 1

            return sorted(aggregated.values(), key=lambda d: d.source_path)
        except Exception:
            return []

    def get_chunks(
        self, source_path: str, collection: str = "default"
    ) -> List[ChunkInfo]:
        """获取指定文档的所有 chunk"""
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=self._settings.vector_store.persist_path
            )
            col = client.get_collection(collection)
            items = col.get(
                where={"source_path": source_path},
                include=["documents", "metadatas"],
                limit=1000,
            )
            ids = items.get("ids") or []
            docs = items.get("documents") or []
            metas = items.get("metadatas") or []
            return [
                ChunkInfo(chunk_id=cid, text=txt or "", metadata=meta or {})
                for cid, txt, meta in zip(ids, docs, metas)
            ]
        except Exception:
            return []

    def get_image_path(self, image_id: str) -> Optional[str]:
        """获取图片文件路径（供 st.image 展示）"""
        try:
            from src.ingestion.storage.image_storage import ImageStorage
            return ImageStorage().get_path(image_id)
        except Exception:
            return None


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
