"""ChromaDB VectorStore 实现 (src/libs/vector_store/chroma_store.py)"""
import os
from typing import List, Dict, Any, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import register_vector_store


@register_vector_store("chroma")
class ChromaStore(BaseVectorStore):
    """ChromaDB 向量存储实现"""

    def __init__(self, persist_path: str = "./data/db/chroma", **kwargs):
        super().__init__(persist_path=persist_path, **kwargs)
        self._client = None
        self._collections: Dict[str, Any] = {}

    def _get_client(self):
        """懒加载 ChromaDB 客户端"""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise RuntimeError("请先安装 chromadb 依赖：pip install chromadb")
            os.makedirs(self.persist_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_path)
        return self._client

    def get_or_create_collection(self, collection_name: str = "default"):
        """获取或创建一个 ChromaDB collection（懒加载，按名缓存）"""
        if collection_name not in self._collections:
            client = self._get_client()
            self._collections[collection_name] = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            )
        return self._collections[collection_name]

    def upsert(self, records: List[Dict[str, Any]], trace=None) -> None:
        """
        批量写入 chunk 记录。

        records 中每条记录需包含：
          - id         : chunk 唯一标识
          - text       : chunk 原文
          - metadata   : 元数据字典（需包含 collection 字段，默认 "default"）
          - dense_vector: 浮点数向量列表
        """
        if not records:
            return

        collection_name = records[0].get("metadata", {}).get("collection", "default")
        collection = self.get_or_create_collection(collection_name)

        ids = [r["id"] for r in records]
        documents = [r["text"] for r in records]
        metadatas = [r.get("metadata", {}) for r in records]
        embeddings = [r["dense_vector"] for r in records]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
        collection_name: str = "default",
        trace=None,
    ) -> List[Dict]:
        """
        向量相似度检索，返回 top_k 条最相近记录。

        Returns:
            每条记录包含 id / text / metadata / score（越小越相似，余弦距离）。
        """
        collection = self.get_or_create_collection(collection_name)

        where = filters if filters else None
        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        records = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                records.append({
                    "id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    # ChromaDB 返回的是距离（越小越相关），转换为相似度分数
                    "score": 1 - results["distances"][0][i],
                })
        return records

    def get_by_ids(self, ids: List[str], collection_name: str = "default") -> List[Dict]:
        """按 ID 批量取回完整记录。"""
        if not ids:
            return []
        collection = self.get_or_create_collection(collection_name)
        results = collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        records = []
        for i, chunk_id in enumerate(results["ids"]):
            records.append({
                "id": chunk_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return records

    def delete_by_ids(
        self,
        ids: List[str],
        collection_name: str = "default",
    ) -> int:
        """按 ID 精确删除记录，返回实际删除数量。"""
        if not ids:
            return 0
        col = self.get_or_create_collection(collection_name)
        existing = col.get(ids=list(ids), include=[])
        existing_ids = list(existing.get("ids", []))
        if existing_ids:
            col.delete(ids=existing_ids)
        return len(existing_ids)

    def delete_by_metadata(
        self,
        filter: Dict = None,
        collection_name: str = "default",
        filters: Dict = None,
        collection: str = None,
    ) -> int:
        """按 metadata 条件批量删除，返回删除数量（支持 filter= 和 filters= 两种签名）。"""
        where = filters if filters is not None else filter
        col_name = collection if collection is not None else collection_name
        if not where:
            return 0
        col = self.get_or_create_collection(col_name)
        results = col.get(where=where, include=[])
        ids_to_delete = results["ids"]
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def get_by_metadata(
        self,
        filters: Dict,
        collection: str = "default",
        limit: int = 1000,
    ):
        """按 metadata 条件查询，返回 RetrievalResult 列表（不含向量）。"""
        from src.core.types import RetrievalResult
        col = self.get_or_create_collection(collection)
        where = filters if filters else None
        items = col.get(
            where=where,
            include=["documents", "metadatas"],
            limit=limit,
        )
        results = []
        for cid, doc, meta in zip(
            items.get("ids", []),
            items.get("documents", []),
            items.get("metadatas", []),
        ):
            results.append(RetrievalResult(
                chunk_id=cid,
                text=doc or "",
                score=None,
                metadata=meta or {},
            ))
        return results

    def list_collections(self) -> list:
        """返回所有 collection 名称列表。"""
        try:
            return [c.name for c in self._get_client().list_collections()]
        except Exception:
            return []
