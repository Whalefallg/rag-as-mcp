"""VectorStore 抽象层 (src/libs/vector_store/base_vector_store.py)"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    def __init__(self, persist_path: str, **kwargs):
        self.persist_path = persist_path
        self.config = kwargs

    @abstractmethod
    def upsert(self, records: List[Dict[str, Any]], trace=None) -> None:
        """
        批量写入/更新记录。

        Args:
            records: 记录列表，每条包含 id/text/metadata/dense_vector/sparse_vector。
            trace: 追踪上下文（可选）。
        """
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
        collection_name: str = "default",
        trace=None,
    ) -> List[Dict]:
        """
        向量相似度检索。

        Args:
            vector: query 向量。
            top_k: 返回最相近的 top_k 条记录。
            filters: 前置过滤条件，例如 {"source": "doc.pdf"}。
            trace: 追踪上下文（可选）。
        Returns:
            记录列表，每条包含 id/text/metadata/score。
        """
        pass

    @abstractmethod
    def get_by_ids(
        self,
        ids: List[str],
        collection_name: str = "default",
    ) -> List[Dict]:
        """
        按 ID 批量取回完整记录。

        Args:
            ids: chunk_id 列表。
        Returns:
            记录列表，每条包含 id/text/metadata。不存在的 ID 跳过。
        """
        pass

    @abstractmethod
    def delete_by_metadata(self, filter: Dict) -> int:
        """
        按 metadata 条件批量删除记录。

        Args:
            filter: 过滤条件，例如 {"source": "doc.pdf"}。
        Returns:
            实际删除的记录数量。
        """
        pass
