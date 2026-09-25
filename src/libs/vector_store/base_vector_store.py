"""
VectorStore 抽象层 (src/libs/vector_store/base_vector_store.py)
===============================================================
为什么需要这个文件：
  VectorStore 是 RAG 系统的核心数据层，upsert/query/get_by_ids/delete
  被多个上层模块依赖。BaseVectorStore 定义契约接口，
  换 Qdrant/Pinecone 只需实现这4个方法，上层代码零改动。
  设计问题：为什么不直接用 ChromaDB？
    → 抽象层让系统不被单一数据库锁定（vendor lock-in）。

本文件定义向量数据库的统一抽象接口。

类说明:
  - BaseVectorStore : 向量存储抽象基类。所有具体后端（Chroma/Qdrant/Pinecone）
                      都必须继承它并实现以下 4 个方法：

      upsert()             : 批量写入 chunk 记录（向量 + 原文 + metadata）。
                             采用 "upsert" 语义（update or insert），同一 chunk_id
                             重复写入只保留最新版本，避免重复索引。
                             在 Ingestion Pipeline 的 Upsert 阶段被 VectorUpserter 调用。

      query()              : 向量相似度检索，输入一个 query 向量，返回最相近的 top_k 条记录。
                             支持 filters 做前置过滤（如按 collection / source 筛选）。
                             在 Retrieval 的 Dense Route 被 DenseRetriever 调用。

      get_by_ids()         : 按 chunk_id 列表批量取回完整记录（原文 + metadata）。
                             BM25 稀疏检索只返回 ID 和分数，需要用这个方法取回内容。
                             在 Retrieval 的 Sparse Route 被 SparseRetriever 调用。

      delete_by_metadata() : 按 metadata 条件批量删除记录（如按 source 删除整篇文档）。
                             在 DocumentManager.delete_document() 中被调用，
                             实现跨存储的协调删除。
"""
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
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
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
