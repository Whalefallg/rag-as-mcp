"""
SparseRetriever (src/core/query_engine/sparse_retriever.py)
===========================================================
为什么需要这个模块：
  BM25（Best Match 25）是经典的稀疏检索算法，在是信息检索中的基础内容。
  它不依赖 Embedding 模型，纯靠词频统计就能工作，对"精确关键词匹配"
  场景（比如代码函数名、专有名词、型号编码）比 Dense Retrieval 更准。
  设计问题：BM25 和 TF-IDF 的区别？
    - TF-IDF：TF 越高分越高，没有上限，长文档天然占优
    - BM25：TF 增益有上限（k1 参数控制饱和），加了文档长度归一化（b 参数），
            长文档不再有不公平优势

类说明:
  - SparseRetriever : BM25 关键词召回器。
                      内部流程：
                        1. keywords → bm25_indexer.query() → [{chunk_id, score}]
                        2. chunk_ids → vector_store.get_by_ids() → [{id, text, metadata}]
                        3. 合并 score + text/metadata → List[RetrievalResult]
                      之所以用两步而不是直接在 BM25 里存文本：
                      文本只存在 VectorStore（单一来源），BM25 只存索引，
                      避免数据重复存储和同步问题。
"""
from typing import List, Optional, Dict

from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.vector_store_factory import create_vector_store


class SparseRetriever:
    """
    BM25 关键词召回器：bm25_indexer.query() → get_by_ids() → RetrievalResult 列表。
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: Optional[BM25Indexer] = None,  # 注入点：测试时传预构建的 indexer
        vector_store=None,                            # 注入点：测试时传 FakeVectorStore
    ):
        self._bm25 = bm25_indexer or BM25Indexer()
        self._store = vector_store or create_vector_store(settings)

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 20,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        用 BM25 对关键词做稀疏检索，返回最相关的 top_k 个 chunk。

        Args:
            keywords: QueryProcessor 提取的关键词列表（已去停用词）。
            top_k: 召回数量。
            trace: 追踪上下文（可选）。
        Returns:
            按 BM25 分数降序排列的 RetrievalResult 列表。
        """
        if not keywords:
            return []

        # Step 1: BM25 查询，得到 [{chunk_id, score}]
        # 把关键词列表转为 {term: 1.0} 作为查询向量（等权重）
        query_terms: Dict[str, float] = {kw: 1.0 for kw in keywords}
        bm25_kwargs = {"top_k": top_k}
        if collection != "default":
            bm25_kwargs["collection"] = collection
        bm25_hits = self._bm25.query(query_terms, **bm25_kwargs)

        if not bm25_hits:
            return []

        # Step 2: 用 chunk_id 从 VectorStore 取回完整文本和 metadata
        chunk_ids = [hit["chunk_id"] for hit in bm25_hits]
        id_to_score = {hit["chunk_id"]: hit["score"] for hit in bm25_hits}

        if collection == "default":
            records = self._store.get_by_ids(chunk_ids)
        else:
            records = self._store.get_by_ids(
                chunk_ids, collection_name=collection
            )
        id_to_record = {r["id"]: r for r in records}

        # Step 3: 合并，按 BM25 分数降序排列
        results = []
        for chunk_id in chunk_ids:
            record = id_to_record.get(chunk_id)
            if record is None:
                # BM25 索引有记录但 VectorStore 没有（索引未同步），跳过
                continue
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=float(id_to_score[chunk_id]),
                text=record.get("text", ""),
                metadata=record.get("metadata", {}),
                source="sparse",
            ))

        return results
