"""
DenseRetriever (src/core/query_engine/dense_retriever.py)
==========================================================
为什么需要这个模块：
  语义检索（Dense Retrieval）是 RAG 的核心竞争力。
  相比 BM25 只能匹配关键词，Dense Retrieval 能理解"苹果公司"和"Apple Inc"
  是同一个意思，处理同义词、近义词和概念相关性。
  原理：把 query 和所有文档 chunk 都转为高维向量，用余弦相似度找最近邻。
  设计要点：为什么 Dense Retrieval 比 BM25 好？又为什么要混合使用？
    - Dense 好处：语义理解强，能处理 paraphrase
    - Dense 缺点：依赖大量训练数据，对罕见词/专有名词不如 BM25 准
    - 混合使用（Hybrid Search）才能取长补短

类说明:
  - DenseRetriever  : 语义召回器。
                      内部流程：query → embed([query]) → vector_store.query() → List[RetrievalResult]
                      支持依赖注入（embedding_client / vector_store 可从外部传入），
                      方便单元测试时用 mock 替换，不依赖真实 API。
"""
from typing import List, Optional, Dict, Any

from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.libs.embedding.embedding_factory import create_embedding
from src.libs.vector_store.vector_store_factory import create_vector_store


class DenseRetriever:
    """
    语义召回器：embed query → 向量检索 → RetrievalResult 列表。

    实现说明：
      向量检索的核心是 ANN（Approximate Nearest Neighbor），
      ChromaDB / Qdrant 底层用 HNSW 图结构实现 O(log N) 的近似检索，
      比暴力 O(N) 快得多，代价是牺牲极小的精度。
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client=None,   # 注入点：测试时传 FakeEmbedding
        vector_store=None,       # 注入点：测试时传 FakeVectorStore
    ):
        self._embedding = embedding_client or create_embedding(settings)
        self._store = vector_store or create_vector_store(settings)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        对 query 做语义向量检索，返回最相关的 top_k 个 chunk。

        Args:
            query: 用户原始查询文本。
            top_k: 召回数量（通常设大一些，后续 Fusion/Reranker 再精选）。
            filters: 元数据前置过滤，例如 {"collection": "my_kb"}。
            collection: 目标 collection 名称（对接 ChromaDB collection）。
            trace: 追踪上下文（可选）。
        Returns:
            按相似度降序排列的 RetrievalResult 列表。
        """
        if not query.strip():
            return []

        # Step 1: 把 query 向量化（与 Ingestion 时用同一个 Embedding 模型）
        vectors = self._embedding.embed([query], trace=trace)
        query_vector = vectors[0]

        # Step 2: 向量相似度检索
        # filters 中补充 collection 条件（如果 store 支持按 collection 过滤）
        resolved_filters = dict(filters or {})
        if collection != "default":
            resolved_filters["collection"] = collection

        raw_results = self._store.query(
            vector=query_vector,
            top_k=top_k,
            filters=resolved_filters or None,
            trace=trace,
        )

        # Step 3: 归一化为 RetrievalResult
        results = []
        for r in raw_results:
            results.append(RetrievalResult(
                chunk_id=r.get("id", ""),
                score=float(r.get("score", 0.0)),
                text=r.get("text", ""),
                metadata=r.get("metadata", {}),
                source="dense",
            ))
        return results
