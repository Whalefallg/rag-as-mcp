"""HybridSearch (src/core/query_engine/hybrid_search.py)"""
import time
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFusion, create_fusion


class HybridSearch:
    """Combine over-fetched dense and sparse candidates using rank fusion."""

    def __init__(
        self,
        settings: Settings,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFusion] = None,
    ):
        retrieval_cfg = settings.retrieval
        self._query_proc = query_processor or QueryProcessor()
        self._dense = dense_retriever or DenseRetriever(settings)
        self._sparse = sparse_retriever or SparseRetriever(settings)
        self._fusion_algorithm = getattr(retrieval_cfg, "fusion_algorithm", "rrf")
        self._fusion = fusion or create_fusion(self._fusion_algorithm)
        self._top_k_dense = retrieval_cfg.top_k_dense
        self._top_k_sparse = retrieval_cfg.top_k_sparse
        self._top_k_final = retrieval_cfg.top_k_final

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        执行混合检索，返回融合排序后的 Top-K 结果。

        Args:
            query: 用户原始查询文本。
            top_k: 返回结果数量，不传则使用 settings.retrieval.top_k_final。
            filters: 额外的 metadata 过滤条件。
            collection: 目标 collection 名称。
            trace: Optional trace shared by all retrieval stages.
        Returns:
            按融合分降序排列的 RetrievalResult 列表。
        """
        final_top_k = top_k or self._top_k_final

        # Step 1: 标准化 query
        if trace:
            trace.set_metadata("user_query", query)
            trace.set_metadata("collection", collection)
        processed = self._query_proc.process(query, filters=filters)

        # Step 2: Dense 召回（失败时降级为空列表）
        dense_results: List[RetrievalResult] = []
        _t0 = time.monotonic()
        try:
            dense_results = self._dense.retrieve(
                query=processed.original,
                top_k=self._top_k_dense,
                filters=processed.filters or None,
                collection=collection,
                trace=trace,
            )
            if trace:
                trace.record_stage(
                    "dense_retrieval",
                    duration_ms=(time.monotonic() - _t0) * 1000,
                    result_count=len(dense_results),
                    method="dense",
                    status="ok",
                )
        except Exception as exc:
            if trace:
                trace.record_stage(
                    "dense_retrieval",
                    duration_ms=(time.monotonic() - _t0) * 1000,
                    result_count=0,
                    error=str(exc),
                    method="dense",
                    status="degraded",
                )

        # Step 3: Sparse 召回（失败时降级为空列表）
        sparse_results: List[RetrievalResult] = []
        _t0 = time.monotonic()
        try:
            sparse_kwargs = {
                "keywords": processed.keywords,
                "top_k": self._top_k_sparse,
                "trace": trace,
            }
            if collection != "default":
                sparse_kwargs["collection"] = collection
            sparse_results = self._sparse.retrieve(**sparse_kwargs)
            if trace:
                trace.record_stage(
                    "sparse_retrieval",
                    duration_ms=(time.monotonic() - _t0) * 1000,
                    result_count=len(sparse_results),
                    method="bm25",
                    status="ok",
                )
        except Exception as exc:
            if trace:
                trace.record_stage(
                    "sparse_retrieval",
                    duration_ms=(time.monotonic() - _t0) * 1000,
                    result_count=0,
                    error=str(exc),
                    method="bm25",
                    status="degraded",
                )

        # 两路都失败
        if not dense_results and not sparse_results:
            return []

        # Step 4: RRF 融合
        _t0 = time.monotonic()
        result_lists = [r for r in [dense_results, sparse_results] if r]
        fused = self._fusion.fuse(result_lists, top_k=final_top_k)
        if trace:
            trace.record_stage(
                "fusion",
                duration_ms=(time.monotonic() - _t0) * 1000,
                algorithm=self._fusion_algorithm,
                input_lists=len(result_lists),
                result_count=len(fused),
            )

        # Step 5: 后置 metadata 过滤（兜底，应对 VectorStore 前置过滤遗漏的情况）
        if processed.filters:
            fused = self._apply_metadata_filters(fused, processed.filters)

        return fused[:final_top_k]

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """
        后置 metadata 过滤：仅保留 metadata 满足所有 filter 条件的结果。
        作为 VectorStore 前置过滤的兜底，确保过滤语义正确。
        """
        def matches(result: RetrievalResult) -> bool:
            for key, value in filters.items():
                if result.metadata.get(key) != value:
                    return False
            return True

        return [r for r in candidates if matches(r)]
