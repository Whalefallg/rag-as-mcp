"""Core Reranker 编排层 (src/core/query_engine/reranker.py)"""
import time
from typing import List, Optional

from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.libs.reranker.reranker_factory import create_reranker


# 对外暴露的别名（兼容 query_knowledge_hub.py 中的 `from ... import Reranker`）
Reranker = None   # 在下方 class 定义后覆盖


class CoreReranker:
    """Adapt retrieval results to a reranker and preserve order on failure."""

    def __init__(self, settings: Settings, reranker=None):
        """
        Args:
            settings: 全局配置，reranker 后端由 settings.rerank.backend 决定。
            reranker: 可注入的 libs.reranker 实例（用于测试时传 mock）。
        """
        self._reranker = reranker or create_reranker(settings)
        self._backend = settings.rerank.backend

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        对候选结果精排，返回重新排序后的列表。

        Args:
            query:      用户原始查询文本。
            candidates: HybridSearch 返回的候选 RetrievalResult 列表。
            top_k:      精排后保留的结果数量，None 时保留全部。
            trace:      Optional trace for rerank timing and fallback metadata.
        Returns:
            精排后的 RetrievalResult 列表；失败时返回原始顺序并标记 fallback。
        """
        if not candidates:
            return []

        limit = top_k or len(candidates)
        _t0 = time.monotonic()

        try:
            texts = [r.text for r in candidates]
            ranked_indices = self._reranker.rerank(query, texts, top_k=limit)

            results = []
            for rank, idx in enumerate(ranked_indices):
                r = candidates[idx]
                results.append(RetrievalResult(
                    chunk_id=r.chunk_id,
                    score=1.0 / (rank + 1),
                    text=r.text,
                    metadata={**r.metadata, "rerank_rank": rank + 1},
                    source="reranked",
                ))

            if trace:
                trace.record_stage(
                    "rerank",
                    duration_ms=(time.monotonic() - _t0) * 1000,
                    backend=self._backend,
                    result_count=len(results),
                    fallback=False,
                )
            return results

        except Exception as exc:
            fallback_results = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    score=r.score,
                    text=r.text,
                    metadata={**r.metadata, "fallback": True, "fallback_reason": str(exc)},
                    source=r.source,
                )
                for r in candidates[:limit]
            ]
            if trace:
                trace.record_stage(
                    "rerank",
                    duration_ms=(time.monotonic() - _t0) * 1000,
                    backend=self._backend,
                    result_count=len(fallback_results),
                    fallback=True,
                    error=str(exc),
                )
            return fallback_results


# 别名：让 query_knowledge_hub.py 的 `from src.core.query_engine.reranker import Reranker` 正常工作
Reranker = CoreReranker
