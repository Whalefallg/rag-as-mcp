"""
Core Reranker 编排层 (src/core/query_engine/reranker.py)
=========================================================
为什么需要这个模块：
  HybridSearch 融合后得到 Top-K 候选，但 RRF 分数只反映"与 query 有多相关"，
  不反映"文本内容是否真的回答了这个问题"。Reranker 做精排：
    - Cross-Encoder：把 (query, chunk) 拼在一起送入模型，比 Bi-Encoder 更准，
                     但速度慢 10x+，所以只对 Top-K 小集合做（先粗排再精排）。
    - LLM Reranker：让大模型给出排序，质量最高但成本最贵。
  设计要点：Bi-Encoder 和 Cross-Encoder 的区别？
    - Bi-Encoder：query 和文档分别编码，支持离线预计算，速度快（用于初始召回）
    - Cross-Encoder：联合编码，必须在线计算，但能捕捉 query-doc 交互（用于精排）
  这个文件是 Core 层的"编排器"，不实现具体模型逻辑，
  而是调用 libs.reranker 层的具体后端，并在失败时优雅降级。

  Phase F：trace 打点，记录 backend / result_count / fallback / duration_ms。

类说明:
  - CoreReranker : Core 层 Reranker 编排器。
                   rerank() 接收 HybridSearch 的输出，调用 libs.reranker 后端精排。
                   降级机制：后端异常或超时时，直接返回原始融合排序，
                   并在 metadata 中标记 fallback=True，不抛出致命异常，
                   不阻塞用户查询。
"""
import time
from typing import List, Optional

from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.libs.reranker.reranker_factory import create_reranker


# 对外暴露的别名（兼容 query_knowledge_hub.py 中的 `from ... import Reranker`）
Reranker = None   # 在下方 class 定义后覆盖


class CoreReranker:
    """
    Core 层 Reranker 编排器：调用 libs.reranker 后端精排，失败时降级。

    实现说明：
      为什么要分 libs 层和 core 层两层？
      libs.reranker 只关心"给我 (query, texts) 还给我排序"，不关心业务对象。
      CoreReranker 负责把 List[RetrievalResult] 转为 libs 层需要的格式，
      拿到结果后再转回来，同时处理异常降级。
      这种"适配器 + 编排"模式让两层都可以独立测试和替换。
    """

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
            trace:      追踪上下文（Phase F 打点）。
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
