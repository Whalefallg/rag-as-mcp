import re
from typing import List

from src.core.agentic.types import QueryAnalysis, QueryIntent, RetrievalPlan, RetrievalStrategy


class RetrievalPlanner:
    def __init__(self, max_subqueries: int = 3, max_candidate_results: int = 30):
        self._max_subqueries = max_subqueries
        self._max_candidates = max_candidate_results

    def plan(self, analysis: QueryAnalysis, filters=None, candidate_k: int = 10) -> RetrievalPlan:
        if analysis.should_decompose:
            subqueries = self._decompose(analysis.normalized_query)
            strategy = RetrievalStrategy.DECOMPOSED if len(subqueries) > 1 else RetrievalStrategy.MULTI_QUERY
        else:
            subqueries = [analysis.normalized_query]
            strategy = RetrievalStrategy.SPARSE_FOCUSED if analysis.intent == QueryIntent.LEXICAL else RetrievalStrategy.HYBRID
        return RetrievalPlan(
            original_query=analysis.original_query,
            normalized_query=analysis.normalized_query,
            intent=analysis.intent,
            strategy=strategy,
            subqueries=subqueries[: self._max_subqueries],
            filters=filters or {},
            candidate_k=min(self._max_candidates, max(candidate_k, candidate_k * 2 if analysis.expand_candidates else candidate_k)),
            reasoning_summary=(
                f"Split into {len(subqueries[:self._max_subqueries])} bounded retrieval queries."
                if len(subqueries) > 1 else "Use the existing hybrid retriever directly."
            ),
        )

    def replan(self, analysis: QueryAnalysis, rewritten_query: str, filters=None, candidate_k: int = 10) -> RetrievalPlan:
        replanned = QueryAnalysis(
            original_query=analysis.original_query,
            normalized_query=rewritten_query,
            intent=analysis.intent,
            has_identifier=analysis.has_identifier,
            should_decompose=False,
            expand_candidates=True,
            reasoning_summary="Corrective query uses a wider hybrid candidate pool.",
        )
        return self.plan(replanned, filters=filters, candidate_k=candidate_k)

    def _decompose(self, query: str) -> List[str]:
        pieces = re.split(r"(?:以及|并且|同时|；|;|，并|, and |\band\b)", query, flags=re.I)
        pieces = [p.strip(" ,，?？") for p in pieces if len(p.strip(" ,，?？")) >= 4]
        if len(pieces) <= 1 and re.search(r"比较|区别|差异|分别", query):
            names = re.findall(r"[A-Za-z][A-Za-z0-9 -]{1,24}|[\u4e00-\u9fff]{2,12}", query)
            meaningful = [n.strip() for n in names if n.strip() not in {"比较", "区别", "分别", "有什么区别"}]
            pieces = [f"{name} 是什么，在检索流程的哪个阶段工作" for name in meaningful[:2]]
        return (pieces or [query])[: self._max_subqueries]
