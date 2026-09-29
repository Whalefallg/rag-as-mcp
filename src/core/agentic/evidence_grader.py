import json
import re
from typing import List, Optional

from src.core.agentic.types import EvidenceAssessment, RetrievalPlan
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatMessage


class EvidenceGrader:
    def __init__(self, min_results: int = 2, min_confidence: float = 0.45, use_llm: bool = False, llm: Optional[BaseLLM] = None):
        self._min_results = min_results
        self._min_confidence = min_confidence
        self._use_llm = use_llm
        self._llm = llm
        self.llm_calls = 0

    def grade(self, query: str, results: List[RetrievalResult], plan: RetrievalPlan) -> EvidenceAssessment:
        deterministic = self._deterministic(query, results, plan)
        if not self._use_llm or self._llm is None or not results:
            return deterministic
        try:
            self.llm_calls += 1
            evidence = "\n".join(r.text[:500] for r in results[:6])
            response = self._llm.chat([ChatMessage("user", f'Return JSON only: {{"sufficient":bool,"confidence":0..1,"coverage":0..1,"reason_codes":[]}}. Query: {query}\nEvidence:\n{evidence}')])
            data = json.loads(response.content)
            return EvidenceAssessment(
                sufficient=bool(data["sufficient"]),
                confidence=max(0.0, min(1.0, float(data["confidence"]))),
                coverage=max(0.0, min(1.0, float(data["coverage"]))),
                reason_codes=[str(x) for x in data.get("reason_codes", [])],
                reasoning_summary="LLM-assisted evidence assessment.",
            )
        except Exception:
            deterministic.degraded = True
            deterministic.reasoning_summary += " LLM grading failed; deterministic fallback used."
            return deterministic

    def _deterministic(self, query: str, results: List[RetrievalResult], plan: RetrievalPlan) -> EvidenceAssessment:
        if not results:
            return EvidenceAssessment(False, 0.0, 0.0, ["NO_RESULTS"], "No retrievable evidence.")
        tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query) if len(t) > 1}
        text = " ".join(r.text for r in results[:10]).lower()
        overlap = sum(1 for token in tokens if token in text) / max(1, len(tokens))
        covered = 0
        for subquery in plan.subqueries:
            terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", subquery) if len(t) > 1]
            if not terms or any(term in text for term in terms):
                covered += 1
        subquery_coverage = covered / max(1, len(plan.subqueries))
        count_score = min(1.0, len(results) / max(1, self._min_results))
        source_count = len({r.metadata.get("source_path") or r.metadata.get("source") or r.chunk_id for r in results})
        diversity = min(1.0, source_count / 2.0)
        coverage = 0.65 * subquery_coverage + 0.35 * overlap
        confidence = 0.4 * count_score + 0.4 * coverage + 0.2 * diversity
        sufficient = len(results) >= self._min_results and confidence >= self._min_confidence and subquery_coverage >= 0.75
        codes = ["ENOUGH_EVIDENCE"] if sufficient else ["LOW_RELEVANCE" if overlap < 0.25 else "INSUFFICIENT_COVERAGE"]
        if len(plan.subqueries) > 1 and subquery_coverage < 1:
            codes.append("PARTIAL_MULTI_HOP_COVERAGE")
        return EvidenceAssessment(sufficient, round(confidence, 4), round(coverage, 4), codes, "Deterministic count, lexical coverage, subquery coverage, and source diversity assessment.")
