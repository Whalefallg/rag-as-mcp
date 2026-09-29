import json
import re
from typing import Optional

from src.core.agentic.types import QueryAnalysis, QueryIntent
from src.libs.llm.base_llm import BaseLLM, ChatMessage

_IDENTIFIER = re.compile(
    r"(?:\b[A-Z][A-Z0-9_]{2,}\b|\b[A-Za-z_][\w.]*\(\)|\b[A-Za-z_][\w.]*_[A-Za-z0-9_]+\b|\b(?:ERR|HTTP)[-_ ]?\d{3,5}\b)",
    re.I,
)
_COMPARE = re.compile(r"\b(?:versus|vs\.?|compare|difference)\b|比较|区别|差异|分别", re.I)
_CONNECT = re.compile(r"\b(?:and|then|also|while)\b|以及|并且|同时|分别|如何.*(?:失败|降级)", re.I)
_QUESTION_PARTS = re.compile(r"[?？;；]|(?:，|,)(?=.{3,})")


class QueryAnalyzer:
    def __init__(self, use_llm: bool = False, llm: Optional[BaseLLM] = None):
        self._use_llm = use_llm
        self._llm = llm
        self.llm_calls = 0

    def analyze(self, query: str) -> QueryAnalysis:
        fallback = self._deterministic(query)
        if not self._use_llm or self._llm is None:
            return fallback
        try:
            self.llm_calls += 1
            prompt = (
                "Return JSON only with intent, should_decompose, should_rewrite, "
                "expand_candidates, complexity_score, reasoning_summary. Intent is one of "
                "factual, lexical, semantic, multi_hop, comparative, unknown. Query: " + query
            )
            response = self._llm.chat([ChatMessage("user", prompt)])
            data = json.loads(response.content)
            return QueryAnalysis(
                original_query=query,
                normalized_query=" ".join(query.split()),
                intent=QueryIntent(data.get("intent", fallback.intent.value)),
                has_identifier=fallback.has_identifier,
                should_decompose=bool(data.get("should_decompose", fallback.should_decompose)),
                should_rewrite=bool(data.get("should_rewrite", fallback.should_rewrite)),
                expand_candidates=bool(data.get("expand_candidates", fallback.expand_candidates)),
                complexity_score=max(0.0, min(1.0, float(data.get("complexity_score", fallback.complexity_score)))),
                reasoning_summary=str(data.get("reasoning_summary", "LLM-assisted structured analysis."))[:240],
            )
        except Exception:
            fallback.degraded = True
            fallback.reasoning_summary += " LLM analysis failed; deterministic fallback used."
            return fallback

    def _deterministic(self, query: str) -> QueryAnalysis:
        normalized = " ".join(query.strip().split())
        identifier = bool(_IDENTIFIER.search(normalized))
        comparative = bool(_COMPARE.search(normalized))
        connected = bool(_CONNECT.search(normalized))
        clauses = len([p for p in _QUESTION_PARTS.split(normalized) if p.strip()])
        long_query = len(normalized) >= 65 or len(normalized.split()) >= 14
        decompose = comparative or (connected and (long_query or clauses > 1))
        if comparative:
            intent = QueryIntent.COMPARATIVE
        elif decompose:
            intent = QueryIntent.MULTI_HOP
        elif identifier and not re.search(r"^(?:what is|什么是|define)\b", normalized, re.I):
            intent = QueryIntent.LEXICAL
        elif normalized.endswith(("?", "？")) or re.search(r"什么|what|how|为什么|why", normalized, re.I):
            intent = QueryIntent.FACTUAL
        elif normalized:
            intent = QueryIntent.SEMANTIC
        else:
            intent = QueryIntent.UNKNOWN
        complexity = min(1.0, 0.15 + 0.3 * comparative + 0.3 * connected + 0.2 * long_query + 0.1 * (clauses > 1))
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            has_identifier=identifier,
            should_decompose=decompose,
            should_rewrite=bool(len(normalized) > 20 or decompose),
            expand_candidates=bool(decompose or long_query),
            complexity_score=complexity,
            reasoning_summary=(
                "Identifier requires preserving lexical signal." if identifier else
                "Query has multiple or comparative information needs." if decompose else
                "Single information need; direct retrieval is sufficient."
            ),
        )

    @staticmethod
    def should_use_agentic(analysis: QueryAnalysis) -> bool:
        return analysis.should_decompose or analysis.complexity_score >= 0.55
