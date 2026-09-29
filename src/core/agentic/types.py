from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"
    UNKNOWN = "unknown"


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"
    DENSE_FOCUSED = "dense_focused"
    SPARSE_FOCUSED = "sparse_focused"
    MULTI_QUERY = "multi_query"
    DECOMPOSED = "decomposed"


class TerminationReason(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    MAX_ITERATIONS = "max_iterations"
    NO_RESULTS = "no_results"
    CLASSIC_ROUTE = "classic_route"
    ERROR_FALLBACK = "error_fallback"


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    intent: QueryIntent
    has_identifier: bool = False
    should_decompose: bool = False
    should_rewrite: bool = False
    expand_candidates: bool = False
    complexity_score: float = 0.0
    reasoning_summary: str = ""
    degraded: bool = False


@dataclass
class RetrievalPlan:
    original_query: str
    normalized_query: str
    intent: QueryIntent
    strategy: RetrievalStrategy
    subqueries: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    candidate_k: int = 10
    use_rerank: bool = True
    reasoning_summary: str = ""


@dataclass
class EvidenceAssessment:
    sufficient: bool
    confidence: float
    coverage: float
    reason_codes: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    degraded: bool = False


@dataclass
class RetrievalAttempt:
    iteration: int
    query: str
    plan: RetrievalPlan
    results: List[RetrievalResult]
    assessment: EvidenceAssessment
    retrieval_calls: int = 0


@dataclass
class AgentState:
    original_query: str
    iteration: int = 0
    retrieval_calls: int = 0
    rewrite_count: int = 0
    subquery_count: int = 0
    llm_calls: int = 0
    degraded: bool = False
    exhausted: bool = False
    termination_reason: Optional[TerminationReason] = None
    attempts: List[RetrievalAttempt] = field(default_factory=list)


@dataclass
class AgenticRetrievalResult:
    results: List[RetrievalResult]
    analysis: QueryAnalysis
    plan: RetrievalPlan
    attempts: List[RetrievalAttempt]
    iteration_count: int
    retrieval_calls: int
    rewrite_count: int
    subquery_count: int
    llm_calls: int
    degraded: bool
    exhausted: bool
    termination_reason: TerminationReason

