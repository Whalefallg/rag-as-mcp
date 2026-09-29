from unittest.mock import MagicMock

from src.core.agentic.orchestrator import AgenticRAGOrchestrator
from src.core.agentic.types import EvidenceAssessment, TerminationReason
from src.core.settings import AgenticConfig
from src.core.types import RetrievalResult


def _settings():
    settings = MagicMock()
    settings.agentic = AgenticConfig(max_iterations=2)
    settings.retrieval.top_k_final = 5
    return settings


def test_planner_failure_degrades_to_direct_plan():
    search = MagicMock()
    search.search.return_value = [RetrievalResult("a", .8, "BM25 definition", {"source_path": "a"}), RetrievalResult("b", .7, "BM25 ranking", {"source_path": "b"})]
    planner = MagicMock()
    planner.plan.side_effect = RuntimeError("planner down")
    grader = MagicMock()
    grader.grade.return_value = EvidenceAssessment(True, .8, 1, ["ENOUGH_EVIDENCE"])
    result = AgenticRAGOrchestrator(_settings(), search, planner=planner, grader=grader).search("What is BM25?")
    assert result.degraded


def test_rewriter_failure_returns_best_available_evidence():
    search = MagicMock()
    search.search.return_value = [RetrievalResult("a", .2, "partial", {})]
    grader = MagicMock()
    grader.grade.return_value = EvidenceAssessment(False, .1, .1, ["LOW_RELEVANCE"])
    rewriter = MagicMock()
    rewriter.rewrite.side_effect = RuntimeError("rewrite down")
    result = AgenticRAGOrchestrator(_settings(), search, grader=grader, rewriter=rewriter).search("unknown topic")
    assert result.termination_reason == TerminationReason.ERROR_FALLBACK
    assert result.results
    assert result.degraded
