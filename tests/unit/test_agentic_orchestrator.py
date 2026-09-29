from unittest.mock import MagicMock

from src.core.agentic.orchestrator import AgenticRAGOrchestrator
from src.core.agentic.types import EvidenceAssessment, TerminationReason
from src.core.settings import AgenticConfig
from src.core.types import RetrievalResult


def _settings(max_iterations=2):
    settings = MagicMock()
    settings.agentic = AgenticConfig(max_iterations=max_iterations)
    settings.retrieval.top_k_final = 5
    return settings


def _result(cid, text="BM25 ranking"):
    return RetrievalResult(cid, .8, text, {"source_path": f"{cid}.pdf"})


def test_first_attempt_sufficient_and_final_rerank():
    search = MagicMock()
    search.search.return_value = [_result("a"), _result("b")]
    grader = MagicMock()
    grader.grade.return_value = EvidenceAssessment(True, .9, 1, ["ENOUGH_EVIDENCE"])
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, c, top_k, trace: list(reversed(c))
    result = AgenticRAGOrchestrator(_settings(), search, reranker=reranker, grader=grader).search("What is BM25?")
    assert result.iteration_count == 1
    assert result.termination_reason == TerminationReason.SUFFICIENT_EVIDENCE
    reranker.rerank.assert_called_once()


def test_insufficient_rewrites_then_retries_with_bound():
    search = MagicMock()
    search.search.side_effect = [[], [_result("a"), _result("b")]]
    grader = MagicMock()
    grader.grade.side_effect = [EvidenceAssessment(False, 0, 0, ["NO_RESULTS"]), EvidenceAssessment(True, .8, 1, ["ENOUGH_EVIDENCE"])]
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "BM25 definition ranking"
    result = AgenticRAGOrchestrator(_settings(), search, grader=grader, rewriter=rewriter).search("please BM25")
    assert result.iteration_count == 2
    assert result.rewrite_count == 1
    assert result.retrieval_calls == 2


def test_multi_query_deduplicates_chunks():
    search = MagicMock()
    search.search.side_effect = [[_result("same"), _result("a")], [_result("same"), _result("b")]]
    grader = MagicMock()
    grader.grade.return_value = EvidenceAssessment(True, .9, 1, ["ENOUGH_EVIDENCE"])
    result = AgenticRAGOrchestrator(_settings(), search, grader=grader).search("比较 BM25 和 Dense Retrieval，并说明区别")
    ids = [r.chunk_id for r in result.results]
    assert len(ids) == len(set(ids))


def test_no_evidence_stops_at_budget_without_fabricating_results():
    search = MagicMock()
    search.search.return_value = []
    grader = MagicMock()
    grader.grade.return_value = EvidenceAssessment(
        False, 0.0, 0.0, ["NO_RESULTS"]
    )
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "unknown topic explicit lookup"

    result = AgenticRAGOrchestrator(
        _settings(max_iterations=2),
        search,
        grader=grader,
        rewriter=rewriter,
    ).search("unknown topic")

    assert result.results == []
    assert result.exhausted
    assert result.iteration_count == 2
    assert result.termination_reason == TerminationReason.NO_RESULTS
