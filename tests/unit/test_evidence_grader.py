from src.core.agentic.evidence_grader import EvidenceGrader
from src.core.agentic.query_analyzer import QueryAnalyzer
from src.core.agentic.retrieval_planner import RetrievalPlanner
from src.core.types import RetrievalResult


def _plan(query="What is BM25?"):
    return RetrievalPlanner().plan(QueryAnalyzer().analyze(query))


def test_empty_results_are_insufficient():
    grade = EvidenceGrader().grade("What is BM25?", [], _plan())
    assert not grade.sufficient
    assert "NO_RESULTS" in grade.reason_codes


def test_good_evidence_is_sufficient():
    results = [
        RetrievalResult("1", .9, "BM25 is a lexical ranking algorithm", {"source_path": "a.pdf"}),
        RetrievalResult("2", .8, "BM25 ranks documents using term frequency", {"source_path": "b.pdf"}),
    ]
    grade = EvidenceGrader(min_confidence=.4).grade("What is BM25?", results, _plan())
    assert grade.sufficient


def test_partial_subquery_coverage_is_insufficient():
    query = "BM25 以及 Cross Encoder 分别做什么"
    plan = RetrievalPlanner().plan(QueryAnalyzer().analyze(query))
    results = [RetrievalResult("1", .8, "BM25 是词法检索算法", {"source_path": "a"}), RetrievalResult("2", .7, "BM25 使用词频", {"source_path": "a"})]
    grade = EvidenceGrader().grade(query, results, plan)
    assert not grade.sufficient

