from src.core.agentic.query_analyzer import QueryAnalyzer
from src.core.agentic.retrieval_planner import RetrievalPlanner
from src.core.agentic.types import RetrievalStrategy


def test_simple_query_uses_hybrid():
    plan = RetrievalPlanner().plan(QueryAnalyzer().analyze("What is BM25?"))
    assert plan.strategy == RetrievalStrategy.HYBRID
    assert plan.subqueries == ["What is BM25?"]


def test_complex_query_is_decomposed_and_bounded():
    analysis = QueryAnalyzer().analyze("比较 BM25 和 Dense Retrieval，并说明 RRF 以及 Cross Encoder 的区别")
    plan = RetrievalPlanner(max_subqueries=2).plan(analysis)
    assert plan.strategy in {RetrievalStrategy.DECOMPOSED, RetrievalStrategy.MULTI_QUERY}
    assert 1 <= len(plan.subqueries) <= 2

