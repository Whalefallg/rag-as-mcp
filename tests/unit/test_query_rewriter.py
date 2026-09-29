from src.core.agentic.query_analyzer import QueryAnalyzer
from src.core.agentic.query_rewriter import QueryRewriter
from src.core.agentic.types import EvidenceAssessment


def test_rewriter_removes_noise_and_preserves_identifier():
    query = "请问 query_knowledge_hub HTTP-503 怎么处理"
    rewritten = QueryRewriter().rewrite(query, QueryAnalyzer().analyze(query), EvidenceAssessment(False, .1, .1, ["LOW_RELEVANCE"]))
    assert not rewritten.startswith("请问")
    assert "query_knowledge_hub" in rewritten
    assert "HTTP-503" in rewritten

