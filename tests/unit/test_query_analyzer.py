from src.core.agentic.query_analyzer import QueryAnalyzer
from src.core.agentic.types import QueryIntent


class FailingLLM:
    def chat(self, messages):
        raise RuntimeError("offline")


def test_simple_query_classification():
    analysis = QueryAnalyzer().analyze("What is BM25?")
    assert analysis.intent == QueryIntent.FACTUAL
    assert not analysis.should_decompose


def test_comparison_is_complex():
    analysis = QueryAnalyzer().analyze("比较 BM25 和 Dense Retrieval，并说明它们分别在哪个阶段工作？")
    assert analysis.intent == QueryIntent.COMPARATIVE
    assert analysis.should_decompose
    assert QueryAnalyzer.should_use_agentic(analysis)


def test_identifier_preserves_lexical_signal():
    analysis = QueryAnalyzer().analyze("query_knowledge_hub 返回 HTTP-503 怎么办？")
    assert analysis.has_identifier
    assert analysis.intent == QueryIntent.LEXICAL


def test_llm_failure_uses_deterministic_fallback():
    analysis = QueryAnalyzer(use_llm=True, llm=FailingLLM()).analyze("What is RRF?")
    assert analysis.degraded
    assert analysis.intent == QueryIntent.FACTUAL

