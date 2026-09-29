from unittest.mock import MagicMock

from src.core.agentic.orchestrator import AgenticRAGOrchestrator
from src.core.agentic.types import TerminationReason
from src.core.settings import AgenticConfig
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


def test_deterministic_agentic_pipeline_records_decisions():
    settings = MagicMock()
    settings.agentic = AgenticConfig(max_iterations=2)
    settings.retrieval.top_k_final = 5
    hybrid = MagicMock()
    evidence = [
        RetrievalResult("rrf", .8, "RRF performs rank fusion in retrieval", {"source_path": "fusion.pdf"}),
        RetrievalResult("ce", .7, "Cross Encoder reranks the final candidates", {"source_path": "rerank.pdf"}),
    ]
    hybrid.search.return_value = evidence
    trace = TraceContext("query")
    result = AgenticRAGOrchestrator(settings, hybrid).search(
        "RRF 和 Cross Encoder 分别在哪个阶段工作，它们有什么区别？",
        trace=trace,
    )
    assert result.results
    assert result.termination_reason in {TerminationReason.SUFFICIENT_EVIDENCE, TerminationReason.MAX_ITERATIONS}
    stages = [stage["stage"] for stage in trace.to_dict()["stages"]]
    assert "query_analysis" in stages
    assert "retrieval_plan" in stages
    assert "retrieval_attempt_1" in stages
    assert "evidence_grade_1" in stages
    assert trace.get_metadata("termination_reason")
