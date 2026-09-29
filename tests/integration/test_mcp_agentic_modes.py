from unittest.mock import MagicMock, patch

import pytest

from src.core.agentic.types import AgenticRetrievalResult, QueryAnalysis, QueryIntent, RetrievalPlan, RetrievalStrategy, TerminationReason
from src.core.settings import AgenticConfig
from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import execute


def _settings():
    settings = MagicMock()
    settings.agentic = AgenticConfig()
    settings.rerank.backend = "none"
    settings.rerank.top_m = 30
    return settings


def _components():
    result = RetrievalResult("c1", .8, "BM25 evidence", {"source_path": "guide.pdf"})
    hybrid = MagicMock()
    hybrid.search.return_value = [result]
    reranker = MagicMock()
    reranker.rerank.return_value = [result]
    analysis = QueryAnalysis("q", "q", QueryIntent.FACTUAL)
    plan = RetrievalPlan("q", "q", QueryIntent.FACTUAL, RetrievalStrategy.HYBRID, ["q"])
    agentic = MagicMock()
    agentic.search.return_value = AgenticRetrievalResult([result], analysis, plan, [], 1, 1, 0, 1, 0, False, False, TerminationReason.SUFFICIENT_EVIDENCE)
    builder = MagicMock()
    builder.build.return_value = [{"type": "text", "text": "answer"}]
    assembler = MagicMock()
    assembler.assemble.return_value = []
    return {"hybrid_search": hybrid, "reranker": reranker, "agentic": agentic, "builder": builder, "assembler": assembler}


@pytest.mark.parametrize("mode", ["classic", "agentic", "auto"])
def test_query_tool_accepts_all_retrieval_modes(mode):
    components = _components()
    query = "比较 BM25 和 Dense Retrieval 的区别" if mode == "auto" else "What is BM25?"
    with patch("src.mcp_server.tools.query_knowledge_hub._get_components", return_value=components):
        content = execute({"query": query, "mode": mode}, _settings())
    assert content[0]["text"] == "answer"
    if mode == "classic":
        components["hybrid_search"].search.assert_called()
    else:
        components["agentic"].search.assert_called()
