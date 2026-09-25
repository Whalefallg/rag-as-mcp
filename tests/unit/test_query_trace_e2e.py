"""Regression tests for end-to-end query trace propagation."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.core.trace.trace_context import TraceContext


def _settings():
    return SimpleNamespace(
        rerank=SimpleNamespace(backend="cross_encoder", top_m=30)
    )


def test_query_tool_propagates_one_trace_end_to_end():
    from src.mcp_server.tools import query_knowledge_hub as tool
    seen = {}

    class Hybrid:
        def search(self, **kwargs):
            seen["hybrid"] = kwargs["trace"]
            return []

    class Reranker:
        def rerank(self, **kwargs):
            seen["rerank"] = kwargs["trace"]
            return kwargs["candidates"]

    class Assembler:
        def assemble(self, results, max_images=3, trace=None):
            seen["assembler"] = trace
            return []

    class Builder:
        def build(self, results, query, image_contents=None, trace=None):
            seen["builder"] = trace
            return [{"type": "text", "text": "ok"}]

    components = {
        "hybrid_search": Hybrid(),
        "reranker": Reranker(),
        "assembler": Assembler(),
        "builder": Builder(),
    }

    with patch.object(tool, "_get_components", return_value=components), \
         patch.object(tool.global_collector, "collect") as collect:
        result = tool.execute(
            {"query": "hello", "top_k": 5, "collection": "A"},
            _settings(),
        )

    assert result[0]["text"] == "ok"
    trace = seen["hybrid"]
    assert isinstance(trace, TraceContext)
    assert seen["assembler"] is trace
    assert seen["builder"] is trace
    collect.assert_called_once_with(trace)
    assert trace.get_metadata("user_query") == "hello"
    assert trace.get_metadata("collection") == "A"


def test_query_tool_collects_trace_on_component_init_failure():
    from src.mcp_server.tools import query_knowledge_hub as tool

    with patch.object(
        tool, "_get_components", side_effect=RuntimeError("init failed")
    ), patch.object(tool.global_collector, "collect") as collect:
        result = tool.execute({"query": "hello"}, _settings())

    assert "初始化失败" in result[0]["text"]
    collect.assert_called_once()
    trace = collect.call_args.args[0]
    stages = trace.to_dict()["stages"]
    assert any(
        s["stage"] == "component_init" and s["error"] == "init failed"
        for s in stages
    )


def test_query_tool_collects_trace_on_search_failure():
    from src.mcp_server.tools import query_knowledge_hub as tool

    class Hybrid:
        def search(self, **kwargs):
            raise RuntimeError("search failed")

    components = {
        "hybrid_search": Hybrid(),
        "reranker": MagicMock(),
        "assembler": MagicMock(),
        "builder": MagicMock(),
    }

    with patch.object(tool, "_get_components", return_value=components), \
         patch.object(tool.global_collector, "collect") as collect:
        result = tool.execute({"query": "hello"}, _settings())

    assert "检索时发生错误" in result[0]["text"]
    collect.assert_called_once()
    trace = collect.call_args.args[0]
    assert trace.get_metadata("status") == "error"


def test_multimodal_assembler_records_trace_stage():
    from src.core.response.multimodal_assembler import MultimodalAssembler

    trace = TraceContext(trace_type="query")
    assembler = MultimodalAssembler(image_storage=None)
    assert assembler.assemble([], trace=trace) == []
    stage = trace.to_dict()["stages"][-1]
    assert stage["stage"] == "multimodal_assembly"
    assert stage["image_count"] == 0


def test_response_builder_records_trace_stage():
    from src.core.response.response_builder import ResponseBuilder

    trace = TraceContext(trace_type="query")
    content = ResponseBuilder().build([], "hello", trace=trace)
    assert content[0]["type"] == "text"
    stage = trace.to_dict()["stages"][-1]
    assert stage["stage"] == "response_build"
    assert stage["result_count"] == 0


def test_reranker_receives_same_trace_when_results_exist():
    from src.mcp_server.tools import query_knowledge_hub as tool
    from src.core.types import RetrievalResult

    seen = {}

    class Hybrid:
        def search(self, **kwargs):
            seen["hybrid"] = kwargs["trace"]
            return [RetrievalResult("c1", 1.0, "text", {})]

    class Reranker:
        def rerank(self, **kwargs):
            seen["rerank"] = kwargs["trace"]
            return kwargs["candidates"]

    class Assembler:
        def assemble(self, results, max_images=3, trace=None):
            return []

    class Builder:
        def build(self, results, query, image_contents=None, trace=None):
            return [{"type": "text", "text": "ok"}]

    components = {
        "hybrid_search": Hybrid(),
        "reranker": Reranker(),
        "assembler": Assembler(),
        "builder": Builder(),
    }

    with patch.object(tool, "_get_components", return_value=components), \
         patch.object(tool.global_collector, "collect"):
        tool.execute({"query": "hello"}, _settings())

    assert seen["rerank"] is seen["hybrid"]
