"""
MCP Server 集成测试 (tests/integration/test_mcp_server.py)
==========================================================
为什么需要这个文件：
  MCP Server 的核心行为是"接受 JSON-RPC 消息 → 路由 → 返回合规响应"。
  这里以 ProtocolHandler 为单元，测试完整的协议交互流程，
  不依赖真实网络或 MCP SDK，纯 Python dict 驱动。
  覆盖：initialize、tools/list、tools/call（含 query_knowledge_hub 的 mock 路径）、
  错误处理（未知方法、无效参数、工具执行失败）。
"""
import pytest
from unittest.mock import MagicMock, patch

from src.core.settings import Settings, LLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.mcp_server.protocol_handler import ProtocolHandler


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_settings(persist_path: str = "/tmp/test_chroma") -> Settings:
    from src.core.settings import SplitterConfig
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="k"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path=persist_path),
        splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


@pytest.fixture
def handler():
    """创建一个 ProtocolHandler，工具模块通过 mock 注入"""
    settings = _make_settings()

    # mock 三个 tool 模块，避免导入真实依赖
    with patch.dict("sys.modules", {
        "chromadb": MagicMock(),
    }):
        # 直接手动注册 mock tools，绕过真实 tool 模块导入
        h = ProtocolHandler.__new__(ProtocolHandler)
        h._settings = settings
        h._tools = {}
        h._initialized = False

        # 注册测试用 stub tools
        h.register_tool(
            name="query_knowledge_hub",
            description="主检索 tool",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=_stub_query_handler,
        )
        h.register_tool(
            name="list_collections",
            description="列出集合",
            input_schema={"type": "object", "properties": {}},
            handler=_stub_list_handler,
        )
        h.register_tool(
            name="get_document_summary",
            description="获取文档摘要",
            input_schema={"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
            handler=_stub_summary_handler,
        )
        yield h


def _stub_query_handler(arguments, settings):
    return [{"type": "text", "text": f"检索结果：{arguments.get('query')}"}]


def _stub_list_handler(arguments, settings):
    return [{"type": "text", "text": "集合：default"}]


def _stub_summary_handler(arguments, settings):
    doc_id = arguments.get("doc_id", "")
    if not doc_id:
        return [{"type": "text", "text": "错误：doc_id 不能为空。"}]
    return [{"type": "text", "text": f"文档摘要：{doc_id}"}]


# ─── initialize ──────────────────────────────────────────────────────────────

class TestInitialize:
    def test_returns_protocol_version(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert "protocolVersion" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "rag-as-mcp"

    def test_capabilities_include_tools(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 2,
            "method": "initialize",
            "params": {},
        })
        caps = resp["result"]["capabilities"]
        assert "tools" in caps

    def test_marks_initialized(self, handler):
        assert not handler._initialized
        handler.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert handler._initialized


# ─── tools/list ──────────────────────────────────────────────────────────────

class TestToolsList:
    def test_returns_all_registered_tools(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "query_knowledge_hub" in names
        assert "list_collections" in names
        assert "get_document_summary" in names

    def test_each_tool_has_required_fields(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}})
        for tool in resp["result"]["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


# ─── tools/call ──────────────────────────────────────────────────────────────

class TestToolsCall:
    def test_query_knowledge_hub_basic(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 5,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {"query": "如何配置 Azure？"},
            },
        })
        result = resp["result"]
        assert result["isError"] is False
        content = result["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "如何配置 Azure？" in content[0]["text"]

    def test_list_collections(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 6,
            "method": "tools/call",
            "params": {"name": "list_collections", "arguments": {}},
        })
        assert resp["result"]["isError"] is False
        assert resp["result"]["content"][0]["type"] == "text"

    def test_get_document_summary(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 7,
            "method": "tools/call",
            "params": {
                "name": "get_document_summary",
                "arguments": {"doc_id": "data/documents/test.pdf"},
            },
        })
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        assert "data/documents/test.pdf" in text

    def test_unknown_tool_returns_method_not_found(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 8,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_missing_name_returns_invalid_params(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 9,
            "method": "tools/call",
            "params": {"arguments": {}},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_tool_exception_returns_is_error(self, handler):
        """工具执行期间抛异常时，返回 isError=True 而非 JSON-RPC 错误"""
        def _failing_handler(arguments, settings):
            raise RuntimeError("boom")

        handler.register_tool("failing_tool", "...", {}, _failing_handler)
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 10,
            "method": "tools/call",
            "params": {"name": "failing_tool", "arguments": {}},
        })
        result = resp["result"]
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]


# ─── 错误处理 ─────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_unknown_method(self, handler):
        resp = handler.handle({
            "jsonrpc": "2.0", "id": 11,
            "method": "unknown/method",
            "params": {},
        })
        assert resp["error"]["code"] == -32601

    def test_notification_returns_none(self, handler):
        """通知消息（无 id，method 以 notifications/ 开头）无需响应"""
        resp = handler.handle({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })
        assert resp is None

    def test_ping_returns_empty_result(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "id": 12, "method": "ping", "params": {}})
        assert resp["result"] == {}

    def test_non_dict_request(self, handler):
        resp = handler.handle("not a dict")
        assert resp["error"]["code"] == -32600

    def test_response_has_jsonrpc_field(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}})
        assert resp["jsonrpc"] == "2.0"
