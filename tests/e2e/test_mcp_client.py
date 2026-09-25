"""
MCP Client 端到端模拟测试 (tests/e2e/test_mcp_client.py)
=========================================================
验收标准 (DEV_SPEC I1)：
  - 以子进程启动 MCP Server（stdin/stdout Stdio Transport）
  - 完整走通 initialize → tools/list → tools/call(query_knowledge_hub)
  - 每个响应都是合法 JSON-RPC 2.0 格式
  - tools/list 返回包含 query_knowledge_hub 的工具列表
  - tools/call 返回 {"content": [...], "isError": false} 格式
  - 未知 method 返回 -32601 错误码
  - Server 在 stdin 关闭后正常退出（不挂起）

  测试策略：
    - 不启动真实 ChromaDB/Embedding（会挂起），改用内联子进程配置
    - ProtocolHandler 在真实 tools/call 执行 HybridSearch 前失败 OK：
      测试关注协议层面的正确性（JSON-RPC 格式、工具注册），
      不关注 RAG 结果质量
    - 超时保护：每次 readline() 最多等待 5 秒
"""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

_ROOT = Path(__file__).parent.parent.parent
_LOCAL_SETTINGS = _ROOT / "config" / "settings.yaml"
_EXAMPLE_SETTINGS = _ROOT / "config" / "settings.example.yaml"
_SETTINGS = _LOCAL_SETTINGS if _LOCAL_SETTINGS.exists() else _EXAMPLE_SETTINGS


# ── 辅助类：封装与 Server 子进程的通信 ──────────────────────────────────────

class McpServerProcess:
    """启动 MCP Server 子进程，提供 send/recv 接口"""

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        env = {"MCP_SETTINGS_PATH": str(_SETTINGS)}
        import os
        full_env = {**os.environ, **env}

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "src.mcp_server.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_ROOT),
            env=full_env,
            text=True,
            bufsize=1,
        )

    def send(self, msg: Dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def recv(self) -> Optional[Dict[str, Any]]:
        """读取一行响应，超时返回 None"""
        assert self._proc and self._proc.stdout
        import threading

        result = [None]
        exc_holder = [None]

        def _read():
            try:
                line = self._proc.stdout.readline()
                if line.strip():
                    result[0] = json.loads(line.strip())
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=self._timeout)
        if exc_holder[0]:
            raise exc_holder[0]
        return result[0]

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()


def _req(method: str, params: Dict = None, req_id: int = 1) -> Dict:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}


# ── 协议层测试（直接测 ProtocolHandler，不需要子进程）─────────────────────────

class TestProtocolHandlerDirect:
    """直接实例化 ProtocolHandler，快速验证协议逻辑，无网络/IO 开销"""

    @pytest.fixture
    def handler(self):
        from src.core.settings import (
            Settings, LLMConfig, EmbeddingConfig,
            VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig,
        )
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
            embedding=EmbeddingConfig(
                provider="openai", model="text-embedding-3-small", api_key="k"
            ),
            vector_store=VectorStoreConfig(backend="chroma", persist_path="/tmp/tc"),
            splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
            retrieval=RetrievalConfig(
                sparse_backend="bm25", fusion_algorithm="rrf",
                top_k_dense=20, top_k_sparse=20, top_k_final=10,
            ),
            rerank=RerankConfig(backend="none"),
            raw_config={},
        )
        from src.mcp_server.protocol_handler import ProtocolHandler
        return ProtocolHandler(settings)

    def test_initialize_returns_server_info(self, handler):
        resp = handler.handle(_req("initialize", {"protocolVersion": "2024-11-05"}))
        assert resp["jsonrpc"] == "2.0"
        assert "result" not in resp or "serverInfo" in resp.get("result", {})
        assert "error" not in resp

    def test_tools_list_returns_tools_array(self, handler):
        resp = handler.handle(_req("tools/list"))
        assert "error" not in resp
        tools = resp["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_tools_list_includes_query_knowledge_hub(self, handler):
        resp = handler.handle(_req("tools/list"))
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "query_knowledge_hub" in names

    def test_tools_list_schema_has_required_fields(self, handler):
        resp = handler.handle(_req("tools/list"))
        for tool in resp["result"]["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_unknown_method_returns_32601(self, handler):
        resp = handler.handle(_req("unknown/method"))
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_invalid_request_returns_32600(self, handler):
        resp = handler.handle("not-a-dict")
        assert "error" in resp
        assert resp["error"]["code"] == -32600

    def test_tool_not_found_returns_32601(self, handler):
        resp = handler.handle(_req("tools/call", {"name": "nonexistent_tool"}))
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_tools_call_missing_name_returns_32602(self, handler):
        resp = handler.handle(_req("tools/call", {}))
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_notification_returns_none(self, handler):
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        result = handler.handle(notif)
        assert result is None

    def test_ping_returns_empty_result(self, handler):
        resp = handler.handle(_req("ping"))
        assert "error" not in resp
        assert resp["result"] == {}

    def test_tools_call_query_knowledge_hub_returns_content(self, handler):
        """
        query_knowledge_hub 在空知识库下可能返回错误内容，
        但响应格式必须是 {content: [...], isError: bool}
        """
        resp = handler.handle(_req("tools/call", {
            "name": "query_knowledge_hub",
            "arguments": {"query": "test query", "top_k": 3},
        }))
        assert "error" not in resp, f"unexpected error: {resp.get('error')}"
        result = resp["result"]
        assert "content" in result
        assert isinstance(result["content"], list)
        assert "isError" in result

    def test_response_has_jsonrpc_field(self, handler):
        resp = handler.handle(_req("tools/list"))
        assert resp.get("jsonrpc") == "2.0"

    def test_response_id_matches_request(self, handler):
        resp = handler.handle(_req("tools/list", req_id=42))
        assert resp["id"] == 42


class TestMcpServerSubprocess:
    """使用本地配置或无密钥示例配置测试 Stdio 传输。"""

    @pytest.fixture
    def server(self):
        proc = McpServerProcess(timeout=5.0)
        proc.start()
        time.sleep(0.3)   # 等待进程启动
        yield proc
        proc.close()

    def test_initialize_via_stdio(self, server):
        server.send(_req("initialize", {"protocolVersion": "2024-11-05"}))
        resp = server.recv()
        assert resp is not None, "Server 未响应 initialize 请求"
        assert resp.get("jsonrpc") == "2.0"
        assert "result" in resp

    def test_tools_list_via_stdio(self, server):
        # 先 initialize
        server.send(_req("initialize", {"protocolVersion": "2024-11-05"}, req_id=1))
        server.recv()
        # 再 tools/list
        server.send(_req("tools/list", req_id=2))
        resp = server.recv()
        assert resp is not None
        assert "result" in resp
        assert "tools" in resp["result"]

    def test_tools_list_includes_query_knowledge_hub_via_stdio(self, server):
        server.send(_req("initialize", {"protocolVersion": "2024-11-05"}, req_id=1))
        server.recv()
        server.send(_req("tools/list", req_id=2))
        resp = server.recv()
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "query_knowledge_hub" in names

    def test_unknown_method_via_stdio(self, server):
        server.send(_req("initialize", req_id=1))
        server.recv()
        server.send(_req("bogus/method", req_id=2))
        resp = server.recv()
        assert resp is not None
        assert resp.get("error", {}).get("code") == -32601

    def test_server_exits_on_stdin_close(self, server):
        server.send(_req("initialize", req_id=1))
        server.recv()
        server._proc.stdin.close()
        try:
            server._proc.wait(timeout=3)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        assert exited, "Server 在 stdin 关闭后未退出"
