"""
MCP Protocol Handler (src/mcp_server/protocol_handler.py)
==========================================================
为什么需要这个文件：
  server.py 只负责 Stdio I/O 的"搬运"，ProtocolHandler 负责真正的"理解"：
    - 解析 JSON-RPC 2.0 方法名（initialize/tools/list/tools/call）
    - 管理 tool 注册表（工厂模式，tool 自我注册）
    - 把 tool 执行结果包装成合规的 MCP 响应格式
    - 将任何异常转换成标准 JSON-RPC 错误码，绝不向 Client 泄露裸堆栈

  JSON-RPC 2.0 错误码语义：
    -32700  Parse error       JSON 解析失败（由 server.py 处理）
    -32600  Invalid Request   不符合 JSON-RPC 规范的请求
    -32601  Method not found  未知方法名
    -32602  Invalid params    参数类型/缺少必填项
    -32603  Internal error    工具执行期间的未预期异常

  MCP 协议核心方法：
    initialize      → 能力协商，返回 serverInfo + capabilities
    tools/list      → 返回所有已注册 tool 的 schema
    tools/call      → 路由到具体 tool 执行，返回 content 数组
    notifications/* → 单向通知（无需响应，返回 None）
"""
import traceback
from typing import Any, Callable, Dict, List, Optional

from src.core.settings import Settings
from src.observability.logger import get_logger

logger = get_logger(__name__)

SERVER_NAME = "rag-as-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


class ToolDefinition:
    """一个已注册 MCP Tool 的完整描述"""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def to_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class ProtocolHandler:
    """
    MCP JSON-RPC 2.0 协议处理器。

    使用方法：
        handler = ProtocolHandler(settings)
        response = handler.handle(request_dict)   # None 表示通知，无需回复
        if response:
            stdout.write(json.dumps(response))
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tools: Dict[str, ToolDefinition] = {}
        self._initialized = False
        self._register_builtin_tools()

    # ──────────────────────────────────────────────────────────────────────
    # 工具注册
    # ──────────────────────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """注册一个 MCP tool。handler 的签名：(arguments: dict, settings: Settings) -> list"""
        self._tools[name] = ToolDefinition(name, description, input_schema, handler)
        logger.debug(f"Tool registered: {name}")

    def _register_builtin_tools(self) -> None:
        """延迟导入各 tool 模块并完成注册，避免循环导入"""
        from src.mcp_server.tools.query_knowledge_hub import (
            TOOL_NAME as qkh_name,
            TOOL_DESCRIPTION as qkh_desc,
            TOOL_INPUT_SCHEMA as qkh_schema,
            execute as qkh_execute,
        )
        from src.mcp_server.tools.list_collections import (
            TOOL_NAME as lc_name,
            TOOL_DESCRIPTION as lc_desc,
            TOOL_INPUT_SCHEMA as lc_schema,
            execute as lc_execute,
        )
        from src.mcp_server.tools.get_document_summary import (
            TOOL_NAME as gds_name,
            TOOL_DESCRIPTION as gds_desc,
            TOOL_INPUT_SCHEMA as gds_schema,
            execute as gds_execute,
        )

        self.register_tool(qkh_name, qkh_desc, qkh_schema, qkh_execute)
        self.register_tool(lc_name, lc_desc, lc_schema, lc_execute)
        self.register_tool(gds_name, gds_desc, gds_schema, gds_execute)

    # ──────────────────────────────────────────────────────────────────────
    # 主分发入口
    # ──────────────────────────────────────────────────────────────────────

    def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        分发 JSON-RPC 请求并返回响应 dict。
        通知（无 id 字段）不需要响应，返回 None。
        """
        if not isinstance(request, dict):
            return self._error_response(None, -32600, "Invalid Request: not a JSON object")

        req_id = request.get("id")          # 通知没有 id
        method = request.get("method", "")
        params = request.get("params") or {}

        # 通知类消息：无需回复
        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            result = self._dispatch(method, params)
        except _McpError as e:
            return self._error_response(req_id, e.code, e.message)
        except Exception as e:
            logger.error(f"Unexpected error in tool [{method}]: {e}", exc_info=True)
            return self._error_response(req_id, -32603, f"Internal error: {type(e).__name__}")

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        """将方法名路由到具体处理函数"""
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "tools/list":
            return self._handle_tools_list(params)
        if method == "tools/call":
            return self._handle_tools_call(params)
        if method in ("ping", "initialized"):
            return {}
        raise _McpError(-32601, f"Method not found: {method}")

    # ──────────────────────────────────────────────────────────────────────
    # 协议方法实现
    # ──────────────────────────────────────────────────────────────────────

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        能力协商：Client 告知自己的协议版本，Server 返回 serverInfo + capabilities。
        MCP 要求 Server 只声明自己实际支持的 capabilities。
        """
        client_version = params.get("protocolVersion", "")
        logger.info(f"Client connecting, protocolVersion={client_version!r}")
        self._initialized = True

        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,   # 运行期间 tool 列表不变
                },
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """返回所有已注册 tool 的 schema 列表"""
        return {"tools": [t.to_schema() for t in self._tools.values()]}

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用指定 tool。
        参数格式：{"name": "tool_name", "arguments": {...}}
        返回格式：{"content": [...], "isError": false}
        """
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if not name:
            raise _McpError(-32602, "Invalid params: 'name' is required")

        tool = self._tools.get(name)
        if tool is None:
            raise _McpError(-32601, f"Tool not found: {name!r}")

        logger.info(f"Calling tool: {name}")
        try:
            content = tool.handler(arguments, self._settings)
        except _McpError:
            raise
        except Exception as e:
            logger.error(f"Tool [{name}] error: {e}", exc_info=True)
            return {
                "content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}],
                "isError": True,
            }

        return {"content": content, "isError": False}

    # ──────────────────────────────────────────────────────────────────────
    # 工具函数
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _error_response(
        req_id: Any, code: int, message: str
    ) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


class _McpError(Exception):
    """内部错误，携带 JSON-RPC 错误码"""
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
