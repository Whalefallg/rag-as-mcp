# D5 — MCP Server 协议

> 对应源码：`src/mcp_server/server.py`, `src/mcp_server/protocol_handler.py`, `src/mcp_server/tools/`

---

## D5.1 MCP 协议概述：JSON-RPC 交互模型与标准规范

### 什么是 MCP

Model Context Protocol（模型上下文协议）是 Anthropic 发布的开放标准，让 AI 助手（Copilot、Claude Desktop 等）通过统一接口调用外部工具和数据源。

本项目作为 **MCP Server**，对外暴露知识库查询能力；Copilot/Claude 作为 **MCP Client/Host**，按需调用这些工具。

### 传输协议：Stdio Transport

```
MCP Client（VS Code Copilot / Claude Desktop）
    │  以子进程方式启动 python main.py
    │  通过 stdin 发送 JSON-RPC 消息
    │  通过 stdout 读取响应
    ▼
MCP Server（本项目）
    stdin  ← 每行一条 JSON-RPC 请求
    stdout → 每行一条 JSON-RPC 响应
    stderr → 日志（不能混入 stdout，否则 Client 崩溃）
```

**选 Stdio 的理由：**
- 零配置：无需网络端口、无需鉴权，Client 配置文件里指定启动命令即可
- 隐私安全：数据不经过网络，适合私有知识库
- 本地优先：完美契合开发者本地工作流

### JSON-RPC 2.0 消息格式

**请求：**
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
 "params": {"name": "query_knowledge_hub", "arguments": {"query": "..."}}}
```

**响应：**
```json
{"jsonrpc": "2.0", "id": 1, "result": {"content": [...], "isError": false}}
```

**错误：**
```json
{"jsonrpc": "2.0", "id": 1,
 "error": {"code": -32601, "message": "Method not found: xxx"}}
```

### MCP 生命周期

```
Client 启动子进程 →  initialize（能力协商）→ tools/list（获取可用工具）
    → tools/call（调用具体工具）× N → Client 关闭子进程（stdin 关闭）
```

---

## D5.2 Tool 注册机制：三个工具的定义、参数与执行逻辑

### 三个内置工具

| 工具名 | 功能 | 核心参数 |
|--------|------|---------|
| `query_knowledge_hub` | 主检索入口，混合检索 + Rerank，返回带引用的结果 | `query: string`, `top_k?: int`, `collection?: string` |
| `list_collections` | 列举知识库中可用的文档集合 | 无 |
| `get_document_summary` | 获取指定文档的摘要与元信息 | `doc_id: string` |

### Tool 定义结构（inputSchema 为 JSON Schema 格式）

```python
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = "在知识库中执行混合检索（Hybrid Search），返回最相关的文档片段"
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "查询问题"},
        "top_k": {"type": "integer", "description": "返回结果数量", "default": 10},
        "collection": {"type": "string", "description": "目标知识库集合"}
    },
    "required": ["query"]
}

def execute(arguments: dict, settings: Settings) -> list:
    """返回 MCP content 数组"""
    ...
```

### 工具注册机制（延迟导入避免循环引用）

```python
class ProtocolHandler:
    def _register_builtin_tools(self) -> None:
        from src.mcp_server.tools.query_knowledge_hub import (
            TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA, execute
        )
        self.register_tool(TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA, execute)
```

每个 tool 文件自己声明 `TOOL_NAME / TOOL_DESCRIPTION / TOOL_INPUT_SCHEMA / execute`，ProtocolHandler 只做注册和路由，新增工具只需添加一个文件。

---

## D5.3 ProtocolHandler：请求路由、分发与能力协商

### 核心分发逻辑

```python
class ProtocolHandler:
    def handle(self, request: dict) -> Optional[dict]:
        req_id = request.get("id")
        method = request.get("method", "")

        # 通知（无 id 字段）→ 不需要响应
        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            result = self._dispatch(method, params)
        except _McpError as e:
            return self._error_response(req_id, e.code, e.message)
        except Exception as e:
            return self._error_response(req_id, -32603, f"Internal error: {type(e).__name__}")

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _dispatch(self, method, params):
        if method == "initialize":   return self._handle_initialize(params)
        if method == "tools/list":   return self._handle_tools_list(params)
        if method == "tools/call":   return self._handle_tools_call(params)
        if method in ("ping", "initialized"):  return {}
        raise _McpError(-32601, f"Method not found: {method}")
```

### 能力协商（initialize 响应）

```python
def _handle_initialize(self, params) -> dict:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False}  # 运行期间 tool 列表不变
        },
        "serverInfo": {"name": "rag-as-mcp", "version": "0.1.0"}
    }
```

Client 通过这个响应知道 Server 支持什么能力（tools/resources/prompts），然后才会调用 `tools/list`。

### 错误码规范（JSON-RPC 2.0）

| 代码 | 含义 | 触发时机 |
|------|------|---------|
| -32700 | Parse error | JSON 解析失败（server.py 层处理） |
| -32600 | Invalid Request | 不是 JSON 对象 |
| -32601 | Method not found | 未知方法名 / 未知工具名 |
| -32602 | Invalid params | 缺少必填参数 |
| -32603 | Internal error | 工具执行期间的未预期异常 |

---

## D5.4 Server 生命周期管理与异常处理

### Stdio Server 主循环

```python
def run_stdio_server(settings_path: str) -> None:
    settings = load_settings(settings_path)   # 失败 → sys.exit(1)
    handler = ProtocolHandler(settings)

    for raw_line in sys.stdin:          # stdin 关闭 → 自然退出 for 循环
        request = json.loads(raw_line)  # 解析失败 → 返回 -32700 错误，继续循环
        response = handler.handle(request)
        if response:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
```

### 关键约束

```python
# ✅ 正确：日志走 stderr
logger.info("...")    # observability/logger.py 配置了 stderr 输出

# ❌ 错误：print 走 stdout，会破坏 MCP Client 的 JSON 解析
print("debug info")
```

### 异常隔离层次

```
单条消息解析失败 → 返回 -32700，继续处理下一条消息（不退出）
工具执行异常   → 捕获后包装为 isError=true 响应（不退出）
配置加载失败   → sys.exit(1)（无法继续运行）
stdin 关闭     → for 循环自然结束，正常退出
```

### MCP Client 配置示例

**GitHub Copilot（`.vscode/mcp.json`）：**
```json
{
  "servers": {
    "rag-as-mcp": {
      "type": "stdio",
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

**Claude Desktop（`claude_desktop_config.json`）：**
```json
{
  "mcpServers": {
    "rag-as-mcp": {
      "command": "python",
      "args": ["/absolute/path/to/main.py"]
    }
  }
}
```

---

# D5 — MCP Server Protocol (English)

> Source files: `src/mcp_server/server.py`, `src/mcp_server/protocol_handler.py`, `src/mcp_server/tools/`

---

## D5.1 MCP Protocol Overview: JSON-RPC Interaction Model

**MCP (Model Context Protocol)** is an open standard by Anthropic that lets AI assistants (Copilot, Claude Desktop, etc.) call external tools through a unified interface.

This project runs as an **MCP Server** exposing knowledge-base query capabilities; Copilot/Claude acts as the **MCP Client/Host**.

### Stdio Transport

```
MCP Client  (VS Code Copilot / Claude Desktop)
    │  spawns: python main.py
    │  sends JSON-RPC messages via stdin
    │  reads responses via stdout
    ▼
MCP Server  (this project)
    stdin  ← one JSON-RPC request per line
    stdout → one JSON-RPC response per line
    stderr → logs only (mixing anything into stdout crashes the client)
```

**Why Stdio**: zero config (no ports, no auth), privacy-safe (data never leaves the machine), local-first.

### MCP Lifecycle

```
Client spawns server  →  initialize (capability negotiation)
  →  tools/list (discover available tools)
  →  tools/call × N (invoke tools)
  →  client closes stdin (server exits naturally)
```

---

## D5.2 Tool Registration: Three Built-in Tools

| Tool name | Purpose | Required params |
|-----------|---------|----------------|
| `query_knowledge_hub` | Main retrieval entry: hybrid search + rerank, returns cited results | `query: string` |
| `list_collections` | List available document collections in the knowledge base | none |
| `get_document_summary` | Retrieve title/summary/tags for a specific document | `doc_id: string` |

Each tool file exports four names: `TOOL_NAME`, `TOOL_DESCRIPTION`, `TOOL_INPUT_SCHEMA` (JSON Schema), and `execute(arguments, settings) -> list`. `ProtocolHandler` uses **delayed imports** to register them, avoiding circular imports.

---

## D5.3 ProtocolHandler: Request Routing and Capability Negotiation

```python
def _dispatch(self, method, params):
    if method == "initialize":   return self._handle_initialize(params)
    if method == "tools/list":   return self._handle_tools_list(params)
    if method == "tools/call":   return self._handle_tools_call(params)
    if method in ("ping", "initialized"):  return {}
    raise _McpError(-32601, f"Method not found: {method}")
```

**`initialize` response** declares server capabilities so the client knows what to call next:
```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": {"tools": {"listChanged": false}},
  "serverInfo": {"name": "rag-as-mcp", "version": "0.1.0"}
}
```

### JSON-RPC 2.0 Error Codes

| Code | Meaning | Triggered by |
|------|---------|-------------|
| -32700 | Parse error | Malformed JSON (handled in `server.py`) |
| -32600 | Invalid Request | Not a JSON object |
| -32601 | Method not found | Unknown method name or unknown tool name |
| -32602 | Invalid params | Missing required parameter |
| -32603 | Internal error | Unexpected exception during tool execution |

---

## D5.4 Server Lifecycle and Exception Handling

```python
for raw_line in sys.stdin:          # natural exit when stdin closes
    request = json.loads(raw_line)  # parse error → return -32700, continue loop
    response = handler.handle(request)
    if response:
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
```

### Exception Isolation Levels

| Scope | Behavior |
|-------|---------|
| Single message parse fails | Return -32700, keep serving |
| Tool execution raises | Wrap as `isError: true` response, keep serving |
| Config load fails | `sys.exit(1)` — cannot continue |
| stdin closes | `for` loop ends, clean exit |

### Client Config Examples

**GitHub Copilot** (`.vscode/mcp.json`):
```json
{"servers": {"rag-as-mcp": {"type": "stdio", "command": "python", "args": ["main.py"], "cwd": "/path/to/project"}}}
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{"mcpServers": {"rag-as-mcp": {"command": "python", "args": ["/absolute/path/to/main.py"]}}}
```
