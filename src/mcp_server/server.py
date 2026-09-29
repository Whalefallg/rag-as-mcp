"""MCP Server 入口 (src/mcp_server/server.py)"""
import sys
import json
import os
from pathlib import Path

# 确保项目根目录在 sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.settings import load_settings
from src.observability.logger import get_logger
from src.mcp_server.protocol_handler import ProtocolHandler

logger = get_logger(__name__)


def _write_response(response: dict) -> None:
    """
    将 JSON-RPC 响应写入 stdout，末尾附换行符。
    stdout 只能有合法 JSON Lines，严禁混入其他输出。
    """
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_stdio_server(settings_path: str = "config/settings.yaml") -> None:
    """
    启动 MCP Stdio Server 主循环。

    工作流程：
      1. 加载配置，初始化 ProtocolHandler（含 HybridSearch、Reranker 等）
      2. 逐行读取 stdin（每行一条 JSON-RPC 消息）
      3. 交给 ProtocolHandler 处理，将响应写回 stdout
      4. stdin 关闭时正常退出
    """
    logger.info("MCP Server starting (stdio transport)")

    try:
        settings = load_settings(settings_path)
        logger.info(f"Config loaded: llm={settings.llm.provider}, "
                    f"embedding={settings.embedding.provider}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    handler = ProtocolHandler(settings)
    logger.info("ProtocolHandler ready, entering read loop")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # 解析 JSON-RPC 请求
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON received: {e}")
            _write_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            })
            continue

        # 分发给 ProtocolHandler
        response = handler.handle(request)
        if response is not None:
            _write_response(response)


def main() -> None:
    """CLI 入口，供 main.py 和 pyproject.toml scripts 调用"""
    settings_path = os.environ.get("MCP_SETTINGS_PATH", "config/settings.yaml")
    run_stdio_server(settings_path)


if __name__ == "__main__":
    main()
