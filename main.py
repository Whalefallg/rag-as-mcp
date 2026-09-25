"""
MCP Server 启动入口 (main.py)
==============================
运行方式：
  python main.py                        # 标准 Stdio MCP Server
  MCP_SETTINGS_PATH=... python main.py  # 指定配置文件路径

MCP Client 配置（GitHub Copilot .vscode/mcp.json）：
  {
    "servers": {
      "rag-as-mcp": {
        "type": "stdio",
        "command": "python",
        "args": ["main.py"],
        "cwd": "<项目根目录>"
      }
    }
  }

MCP Client 配置（Claude Desktop claude_desktop_config.json）：
  {
    "mcpServers": {
      "rag-as-mcp": {
        "command": "python",
        "args": ["/absolute/path/to/main.py"]
      }
    }
  }

注意：stdout 只输出 MCP JSON-RPC 消息，日志全部走 stderr。
"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_server.server import main

if __name__ == "__main__":
    main()
