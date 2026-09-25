"""
Dashboard 启动脚本 (scripts/start_dashboard.py)
================================================
运行方式：
  python scripts/start_dashboard.py           # 默认 port 8501
  python scripts/start_dashboard.py --port 8502
  MCP_SETTINGS_PATH=config/prod.yaml python scripts/start_dashboard.py

等效命令：
  streamlit run src/observability/dashboard/app.py
"""
import subprocess
import sys
import os
from pathlib import Path


def main() -> None:
    port = "8501"
    args = sys.argv[1:]
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = args[idx + 1]

    app_path = Path(__file__).parent.parent / "src" / "observability" / "dashboard" / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", port,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    print(f"启动 Dashboard：http://localhost:{port}")
    print(f"应用路径：{app_path}")
    print("按 Ctrl+C 停止\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard 已停止。")
    except FileNotFoundError:
        print("错误：streamlit 未安装。请运行：pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
