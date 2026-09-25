"""
结构化日志模块 (src/observability/logger.py)
============================================
为什么需要这个文件：
  MCP Server 的 stdout 只能写 JSON-RPC 消息，所有业务日志必须走 stderr。
  Phase F 升级为双轨输出：
    1. stderr: 人类可读的格式化日志（开发调试用）
    2. logs/traces.jsonl: JSON Lines 格式的 trace 持久化（Dashboard 读取）

  JSON Lines 格式每行一个独立 JSON 对象，方便 tail -f 实时追踪，
  也方便 jq/grep 命令行查询，不依赖任何外部存储。

  write_trace() 是 TraceCollector 的底层写入接口，
  get_trace_logger() 是暴露给 TraceCollector 的专用 logger。
"""
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# 默认 trace 日志文件路径（可通过环境变量覆盖）
_DEFAULT_TRACE_FILE = os.environ.get("MCP_TRACE_FILE", "logs/traces.jsonl")

# 写文件锁，防止多线程并发写入乱序
_file_lock = threading.Lock()


# ─── 人类可读 logger ──────────────────────────────────────────────────────────

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    获取指定名称的 logger，输出到 stderr（人类可读格式）。

    Args:
        name:  logger 名称，通常传 __name__，便于定位来源模块。
        level: 日志级别，默认 INFO。
    Returns:
        配置好的 Logger（同名 logger 只配置一次，避免重复 handler）。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


# ─── JSON Lines trace logger ─────────────────────────────────────────────────

class _JsonLinesHandler(logging.FileHandler):
    """将日志记录序列化为单行 JSON 并写入文件"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            # record.msg 可能直接是 dict（由 write_trace 传入）
            if isinstance(record.msg, dict):
                payload = record.msg
            else:
                payload = {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": msg,
                    "timestamp": datetime.fromtimestamp(
                        record.created, tz=timezone.utc
                    ).isoformat(),
                }
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with _file_lock:
                self.stream.write(line + "\n")
                self.stream.flush()
        except Exception:
            self.handleError(record)


def get_trace_logger(trace_file: Optional[str] = None) -> logging.Logger:
    """
    获取写入 JSON Lines 文件的专用 logger。
    重复调用返回同一实例（不重复添加 handler）。
    """
    logger_name = "trace_jsonl"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        path = Path(trace_file or _DEFAULT_TRACE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = _JsonLinesHandler(str(path), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False   # 不向上传播，避免二次输出到 stderr
    return logger


def write_trace(trace_dict: Dict[str, Any], trace_file: Optional[str] = None) -> None:
    """
    将 trace 字典作为一行 JSON 追加写入 traces.jsonl。

    这是 TraceCollector 的底层写入接口。
    trace_dict 通常来自 TraceContext.to_dict()。

    Args:
        trace_dict:  要持久化的 trace 字典（必须可 json.dumps）。
        trace_file:  覆盖默认文件路径（主要用于测试）。
    """
    path = Path(trace_file or _DEFAULT_TRACE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(trace_dict, ensure_ascii=False, default=str) + "\n"
    with _file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


default_logger = get_logger()
