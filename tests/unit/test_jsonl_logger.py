"""
JSON Lines Logger 单元测试 (tests/unit/test_jsonl_logger.py)
=============================================================
验收标准 (DEV_SPEC F2)：
  - write_trace() 写入后文件新增一行合法 JSON
  - 该行包含 trace_type 字段
  - 并发写入不产生乱行（多线程安全）
  - 目录不存在时自动创建
  - get_logger() 仍正常输出到 stderr（人类可读）
"""
import json
import os
import tempfile
import threading
from pathlib import Path

import pytest

from src.observability.logger import write_trace, get_logger


class TestWriteTrace:
    def test_creates_file_and_writes_one_line(self, tmp_path):
        trace_file = str(tmp_path / "traces.jsonl")
        payload = {
            "trace_id": "abc-123",
            "trace_type": "query",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "total_elapsed_ms": 1000.0,
            "metadata": {},
            "stages": [],
        }
        write_trace(payload, trace_file=trace_file)

        lines = Path(trace_file).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["trace_id"] == "abc-123"

    def test_trace_type_field_present(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        write_trace({"trace_type": "ingestion", "trace_id": "x"}, trace_file=trace_file)
        line = Path(trace_file).read_text().strip()
        assert json.loads(line)["trace_type"] == "ingestion"

    def test_appends_multiple_traces(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        for i in range(5):
            write_trace({"trace_id": str(i)}, trace_file=trace_file)
        lines = Path(trace_file).read_text().splitlines()
        assert len(lines) == 5
        for i, line in enumerate(lines):
            assert json.loads(line)["trace_id"] == str(i)

    def test_auto_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "logs" / "sub" / "traces.jsonl")
        write_trace({"trace_id": "y"}, trace_file=nested)
        assert Path(nested).exists()

    def test_concurrent_writes_no_corruption(self, tmp_path):
        trace_file = str(tmp_path / "concurrent.jsonl")
        n_threads = 20

        def _write(i):
            write_trace({"trace_id": str(i), "data": "x" * 50}, trace_file=trace_file)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = Path(trace_file).read_text().splitlines()
        assert len(lines) == n_threads
        for line in lines:
            parsed = json.loads(line)   # 每行必须是合法 JSON
            assert "trace_id" in parsed

    def test_non_ascii_content_preserved(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        write_trace({"trace_id": "zh", "query": "你好世界"}, trace_file=trace_file)
        line = Path(trace_file).read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert parsed["query"] == "你好世界"

    def test_non_serializable_value_uses_str_fallback(self, tmp_path):
        """default=str 确保非序列化对象不抛异常"""
        trace_file = str(tmp_path / "t.jsonl")
        from datetime import datetime
        write_trace({"ts": datetime(2026, 1, 1)}, trace_file=trace_file)
        line = Path(trace_file).read_text().strip()
        assert json.loads(line)  # 不抛异常即可


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test.module")
        assert logger is not None
        assert logger.name == "test.module"

    def test_same_name_returns_same_instance(self):
        l1 = get_logger("same.name")
        l2 = get_logger("same.name")
        assert l1 is l2

    def test_has_stderr_handler(self):
        import sys
        logger = get_logger("stderr.check")
        has_stderr = any(
            getattr(h, "stream", None) is sys.stderr
            for h in logger.handlers
        )
        assert has_stderr
