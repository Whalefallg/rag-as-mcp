"""
TraceCollector 单元测试 (tests/unit/test_trace_collector.py)
=============================================================
验收标准 (DEV_SPEC F3)：
  - collect() 自动调用 trace.finish()（幂等）
  - collect() 把 trace 写入指定 traces.jsonl 文件
  - 文件内容可被解析为包含 trace_type 的合法 JSON
  - 写入失败时不抛出异常（降级日志）
  - global_collector 是 TraceCollector 实例
"""
import json
import pytest
from pathlib import Path
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import TraceCollector, global_collector


class TestTraceCollector:
    def test_collect_writes_to_file(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        collector = TraceCollector(trace_file=trace_file)

        trace = TraceContext(trace_type="query")
        trace.record_stage("dense_retrieval", result_count=10)
        trace.finish()

        collector.collect(trace)

        lines = Path(trace_file).read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["trace_type"] == "query"
        assert parsed["trace_id"] == trace.trace_id

    def test_collect_auto_finishes_unfinished_trace(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        collector = TraceCollector(trace_file=trace_file)

        trace = TraceContext()
        # 不调用 trace.finish()
        collector.collect(trace)

        parsed = json.loads(Path(trace_file).read_text().strip())
        assert parsed["total_elapsed_ms"] is not None   # finish() 已被自动调用

    def test_collect_multiple_traces(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        collector = TraceCollector(trace_file=trace_file)

        for i in range(3):
            t = TraceContext(trace_type="ingestion")
            t.finish()
            collector.collect(t)

        lines = Path(trace_file).read_text().splitlines()
        assert len(lines) == 3

    def test_collect_stages_in_output(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        collector = TraceCollector(trace_file=trace_file)

        trace = TraceContext(trace_type="ingestion")
        trace.record_stage("load", duration_ms=50.0)
        trace.record_stage("split", duration_ms=20.0, chunk_count=15)
        trace.finish()
        collector.collect(trace)

        parsed = json.loads(Path(trace_file).read_text().strip())
        stages = parsed["stages"]
        assert len(stages) == 2
        assert stages[0]["stage"] == "load"
        assert stages[1]["chunk_count"] == 15

    def test_write_failure_does_not_raise(self, tmp_path):
        """写入路径不可用时 collect() 不应抛出，只记录日志"""
        # 用一个无法写入的路径
        collector = TraceCollector(trace_file="/dev/full/impossible/path/t.jsonl")
        trace = TraceContext()
        trace.finish()
        # 不应抛出异常
        collector.collect(trace)

    def test_idempotent_finish(self, tmp_path):
        """collect() 对已 finish 的 trace 应幂等"""
        trace_file = str(tmp_path / "t.jsonl")
        collector = TraceCollector(trace_file=trace_file)

        trace = TraceContext()
        trace.finish()
        first_elapsed = trace.elapsed_ms()

        import time; time.sleep(0.01)
        collector.collect(trace)  # 再次 finish 不改变 elapsed

        parsed = json.loads(Path(trace_file).read_text().strip())
        assert abs(parsed["total_elapsed_ms"] - first_elapsed) < 1


class TestGlobalCollector:
    def test_is_trace_collector_instance(self):
        assert isinstance(global_collector, TraceCollector)

    def test_has_default_trace_file(self):
        # global_collector 的 _trace_file 为 None（使用默认路径）
        assert global_collector._trace_file is None
