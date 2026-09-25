"""
TraceContext 单元测试 (tests/unit/test_trace_context.py)
=========================================================
验收标准 (DEV_SPEC F1)：
  - record_stage 追加阶段数据
  - finish() 后 to_dict() 输出包含所有必填字段
  - to_dict() 输出可直接 json.dumps() 序列化
  - elapsed_ms() 查询指定阶段或总耗时
  - trace_type 字段正确传递
  - span() 上下文管理器自动计时
"""
import json
import time
import pytest
from src.core.trace.trace_context import TraceContext


class TestTraceContextInit:
    def test_default_trace_type_is_query(self):
        t = TraceContext()
        assert t.trace_type == "query"

    def test_custom_trace_type(self):
        t = TraceContext(trace_type="ingestion")
        assert t.trace_type == "ingestion"

    def test_auto_generates_trace_id(self):
        t1, t2 = TraceContext(), TraceContext()
        assert t1.trace_id != t2.trace_id
        assert len(t1.trace_id) == 36   # UUID4 格式

    def test_custom_trace_id(self):
        t = TraceContext(trace_id="fixed-id")
        assert t.trace_id == "fixed-id"


class TestRecordStage:
    def test_appends_stage(self):
        t = TraceContext()
        t.record_stage("load", chunk_count=5)
        d = t.to_dict()
        stages = d["stages"]
        assert len(stages) == 1
        assert stages[0]["stage"] == "load"
        assert stages[0]["chunk_count"] == 5

    def test_multiple_stages_ordered(self):
        t = TraceContext()
        t.record_stage("split")
        t.record_stage("encode")
        t.record_stage("upsert")
        stages = [s["stage"] for s in t.to_dict()["stages"]]
        assert stages == ["split", "encode", "upsert"]

    def test_duration_ms_stored(self):
        t = TraceContext()
        t.record_stage("rerank", duration_ms=12.34)
        assert t.to_dict()["stages"][0]["duration_ms"] == 12.34

    def test_error_stored(self):
        t = TraceContext()
        t.record_stage("dense_retrieval", error="timeout")
        assert t.to_dict()["stages"][0]["error"] == "timeout"

    def test_no_required_kwargs(self):
        t = TraceContext()
        t.record_stage("fusion")   # 不传 kwargs 不应报错
        assert t.to_dict()["stages"][0]["stage"] == "fusion"


class TestFinishAndElapsed:
    def test_finish_sets_total_elapsed_ms(self):
        t = TraceContext()
        time.sleep(0.01)
        t.finish()
        total = t.elapsed_ms()
        assert total is not None
        assert total >= 10   # 至少 10ms

    def test_elapsed_before_finish_returns_none(self):
        t = TraceContext()
        assert t.elapsed_ms() is None

    def test_finish_idempotent(self):
        t = TraceContext()
        t.finish()
        first = t.elapsed_ms()
        time.sleep(0.01)
        t.finish()   # 第二次调用不应改变结束时间
        second = t.elapsed_ms()
        assert abs(first - second) < 1   # 差值 < 1ms

    def test_elapsed_ms_for_specific_stage(self):
        t = TraceContext()
        t.record_stage("load", duration_ms=50.0)
        t.record_stage("split", duration_ms=20.0)
        assert t.elapsed_ms("load") == 50.0
        assert t.elapsed_ms("split") == 20.0

    def test_elapsed_ms_unknown_stage_returns_none(self):
        t = TraceContext()
        assert t.elapsed_ms("nonexistent") is None


class TestToDict:
    def test_required_fields_present(self):
        t = TraceContext(trace_type="ingestion")
        t.record_stage("load")
        t.finish()
        d = t.to_dict()
        for key in ("trace_id", "trace_type", "started_at",
                    "finished_at", "total_elapsed_ms", "metadata", "stages"):
            assert key in d, f"missing key: {key}"

    def test_trace_type_in_output(self):
        t = TraceContext(trace_type="ingestion")
        t.finish()
        assert t.to_dict()["trace_type"] == "ingestion"

    def test_json_serializable(self):
        t = TraceContext()
        t.record_stage("rerank", duration_ms=5.5, result_count=10)
        t.finish()
        raw = json.dumps(t.to_dict())   # 不应抛异常
        data = json.loads(raw)
        assert data["trace_id"] == t.trace_id

    def test_metadata_included(self):
        t = TraceContext()
        t.set_metadata("user_query", "hello")
        t.set_metadata("collection", "default")
        t.finish()
        meta = t.to_dict()["metadata"]
        assert meta["user_query"] == "hello"
        assert meta["collection"] == "default"

    def test_started_at_is_iso_string(self):
        t = TraceContext()
        t.finish()
        started = t.to_dict()["started_at"]
        assert "T" in started   # ISO 8601 格式含 T 分隔符


class TestSpanContextManager:
    def test_span_records_duration(self):
        t = TraceContext()
        with t.span("dense_retrieval") as span:
            span.data["method"] = "dense"
            time.sleep(0.005)
        stages = t.to_dict()["stages"]
        assert len(stages) == 1
        assert stages[0]["stage"] == "dense_retrieval"
        assert stages[0]["duration_ms"] >= 4   # ≥ 4ms
        assert stages[0]["method"] == "dense"

    def test_span_records_exception(self):
        t = TraceContext()
        with pytest.raises(ValueError):
            with t.span("load") as span:
                raise ValueError("file not found")
        stage = t.to_dict()["stages"][0]
        assert stage["error"] == "file not found"
        assert stage["duration_ms"] is not None

    def test_span_reraises_exception(self):
        t = TraceContext()
        with pytest.raises(RuntimeError):
            with t.span("encode"):
                raise RuntimeError("boom")

    def test_multiple_spans_ordered(self):
        t = TraceContext()
        with t.span("step1"):
            pass
        with t.span("step2"):
            pass
        stages = [s["stage"] for s in t.to_dict()["stages"]]
        assert stages == ["step1", "step2"]


class TestMetadata:
    def test_set_and_get(self):
        t = TraceContext()
        t.set_metadata("key", "val")
        assert t.get_metadata("key") == "val"

    def test_get_missing_returns_default(self):
        t = TraceContext()
        assert t.get_metadata("missing") is None
        assert t.get_metadata("missing", 42) == 42

    def test_overwrite_metadata(self):
        t = TraceContext()
        t.set_metadata("k", "v1")
        t.set_metadata("k", "v2")
        assert t.get_metadata("k") == "v2"
