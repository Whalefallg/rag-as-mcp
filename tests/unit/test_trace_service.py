"""
TraceService 单元测试 (tests/unit/test_trace_service.py)
=========================================================
验收标准 (DEV_SPEC G5)：
  - list() 按时间倒序返回记录
  - trace_type 过滤正确
  - keyword 关键词过滤（user_query / source_path）
  - 文件不存在时返回空列表
  - get() 按 trace_id 精确查找
  - exists() 正确反映文件是否存在
  - 格式错误的行被跳过（容错）
"""
import json
import pytest
from pathlib import Path
from src.observability.dashboard.services.trace_service import TraceService, TraceRecord


def _write_traces(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_trace(
    trace_id: str,
    trace_type: str = "query",
    started_at: str = "2026-01-01T00:00:00+00:00",
    user_query: str = "test query",
    source_path: str = "",
    stages: list = None,
    total_ms: float = 100.0,
) -> dict:
    return {
        "trace_id": trace_id,
        "trace_type": trace_type,
        "started_at": started_at,
        "finished_at": started_at,
        "total_elapsed_ms": total_ms,
        "metadata": {
            "user_query": user_query,
            "source_path": source_path,
            "collection": "default",
        },
        "stages": stages or [],
    }


class TestTraceServiceList:
    def test_returns_empty_when_file_missing(self, tmp_path):
        svc = TraceService(trace_file=str(tmp_path / "missing.jsonl"))
        assert svc.list() == []

    def test_returns_all_records(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [
            _make_trace("id1", started_at="2026-01-01T10:00:00+00:00"),
            _make_trace("id2", started_at="2026-01-02T10:00:00+00:00"),
        ])
        svc = TraceService(trace_file=str(f))
        records = svc.list()
        assert len(records) == 2

    def test_sorted_descending_by_started_at(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [
            _make_trace("old", started_at="2026-01-01T00:00:00+00:00"),
            _make_trace("new", started_at="2026-06-01T00:00:00+00:00"),
        ])
        svc = TraceService(trace_file=str(f))
        records = svc.list()
        assert records[0].trace_id == "new"
        assert records[1].trace_id == "old"

    def test_filter_by_trace_type_query(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [
            _make_trace("q1", trace_type="query"),
            _make_trace("i1", trace_type="ingestion"),
            _make_trace("q2", trace_type="query"),
        ])
        svc = TraceService(trace_file=str(f))
        records = svc.list(trace_type="query")
        assert len(records) == 2
        assert all(r.trace_type == "query" for r in records)

    def test_filter_by_trace_type_ingestion(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [
            _make_trace("q1", trace_type="query"),
            _make_trace("i1", trace_type="ingestion"),
        ])
        svc = TraceService(trace_file=str(f))
        records = svc.list(trace_type="ingestion")
        assert len(records) == 1
        assert records[0].trace_id == "i1"

    def test_keyword_filters_by_user_query(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [
            _make_trace("r1", user_query="Azure 配置问题"),
            _make_trace("r2", user_query="OpenAI 接口"),
        ])
        svc = TraceService(trace_file=str(f))
        records = svc.list(keyword="azure")
        assert len(records) == 1
        assert records[0].trace_id == "r1"

    def test_keyword_case_insensitive(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [_make_trace("r1", user_query="Hello World")])
        svc = TraceService(trace_file=str(f))
        assert len(svc.list(keyword="hello")) == 1
        assert len(svc.list(keyword="WORLD")) == 1

    def test_limit_respected(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [_make_trace(f"id{i}") for i in range(20)])
        svc = TraceService(trace_file=str(f))
        assert len(svc.list(limit=5)) == 5

    def test_malformed_lines_skipped(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text(
            '{"trace_id":"ok","trace_type":"query","started_at":"2026-01-01","metadata":{},"stages":[]}\n'
            'NOT VALID JSON\n'
            '\n'
        )
        svc = TraceService(trace_file=str(f))
        records = svc.list()
        assert len(records) == 1
        assert records[0].trace_id == "ok"


class TestTraceServiceGet:
    def test_get_by_trace_id(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [
            _make_trace("target-id"),
            _make_trace("other-id"),
        ])
        svc = TraceService(trace_file=str(f))
        rec = svc.get("target-id")
        assert rec is not None
        assert rec.trace_id == "target-id"

    def test_get_missing_returns_none(self, tmp_path):
        f = tmp_path / "t.jsonl"
        _write_traces(f, [_make_trace("id1")])
        svc = TraceService(trace_file=str(f))
        assert svc.get("nonexistent") is None


class TestTraceServiceExists:
    def test_exists_true_when_file_present(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text("")
        svc = TraceService(trace_file=str(f))
        assert svc.exists() is True

    def test_exists_false_when_file_missing(self, tmp_path):
        svc = TraceService(trace_file=str(tmp_path / "missing.jsonl"))
        assert svc.exists() is False


class TestTraceRecord:
    def test_user_query_from_metadata(self):
        rec = TraceRecord(
            trace_id="x", trace_type="query",
            started_at="2026-01-01", finished_at=None, total_elapsed_ms=50.0,
            metadata={"user_query": "hello", "collection": "test"},
        )
        assert rec.user_query == "hello"
        assert rec.collection == "test"

    def test_stage_by_name_found(self):
        rec = TraceRecord(
            trace_id="x", trace_type="query",
            started_at="2026-01-01", finished_at=None, total_elapsed_ms=50.0,
            stages=[
                {"stage": "dense_retrieval", "result_count": 10},
                {"stage": "rerank", "result_count": 5},
            ],
        )
        s = rec.stage_by_name("rerank")
        assert s is not None
        assert s["result_count"] == 5

    def test_stage_by_name_not_found(self):
        rec = TraceRecord(
            trace_id="x", trace_type="query",
            started_at="2026-01-01", finished_at=None, total_elapsed_ms=None,
        )
        assert rec.stage_by_name("nonexistent") is None
