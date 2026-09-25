"""
EvalRunner 单元测试 (tests/unit/test_eval_runner.py)
=====================================================
验收标准 (DEV_SPEC H3)：
  - run() 返回 EvalReport，summary 含均值指标
  - per_query 长度 == 测试用例数
  - golden_test_set 不存在时抛 FileNotFoundError
  - 空 test_cases 时抛 ValueError
  - 检索失败时 QueryResult.error 非 None，failed_cases 计数
  - compute_summary 对空列表返回 {}
  - EvalReport.print_summary() 不抛异常
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.observability.evaluation.eval_runner import EvalRunner, EvalReport, QueryResult
from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
from src.core.types import RetrievalResult


def _make_settings():
    from src.core.settings import (
        Settings, LLMConfig, EmbeddingConfig,
        VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig,
    )
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(
            provider="openai", model="text-embedding-3-small", api_key="k"
        ),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="/tmp/tc"),
        splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10,
        ),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def _write_test_set(path: Path, cases: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"test_cases": cases}),
        encoding="utf-8",
    )


def _make_mock_search(results_map: dict = None):
    """results_map: {query: [RetrievalResult, ...]}"""
    mock = MagicMock()
    _map = results_map or {}

    def _search(query, top_k=10, collection="default", **kw):
        return _map.get(query, [])

    mock.search.side_effect = _search
    return mock


_SAMPLE_CASES = [
    {
        "query": "query_a",
        "expected_chunk_ids": ["c1"],
        "expected_sources": [],
    },
    {
        "query": "query_b",
        "expected_chunk_ids": [],
        "expected_sources": ["doc.pdf"],
    },
]


class TestEvalRunnerBasic:
    def test_returns_eval_report(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, _SAMPLE_CASES)

        mock_search = _make_mock_search({
            "query_a": [RetrievalResult("c1", 0.9, "text", {"source_path": ""})],
            "query_b": [RetrievalResult("cx", 0.8, "text", {"source_path": "data/doc.pdf"})],
        })
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=mock_search,
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))

        assert isinstance(report, EvalReport)
        assert report.total_cases == 2
        assert report.failed_cases == 0

    def test_per_query_length_matches_cases(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, _SAMPLE_CASES)
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=_make_mock_search(),
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))
        assert len(report.per_query) == 2

    def test_summary_contains_hit_rate(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, _SAMPLE_CASES)
        mock_search = _make_mock_search({
            "query_a": [RetrievalResult("c1", 0.9, "text", {})],
            "query_b": [],
        })
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=mock_search,
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))
        assert "hit_rate" in report.summary
        assert isinstance(report.summary["hit_rate"], float)

    def test_elapsed_ms_positive(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, _SAMPLE_CASES)
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=_make_mock_search(),
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))
        assert report.elapsed_ms > 0

    def test_test_set_path_in_report(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, _SAMPLE_CASES)
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=_make_mock_search(),
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))
        assert "gs.json" in report.test_set_path


class TestEvalRunnerErrors:
    def test_missing_file_raises(self):
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=_make_mock_search(),
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        with pytest.raises(FileNotFoundError):
            runner.run("/nonexistent/golden.json")

    def test_empty_test_cases_raises(self, tmp_path):
        f = tmp_path / "gs.json"
        f.write_text(json.dumps({"test_cases": []}))
        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=_make_mock_search(),
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        with pytest.raises(ValueError):
            runner.run(str(f))

    def test_search_failure_recorded_as_error(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, [_SAMPLE_CASES[0]])

        mock_search = MagicMock()
        mock_search.search.side_effect = RuntimeError("timeout")

        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=mock_search,
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))
        assert report.failed_cases == 1
        assert report.per_query[0].error is not None

    def test_summary_excludes_failed_cases(self, tmp_path):
        f = tmp_path / "gs.json"
        _write_test_set(f, _SAMPLE_CASES)

        mock_search = MagicMock()
        mock_search.search.side_effect = RuntimeError("all fail")

        runner = EvalRunner(
            settings=_make_settings(),
            hybrid_search=mock_search,
            evaluator=LocalRetrievalEvaluator(k=10),
        )
        report = runner.run(str(f))
        # 全部失败时 summary 为空
        assert report.summary == {}


class TestComputeSummary:
    def test_returns_empty_for_no_results(self):
        assert EvalRunner._compute_summary([]) == {}

    def test_averages_metric_correctly(self):
        results = [
            QueryResult("q1", [], [], {"hit_rate": 1.0, "mrr": 1.0, "precision_at_k": 0.5}),
            QueryResult("q2", [], [], {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0}),
        ]
        summary = EvalRunner._compute_summary(results)
        assert summary["hit_rate"] == pytest.approx(0.5)
        assert summary["mrr"] == pytest.approx(0.5)

    def test_excludes_underscore_keys(self):
        results = [
            QueryResult("q1", [], [], {"hit_rate": 1.0, "_errors": {"A": "fail"}}),
        ]
        summary = EvalRunner._compute_summary(results)
        assert "_errors" not in summary
        assert "hit_rate" in summary

    def test_ignores_error_records(self):
        results = [
            QueryResult("q1", [], [], {"hit_rate": 1.0}),
            QueryResult("q2", [], [], {}, error="timeout"),
        ]
        summary = EvalRunner._compute_summary(results)
        # q2 没有指标，不影响 q1 的结果
        assert summary["hit_rate"] == pytest.approx(1.0)


class TestEvalReport:
    def test_print_summary_no_exception(self, capsys):
        report = EvalReport(
            summary={"hit_rate": 0.8, "mrr": 0.6},
            per_query=[],
            elapsed_ms=123.4,
            test_set_path="/tmp/gs.json",
            total_cases=5,
            failed_cases=1,
        )
        report.print_summary()   # 不应抛异常
        captured = capsys.readouterr()
        assert "hit_rate" in captured.out
        assert "0.8000" in captured.out
