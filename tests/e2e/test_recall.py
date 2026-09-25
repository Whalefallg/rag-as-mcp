"""
Recall 回归测试 (tests/e2e/test_recall.py)
==========================================
验收标准 (DEV_SPEC H5)：
  - 基于 golden_test_set.json 运行检索，hit@k 达到最小阈值
  - 阈值写死在测试里，便于 CI 回归检测
  - 使用 mock HybridSearch，完全离线（无需真实 ChromaDB）
  - 通过率 < MIN_HIT_RATE 时测试失败

  注意：这是 E2E 测试，关注的是"评估流程能跑通"，
  而不是"真实检索质量"（真实质量需摄取真实数据后手动验证）。
  因此使用 mock search，令部分 query 命中期望 source，模拟真实场景。
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
from src.core.types import RetrievalResult

# ── 回归阈值（CI 红线）─────────────────────────────────────────────────────
MIN_HIT_RATE = 0.5    # 至少 50% 的测试用例命中期望文档
MIN_MRR      = 0.3    # MRR 至少 0.3


_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_GOLDEN_SET   = _FIXTURES_DIR / "golden_test_set.json"


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


def _make_mock_search(golden_cases: list):
    """
    构造一个 mock HybridSearch，对每条 golden case 的 expected_sources 返回命中结果。
    每 2 条命中 1 条（保证 hit_rate ≈ 0.5 稳定高于阈值）。
    """
    mock = MagicMock()

    case_map = {}
    for i, case in enumerate(golden_cases):
        query = case["query"]
        sources = case.get("expected_sources", [])
        # 偶数索引返回命中结果，奇数索引返回不命中
        if i % 2 == 0 and sources:
            results = [
                RetrievalResult(
                    chunk_id=f"chunk_{i}_0",
                    text="matching content",
                    score=0.9,
                    metadata={"source_path": f"data/{sources[0]}"},
                )
            ]
        else:
            results = [
                RetrievalResult(
                    chunk_id=f"chunk_{i}_miss",
                    text="unrelated content",
                    score=0.5,
                    metadata={"source_path": "other/irrelevant.pdf"},
                )
            ]
        case_map[query] = results

    def _search(query, top_k=10, collection="default", **kwargs):
        return case_map.get(query, [])

    mock.search.side_effect = _search
    return mock


@pytest.fixture
def golden_cases():
    assert _GOLDEN_SET.exists(), f"golden_test_set.json 不存在：{_GOLDEN_SET}"
    data = json.loads(_GOLDEN_SET.read_text(encoding="utf-8"))
    return data["test_cases"]


class TestRecallRegression:
    def test_golden_test_set_exists(self):
        assert _GOLDEN_SET.exists(), "tests/fixtures/golden_test_set.json 必须存在"

    def test_golden_test_set_has_cases(self, golden_cases):
        assert len(golden_cases) >= 5, "黄金测试集至少需要 5 条用例"

    def test_golden_cases_have_required_fields(self, golden_cases):
        for case in golden_cases:
            assert "query" in case, f"缺少 query 字段：{case}"
            # expected_chunk_ids 或 expected_sources 至少有一个
            has_ids = bool(case.get("expected_chunk_ids"))
            has_sources = bool(case.get("expected_sources"))
            assert has_ids or has_sources, f"测试用例缺少期望结果：{case['query']}"

    def test_hit_rate_above_threshold(self, golden_cases):
        """
        E2E：使用 mock search 跑完整评估流程，验证 hit_rate >= MIN_HIT_RATE。
        mock search 模拟"一半命中"，用于验证评估逻辑正确性。
        """
        settings = _make_settings()
        mock_search = _make_mock_search(golden_cases)
        evaluator = LocalRetrievalEvaluator(k=10)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=mock_search,
            evaluator=evaluator,
        )
        report = runner.run(str(_GOLDEN_SET))

        assert report.total_cases == len(golden_cases)
        assert report.failed_cases == 0, f"评估失败 {report.failed_cases} 条"

        hit_rate = report.summary.get("hit_rate", 0.0)
        assert hit_rate >= MIN_HIT_RATE, (
            f"hit_rate={hit_rate:.4f} 低于阈值 {MIN_HIT_RATE}。"
            f"请检查检索链路或降低阈值。"
        )

    def test_mrr_above_threshold(self, golden_cases):
        """MRR 回归：第一个命中结果的倒数排名均值 >= MIN_MRR。"""
        settings = _make_settings()
        mock_search = _make_mock_search(golden_cases)
        evaluator = LocalRetrievalEvaluator(k=10)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=mock_search,
            evaluator=evaluator,
        )
        report = runner.run(str(_GOLDEN_SET))

        mrr = report.summary.get("mrr", 0.0)
        assert mrr >= MIN_MRR, (
            f"mrr={mrr:.4f} 低于阈值 {MIN_MRR}。"
        )

    def test_eval_report_structure(self, golden_cases):
        """验证 EvalReport 结构完整性"""
        settings = _make_settings()
        mock_search = _make_mock_search(golden_cases)
        evaluator = LocalRetrievalEvaluator(k=10)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=mock_search,
            evaluator=evaluator,
        )
        report = runner.run(str(_GOLDEN_SET))

        assert report.total_cases > 0
        assert report.elapsed_ms > 0
        assert isinstance(report.summary, dict)
        assert len(report.per_query) == report.total_cases

        for r in report.per_query:
            assert r.query
            assert isinstance(r.metrics, dict)
            assert "hit_rate" in r.metrics
            assert "mrr" in r.metrics

    def test_per_query_retrieved_sources_populated(self, golden_cases):
        """验证每条用例的 retrieved_sources 有值"""
        settings = _make_settings()
        mock_search = _make_mock_search(golden_cases)
        evaluator = LocalRetrievalEvaluator(k=10)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=mock_search,
            evaluator=evaluator,
        )
        report = runner.run(str(_GOLDEN_SET))
        # 至少有一条用例有非空的 retrieved_sources
        has_results = any(r.retrieved_sources for r in report.per_query)
        assert has_results, "所有用例的检索结果均为空，mock search 可能未正确配置"
