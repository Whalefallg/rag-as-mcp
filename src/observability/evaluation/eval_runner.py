"""
EvalRunner (src/observability/evaluation/eval_runner.py)
=========================================================
为什么需要这个文件：
  评估不是"跑一条"而是"跑一批"——golden_test_set.json 里有 N 条测试用例，
  每条都需要走完整的 Retrieval 链路然后打分。
  EvalRunner 把"读取测试集 → 逐条检索 → 逐条评估 → 汇总报告"串成一次调用，
  CI 里直接跑 `python scripts/evaluate.py` 即可输出 Hit Rate / MRR 等指标。

  EvalReport 结构：
    - summary：汇总指标（所有用例的均值）
    - per_query：每条用例的详细结果
    - elapsed_ms：总耗时
    - test_set_path：使用的测试集路径

  离线优先：默认使用 LocalRetrievalEvaluator，无需 API Key 即可在 CI 中运行。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """单条测试用例的评估结果"""
    query: str
    retrieved_chunk_ids: List[str]
    retrieved_sources: List[str]
    metrics: Dict[str, float]
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvalReport:
    """完整评估报告"""
    summary: Dict[str, float]       # 各指标均值
    per_query: List[QueryResult]
    elapsed_ms: float
    test_set_path: str
    total_cases: int
    failed_cases: int

    def print_summary(self) -> None:
        """打印人类可读的汇总报告"""
        print(f"\n{'='*50}")
        print(f"评估报告  |  测试集：{self.test_set_path}")
        print(f"{'='*50}")
        print(f"用例总数：{self.total_cases}  失败：{self.failed_cases}")
        print(f"总耗时：{self.elapsed_ms:.0f} ms\n")
        print("指标汇总：")
        for k, v in self.summary.items():
            print(f"  {k:30s} {v:.4f}")
        print(f"{'='*50}\n")


class EvalRunner:
    """
    评估运行器：读取 golden_test_set.json，逐条检索并评估。

    Usage:
        runner = EvalRunner(settings=settings)
        report = runner.run("tests/fixtures/golden_test_set.json")
        report.print_summary()
    """

    def __init__(
        self,
        settings,
        hybrid_search=None,
        evaluator=None,
        collection: str = "default",
    ) -> None:
        """
        Args:
            settings:      全局配置。
            hybrid_search: HybridSearch 实例（None 时懒加载）。
            evaluator:     评估器实例（None 时使用 LocalRetrievalEvaluator）。
            collection:    目标知识库集合。
        """
        self._settings = settings
        self._search = hybrid_search
        self._evaluator = evaluator
        self._collection = collection

    def run(self, test_set_path: str = "tests/fixtures/golden_test_set.json") -> EvalReport:
        """
        读取 golden_test_set.json，逐条执行检索 + 评估，返回 EvalReport。

        Args:
            test_set_path: golden test set 文件路径。
        Returns:
            EvalReport（含 summary / per_query / elapsed_ms）。
        """
        test_cases = self._load_test_set(test_set_path)
        evaluator = self._get_evaluator()
        search = self._get_search()

        results: List[QueryResult] = []
        t_start = time.monotonic()

        for case in test_cases:
            result = self._run_one(case, search, evaluator)
            results.append(result)

        elapsed_ms = (time.monotonic() - t_start) * 1000
        summary = self._compute_summary(results)
        failed = sum(1 for r in results if r.error is not None)

        return EvalReport(
            summary=summary,
            per_query=results,
            elapsed_ms=elapsed_ms,
            test_set_path=str(Path(test_set_path).resolve()),
            total_cases=len(results),
            failed_cases=failed,
        )

    # ──────────────────────────────────────────────────────────────────────

    def _run_one(self, case: Dict[str, Any], search, evaluator) -> QueryResult:
        query = case.get("query", "")
        expected_ids = case.get("expected_chunk_ids", [])
        expected_sources = case.get("expected_sources", [])

        try:
            raw_results = search.search(
                query=query,
                top_k=self._settings.retrieval.top_k_final,
                collection=self._collection,
            )
            chunks = [
                {
                    "id": r.chunk_id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in raw_results
            ]
            metrics = evaluator.evaluate(
                query=query,
                retrieved_chunks=chunks,
                ground_truth={
                    "expected_chunk_ids": expected_ids,
                    "expected_sources": expected_sources,
                    "answer": case.get("expected_answer", ""),
                },
            )
            return QueryResult(
                query=query,
                retrieved_chunk_ids=[c["id"] for c in chunks],
                retrieved_sources=list({
                    c["metadata"].get("source_path", "") for c in chunks
                }),
                metrics=metrics,
                expected_chunk_ids=expected_ids,
                expected_sources=expected_sources,
            )
        except Exception as exc:
            return QueryResult(
                query=query,
                retrieved_chunk_ids=[],
                retrieved_sources=[],
                metrics={},
                expected_chunk_ids=expected_ids,
                expected_sources=expected_sources,
                error=str(exc),
            )

    def _load_test_set(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"golden_test_set 文件不存在：{path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        cases = data.get("test_cases", [])
        if not cases:
            raise ValueError(f"golden_test_set 为空：{path}")
        return cases

    def _get_search(self):
        if self._search is not None:
            return self._search
        from src.core.query_engine.hybrid_search import HybridSearch
        self._search = HybridSearch(self._settings)
        return self._search

    def _get_evaluator(self):
        if self._evaluator is not None:
            return self._evaluator
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        self._evaluator = LocalRetrievalEvaluator(
            k=self._settings.retrieval.top_k_final
        )
        return self._evaluator

    @staticmethod
    def _compute_summary(results: List[QueryResult]) -> Dict[str, float]:
        """计算所有成功用例的指标均值"""
        successful = [r for r in results if r.error is None and r.metrics]
        if not successful:
            return {}
        # 收集所有指标 key
        all_keys = set()
        for r in successful:
            all_keys.update(k for k in r.metrics if not k.startswith("_"))

        summary = {}
        for key in sorted(all_keys):
            values = [r.metrics[key] for r in successful if key in r.metrics]
            if values:
                summary[key] = round(sum(values) / len(values), 4)
        return summary
