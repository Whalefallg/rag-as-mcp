"""CompositeEvaluator (src/observability/evaluation/composite_evaluator.py)"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    """
    组合多个 Evaluator 并行执行，合并所有指标。

    Usage:
        composite = CompositeEvaluator([
            LocalRetrievalEvaluator(),
            RagasEvaluator(llm=my_llm),
        ])
        metrics = composite.evaluate(
            query="...",
            retrieved_chunks=[...],
            generated_answer="...",
        )
        # → {"hit_rate": 0.9, "faithfulness": 0.85, "answer_relevancy": 0.8}
    """

    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        max_workers: int = 4,
        **kwargs,
    ) -> None:
        """
        Args:
            evaluators:  要并行运行的 Evaluator 实例列表。
            max_workers: ThreadPoolExecutor 的最大线程数。
        """
        super().__init__(**kwargs)
        if not evaluators:
            raise ValueError("CompositeEvaluator 至少需要一个 Evaluator。")
        self._evaluators = evaluators
        self._max_workers = max_workers

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        并行执行所有 Evaluator，合并指标。

        Args:
            query:            用户原始问题。
            retrieved_chunks: 检索结果列表。
            generated_answer: LLM 生成的回答（可选）。
            ground_truth:     标准答案（可选）。
        Returns:
            合并后的指标字典，额外包含 _errors 字段（空 dict 表示全部成功）。
        """
        merged: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        def _run_one(evaluator: BaseEvaluator) -> Dict[str, Any]:
            return evaluator.evaluate(
                query=query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_name = {
                executor.submit(_run_one, ev): _evaluator_name(ev)
                for ev in self._evaluators
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    merged.update(result)
                except Exception as exc:
                    errors[name] = str(exc)

        if errors:
            merged["_errors"] = errors
        return merged

    @property
    def evaluators(self) -> List[BaseEvaluator]:
        return list(self._evaluators)


def _evaluator_name(evaluator: BaseEvaluator) -> str:
    return type(evaluator).__name__
