"""Evaluator 抽象层 (src/libs/evaluator/base_evaluator.py)"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseEvaluator(ABC):
    """评估器抽象基类"""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        评估一次 RAG 问答的质量。

        Args:
            query: 用户的原始问题。
            retrieved_chunks: 检索返回的 chunk 列表，每条包含 id/text/score。
            generated_answer: LLM 生成的回答（可选，不传则只评估检索质量）。
            ground_truth: 标准答案，来自 golden_test_set.json（可选）。
        Returns:
            指标字典，例如 {"hit_rate": 0.9, "faithfulness": 0.85}。
        """
        pass
