"""
Evaluator 抽象层 (src/libs/evaluator/base_evaluator.py)
=======================================================
为什么需要这个文件：
  RAG 系统的质量需要量化——不能只靠主观感受判断检索好不好。
  BaseEvaluator 定义评估接口，支持只评估检索（Hit Rate/MRR）
  或完整 RAG 评估（Faithfulness/Answer Relevancy）。
  Phase G 的 Dashboard 会调用具体实现来驱动评估面板。

本文件定义 RAG 质量评估的统一抽象接口。

类说明:
  - BaseEvaluator : 评估器抽象基类。所有具体评估框架（Ragas/DeepEval/自定义指标）
                    都必须继承它并实现 evaluate() 方法。
                    evaluate() 接收一次完整的 RAG 问答记录，返回各维度的量化分数。
                    支持以下评估模式：
                      - 纯检索评估：只传 query + retrieved_chunks，计算 Hit Rate/MRR 等
                      - 完整 RAG 评估：加上 generated_answer + ground_truth，
                        计算 Faithfulness/Answer Relevancy 等
                    在 EvalRunner 中被调用，结果用于 Dashboard 评估面板展示。
"""
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
