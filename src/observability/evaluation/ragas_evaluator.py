"""
RagasEvaluator (src/observability/evaluation/ragas_evaluator.py)
=================================================================
为什么需要这个文件：
  Ragas 是目前最成熟的 RAG 评估框架，提供：
    - Faithfulness：答案是否忠实于检索内容（不捏造）
    - Answer Relevancy：答案与问题的相关程度
    - Context Precision：检索内容中有用部分的比例
  这三个指标覆盖了 RAG 系统最核心的质量维度。

  优雅降级策略（离线优先）：
    - Ragas 未安装 → 抛 ImportError，提示安装命令
    - LLM/Embedding 不可用 → 回退到 LocalRetrievalEvaluator，
      只计算 Hit Rate / MRR（无需 LLM）
    - 这样在没有 API Key 的环境里，评估系统依然可以运行

  关于 Mock LLM 测试：
    Ragas >= 0.2 支持传入 langchain LLM 对象作为 judge_llm，
    测试时注入 FakeLLM 即可在无网络环境下验证逻辑。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.evaluator_factory import register_evaluator


@register_evaluator("ragas")
class RagasEvaluator(BaseEvaluator):
    """
    封装 Ragas 框架的 RAG 质量评估器。

    支持指标：faithfulness / answer_relevancy / context_precision。
    Ragas 未安装时抛出带安装说明的 ImportError。

    Usage:
        evaluator = RagasEvaluator(llm=my_llm, embeddings=my_embeddings)
        metrics = evaluator.evaluate(
            query="如何配置 Azure？",
            retrieved_chunks=[{"id": "c1", "text": "..."}],
            generated_answer="答案...",
            ground_truth={"answer": "标准答案"},
        )
        # → {"faithfulness": 0.9, "answer_relevancy": 0.85, ...}
    """

    def __init__(self, llm=None, embeddings=None, **kwargs) -> None:
        """
        Args:
            llm:        Ragas judge LLM（langchain BaseLLM 实例）。
                        None 时 Ragas 使用默认 OpenAI（需配置 OPENAI_API_KEY）。
            embeddings: Ragas 用的 Embedding 模型。None 时使用默认。
            **kwargs:   透传给 BaseEvaluator。
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._embeddings = embeddings
        self._check_ragas()

    @staticmethod
    def _check_ragas() -> None:
        """验证 Ragas 是否已安装，未安装时给出明确提示"""
        try:
            import ragas  # noqa: F401
        except ImportError:
            raise ImportError(
                "Ragas 未安装。请运行：\n"
                "  pip install ragas\n"
                "或在 requirements.txt 中取消 ragas>=0.1.0 的注释。"
            )

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        调用 Ragas 评估一次 RAG 问答。

        Args:
            query:            用户原始问题。
            retrieved_chunks: 检索结果，每条含 id/text/score。
            generated_answer: LLM 生成的回答（可选）。
            ground_truth:     标准答案 dict，{"answer": "...", "chunk_ids": [...]}（可选）。
        Returns:
            指标字典，例如 {"faithfulness": 0.9, "answer_relevancy": 0.85,
                           "context_precision": 0.78}。
        Raises:
            ImportError: ragas 未安装。
            RuntimeError: Ragas 评估执行失败（LLM 调用异常等）。
        """
        try:
            return self._run_ragas(query, retrieved_chunks, generated_answer, ground_truth)
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Ragas 评估失败：{exc}") from exc

    def _run_ragas(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: Optional[str],
        ground_truth: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """实际调用 Ragas 的内部方法，便于测试时 patch"""
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        contexts = [c.get("text", "") for c in retrieved_chunks]
        answer = generated_answer or ""
        gt_answer = (ground_truth or {}).get("answer", "")

        # Ragas Dataset 格式
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [gt_answer],
        }
        dataset = Dataset.from_dict(data)

        # 选择度量指标（无 answer 时只评 context_precision）
        metrics = [context_precision]
        if answer:
            metrics = [faithfulness, answer_relevancy, context_precision]

        kwargs: Dict[str, Any] = {}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        if self._embeddings is not None:
            kwargs["embeddings"] = self._embeddings

        result = ragas_evaluate(dataset=dataset, metrics=metrics, **kwargs)
        # result 是 dict-like，取 mean 值
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
