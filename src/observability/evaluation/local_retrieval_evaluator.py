"""
LocalRetrievalEvaluator (src/observability/evaluation/local_retrieval_evaluator.py)
====================================================================================
为什么需要这个文件：
  Ragas 需要 LLM judge（有 API 成本和网络依赖），但"检索对不对"
  可以用纯离线的方式量化——只需把检索结果的 chunk_id/source
  与 golden_test_set 里的 expected_chunk_ids/expected_sources 对比。
  LocalRetrievalEvaluator 实现三个离线指标：
    - hit_rate@k：Top-K 里是否包含至少一个期望 chunk（0 或 1）
    - mrr（Mean Reciprocal Rank）：期望 chunk 在排名中的倒数位置
    - precision@k：Top-K 里期望 chunk 占比

  这是 EvalRunner 的默认 Evaluator，在没有 API Key 的 CI 环境里也能运行。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.evaluator_factory import register_evaluator


@register_evaluator("local")
class LocalRetrievalEvaluator(BaseEvaluator):
    """
    纯离线检索质量评估器：Hit Rate / MRR / Precision@K。

    ground_truth 格式（来自 golden_test_set.json）：
      {
        "expected_chunk_ids": ["chunk_abc_001", "chunk_abc_002"],
        "expected_sources":   ["config_guide.pdf"]
      }

    Usage:
        evaluator = LocalRetrievalEvaluator(k=10)
        metrics = evaluator.evaluate(
            query="如何配置 Azure？",
            retrieved_chunks=[{"id": "chunk_abc_001", "text": "..."}],
            ground_truth={"expected_chunk_ids": ["chunk_abc_001"]},
        )
        # → {"hit_rate": 1.0, "mrr": 1.0, "precision_at_k": 0.1}
    """

    def __init__(self, k: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._k = k

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        计算 Hit Rate / MRR / Precision@K。

        ground_truth 中需包含 expected_chunk_ids 或 expected_sources 中至少一项。
        两者都没有时返回全 0（无法评估）。
        """
        if not ground_truth:
            return {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0}

        expected_ids = set(ground_truth.get("expected_chunk_ids", []))
        expected_sources = set(ground_truth.get("expected_sources", []))

        if not expected_ids and not expected_sources:
            return {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0}

        top_k = retrieved_chunks[: self._k]

        # 判断每个检索结果是否"命中"
        hit_flags = [
            _is_hit(chunk, expected_ids, expected_sources)
            for chunk in top_k
        ]

        hit_rate = 1.0 if any(hit_flags) else 0.0

        # MRR：第一个命中结果的倒数排名
        mrr = 0.0
        for rank, hit in enumerate(hit_flags, start=1):
            if hit:
                mrr = 1.0 / rank
                break

        # Precision@K
        precision_at_k = sum(hit_flags) / len(top_k) if top_k else 0.0

        return {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "precision_at_k": round(precision_at_k, 4),
        }


def _is_hit(
    chunk: Dict[str, Any],
    expected_ids: set,
    expected_sources: set,
) -> bool:
    """判断单个 chunk 是否命中（chunk_id 或 source_path 匹配任意期望值）"""
    chunk_id = chunk.get("id") or chunk.get("chunk_id", "")
    source = chunk.get("metadata", {}).get("source_path", "")
    source_basename = source.replace("\\", "/").rsplit("/", 1)[-1]

    if expected_ids and chunk_id in expected_ids:
        return True
    if expected_sources:
        # 支持完整路径或文件名两种格式匹配
        if source in expected_sources or source_basename in expected_sources:
            return True
    return False
