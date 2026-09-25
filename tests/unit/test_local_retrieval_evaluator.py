"""
LocalRetrievalEvaluator 单元测试 (tests/unit/test_local_retrieval_evaluator.py)
================================================================================
验收标准：
  - hit_rate=1 当 expected chunk_id 在 top-k 中
  - hit_rate=0 当 expected chunk_id 不在 top-k 中
  - mrr 按第一个命中的倒数排名计算
  - precision_at_k = 命中数 / top_k
  - ground_truth 为 None 或空时返回全 0
  - expected_sources 匹配（完整路径 / 文件名两种格式）
  - 注册到工厂 create_evaluator("local") 可用
"""
import pytest
from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator


def _chunk(chunk_id: str, source: str = "", score: float = 0.9) -> dict:
    return {
        "id": chunk_id,
        "text": "content",
        "score": score,
        "metadata": {"source_path": source},
    }


class TestHitRate:
    def test_hit_when_chunk_id_in_results(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1"), _chunk("c2"), _chunk("c3")],
            ground_truth={"expected_chunk_ids": ["c2"]},
        )
        assert result["hit_rate"] == 1.0

    def test_miss_when_chunk_id_not_in_results(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1"), _chunk("c2")],
            ground_truth={"expected_chunk_ids": ["c99"]},
        )
        assert result["hit_rate"] == 0.0

    def test_hit_when_source_matches_full_path(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1", source="data/docs/guide.pdf")],
            ground_truth={"expected_sources": ["data/docs/guide.pdf"]},
        )
        assert result["hit_rate"] == 1.0

    def test_hit_when_source_matches_basename(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1", source="data/docs/guide.pdf")],
            ground_truth={"expected_sources": ["guide.pdf"]},
        )
        assert result["hit_rate"] == 1.0

    def test_miss_when_source_not_in_results(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1", source="other.pdf")],
            ground_truth={"expected_sources": ["guide.pdf"]},
        )
        assert result["hit_rate"] == 0.0

    def test_respects_k_cutoff(self):
        """k=2 时只看前 2 个结果"""
        ev = LocalRetrievalEvaluator(k=2)
        # 期望 chunk 在位置 3（超出 k）
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1"), _chunk("c2"), _chunk("target")],
            ground_truth={"expected_chunk_ids": ["target"]},
        )
        assert result["hit_rate"] == 0.0


class TestMRR:
    def test_mrr_first_position(self):
        ev = LocalRetrievalEvaluator(k=10)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("target"), _chunk("c2")],
            ground_truth={"expected_chunk_ids": ["target"]},
        )
        assert result["mrr"] == pytest.approx(1.0)

    def test_mrr_second_position(self):
        ev = LocalRetrievalEvaluator(k=10)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1"), _chunk("target")],
            ground_truth={"expected_chunk_ids": ["target"]},
        )
        assert result["mrr"] == pytest.approx(0.5)

    def test_mrr_fifth_position(self):
        ev = LocalRetrievalEvaluator(k=10)
        chunks = [_chunk(f"c{i}") for i in range(4)] + [_chunk("target")]
        result = ev.evaluate(
            query="q",
            retrieved_chunks=chunks,
            ground_truth={"expected_chunk_ids": ["target"]},
        )
        assert result["mrr"] == pytest.approx(0.2)

    def test_mrr_zero_when_no_hit(self):
        ev = LocalRetrievalEvaluator(k=10)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1")],
            ground_truth={"expected_chunk_ids": ["missing"]},
        )
        assert result["mrr"] == 0.0


class TestPrecisionAtK:
    def test_precision_two_hits_out_of_four(self):
        ev = LocalRetrievalEvaluator(k=4)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")],
            ground_truth={"expected_chunk_ids": ["a", "c"]},
        )
        assert result["precision_at_k"] == pytest.approx(0.5)

    def test_precision_all_hit(self):
        ev = LocalRetrievalEvaluator(k=3)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("a"), _chunk("b"), _chunk("c")],
            ground_truth={"expected_chunk_ids": ["a", "b", "c"]},
        )
        assert result["precision_at_k"] == pytest.approx(1.0)

    def test_precision_zero_hits(self):
        ev = LocalRetrievalEvaluator(k=3)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("a"), _chunk("b")],
            ground_truth={"expected_chunk_ids": ["x", "y"]},
        )
        assert result["precision_at_k"] == 0.0


class TestEdgeCases:
    def test_empty_ground_truth_returns_zeros(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(query="q", retrieved_chunks=[_chunk("c1")])
        assert result == {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0}

    def test_none_ground_truth_returns_zeros(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q", retrieved_chunks=[_chunk("c1")], ground_truth=None
        )
        assert result == {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0}

    def test_empty_chunks_returns_zeros(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[],
            ground_truth={"expected_chunk_ids": ["c1"]},
        )
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0

    def test_registered_in_factory(self):
        from src.libs.evaluator.evaluator_factory import _EVALUATOR_REGISTRY
        import src.observability.evaluation.local_retrieval_evaluator  # noqa: F401
        assert "local" in _EVALUATOR_REGISTRY

    def test_required_keys_always_present(self):
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate(
            query="q",
            retrieved_chunks=[_chunk("c1")],
            ground_truth={"expected_chunk_ids": ["c1"]},
        )
        for key in ("hit_rate", "mrr", "precision_at_k"):
            assert key in result
