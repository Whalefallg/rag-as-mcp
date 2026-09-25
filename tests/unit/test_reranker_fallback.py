"""
CoreReranker fallback 测试 (tests/unit/test_reranker_fallback.py)
=================================================================
验收标准（DEV_SPEC D6）：
  - 正常情况：后端返回有效排序，结果重新排列
  - 失败降级：后端抛出异常时，返回原始顺序并标记 fallback=True
  - 空候选列表返回空
  - top_k 截断正确
  - 精排后 source 标记为 "reranked"
"""
import pytest
from src.core.types import RetrievalResult
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.core.query_engine.reranker import CoreReranker


def _make_settings():
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./tmp"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def _make_candidates(n: int = 3) -> list:
    return [
        RetrievalResult(
            chunk_id=f"c{i}", score=1.0 / (i + 1),
            text=f"text of chunk {i}",
            metadata={"source_path": "/test.pdf"},
            source="fused",
        )
        for i in range(n)
    ]


# ── Fake Reranker 后端 ────────────────────────────────────────────────────────

class FakeReranker:
    """模拟 libs.reranker：把顺序反转（最后一个排第一）"""
    def rerank(self, query, texts, top_k=None):
        indices = list(range(len(texts) - 1, -1, -1))
        return indices[:top_k] if top_k else indices


class FailingReranker:
    """模拟后端异常"""
    def rerank(self, query, texts, top_k=None):
        raise RuntimeError("reranker backend timeout")


# ── 正常路径 ──────────────────────────────────────────────────────────────────

def test_rerank_returns_list():
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    candidates = _make_candidates(3)
    results = reranker.rerank("test query", candidates)
    assert isinstance(results, list)
    assert len(results) == 3


def test_rerank_reverses_order():
    """FakeReranker 把顺序反转，c2 应排第一"""
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    candidates = _make_candidates(3)
    results = reranker.rerank("test query", candidates)
    assert results[0].chunk_id == "c2"  # 原来的最后一个


def test_rerank_source_marked_reranked():
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    results = reranker.rerank("query", _make_candidates(2))
    assert all(r.source == "reranked" for r in results)


def test_rerank_rank_in_metadata():
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    results = reranker.rerank("query", _make_candidates(3))
    for i, r in enumerate(results):
        assert r.metadata["rerank_rank"] == i + 1


def test_rerank_top_k_truncation():
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    results = reranker.rerank("query", _make_candidates(5), top_k=2)
    assert len(results) == 2


def test_rerank_empty_candidates_returns_empty():
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    assert reranker.rerank("query", []) == []


# ── 降级行为 ──────────────────────────────────────────────────────────────────

def test_rerank_fallback_on_exception():
    reranker = CoreReranker(_make_settings(), reranker=FailingReranker())
    candidates = _make_candidates(3)
    results = reranker.rerank("query", candidates)
    # 降级：返回原始顺序，长度不变
    assert len(results) == 3
    assert results[0].chunk_id == "c0"


def test_rerank_fallback_marks_metadata():
    reranker = CoreReranker(_make_settings(), reranker=FailingReranker())
    results = reranker.rerank("query", _make_candidates(2))
    for r in results:
        assert r.metadata.get("fallback") is True
        assert "fallback_reason" in r.metadata


def test_rerank_fallback_top_k_respected():
    reranker = CoreReranker(_make_settings(), reranker=FailingReranker())
    results = reranker.rerank("query", _make_candidates(5), top_k=2)
    assert len(results) == 2


def test_rerank_scores_are_rank_based():
    """精排后 score = 1 / rank，第 1 名得分最高"""
    reranker = CoreReranker(_make_settings(), reranker=FakeReranker())
    results = reranker.rerank("query", _make_candidates(3))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
