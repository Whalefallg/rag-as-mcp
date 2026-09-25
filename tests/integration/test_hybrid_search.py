"""
HybridSearch 集成测试 (tests/integration/test_hybrid_search.py)
================================================================
验收标准（DEV_SPEC D5）：
  - 对 fixtures 数据，能返回 Top-K（包含 chunk 文本与 metadata）
  - 支持 filters 参数过滤
  - Dense 路径失败时能降级到 Sparse 单路结果
  - Sparse 路径失败时能降级到 Dense 单路结果
  - 两路都失败时返回空列表
  - 结果数量不超过 top_k
"""
import pytest
from src.core.types import RetrievalResult
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFusion
from src.core.query_engine.hybrid_search import HybridSearch


def _make_settings(top_k_final: int = 5):
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./tmp"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=10, top_k_sparse=10, top_k_final=top_k_final),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


# ── Fake 组件 ─────────────────────────────────────────────────────────────────

def _make_result(cid: str, score: float = 0.9, text: str = "test", meta: dict = None, source: str = "dense") -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=score, text=text,
                           metadata=meta or {"source_path": "/test.pdf"}, source=source)


class FakeDense:
    def __init__(self, results=None, fail=False):
        self._results = results or [_make_result("c1", 0.9), _make_result("c2", 0.8)]
        self._fail = fail

    def retrieve(self, query, top_k=20, filters=None, collection="default", trace=None):
        if self._fail:
            raise RuntimeError("dense failed")
        return self._results[:top_k]


class FakeSparse:
    def __init__(self, results=None, fail=False):
        self._results = results or [_make_result("c1", 2.0, source="sparse"),
                                     _make_result("c3", 1.5, source="sparse")]
        self._fail = fail

    def retrieve(self, keywords, top_k=20, trace=None):
        if self._fail:
            raise RuntimeError("sparse failed")
        return self._results[:top_k]


# ── 基本功能 ──────────────────────────────────────────────────────────────────

def test_search_returns_results():
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(), FakeDense(), FakeSparse(), RRFusion())
    results = hs.search("machine learning", top_k=5)
    assert len(results) > 0


def test_search_result_type():
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(), FakeDense(), FakeSparse(), RRFusion())
    results = hs.search("test query")
    for r in results:
        assert isinstance(r, RetrievalResult)


def test_search_top_k_respected():
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(), FakeDense(), FakeSparse(), RRFusion())
    results = hs.search("test", top_k=2)
    assert len(results) <= 2


def test_search_chunk_appearing_in_both_lists_is_ranked_higher():
    """c1 同时出现在 Dense 和 Sparse 中，应排在最前面"""
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(), FakeDense(), FakeSparse(), RRFusion())
    results = hs.search("test", top_k=5)
    ids = [r.chunk_id for r in results]
    assert ids[0] == "c1"


# ── 降级行为 ──────────────────────────────────────────────────────────────────

def test_search_dense_fails_falls_back_to_sparse():
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(),
                      FakeDense(fail=True), FakeSparse(), RRFusion())
    results = hs.search("test")
    assert len(results) > 0


def test_search_sparse_fails_falls_back_to_dense():
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(),
                      FakeDense(), FakeSparse(fail=True), RRFusion())
    results = hs.search("test")
    assert len(results) > 0


def test_search_both_fail_returns_empty():
    settings = _make_settings()
    hs = HybridSearch(settings, QueryProcessor(),
                      FakeDense(fail=True), FakeSparse(fail=True), RRFusion())
    results = hs.search("test")
    assert results == []


# ── filters 过滤 ──────────────────────────────────────────────────────────────

def test_search_metadata_filter_applied():
    """collection=pdf_only 过滤后，不匹配的结果被剔除"""
    settings = _make_settings()
    dense = FakeDense(results=[
        _make_result("c1", meta={"source_path": "/a.pdf", "collection": "pdf_only"}),
        _make_result("c2", meta={"source_path": "/b.pdf", "collection": "other"}),
    ])
    sparse = FakeSparse(results=[])
    hs = HybridSearch(settings, QueryProcessor(), dense, sparse, RRFusion())
    results = hs.search("test", filters={"collection": "pdf_only"}, top_k=5)
    for r in results:
        assert r.metadata.get("collection") == "pdf_only"


def test_search_empty_results_with_strict_filter():
    settings = _make_settings()
    dense = FakeDense(results=[
        _make_result("c1", meta={"source_path": "/a.pdf", "collection": "A"}),
    ])
    sparse = FakeSparse(results=[])
    hs = HybridSearch(settings, QueryProcessor(), dense, sparse, RRFusion())
    results = hs.search("test", filters={"collection": "nonexistent"}, top_k=5)
    assert results == []
