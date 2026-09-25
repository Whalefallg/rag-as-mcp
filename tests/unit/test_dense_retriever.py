"""
DenseRetriever 单元测试 (tests/unit/test_dense_retriever.py)
============================================================
验收标准（DEV_SPEC D2）：
  - 对输入 query 能生成 embedding 并调用 VectorStore 检索
  - 返回结果包含 chunk_id、score、text、metadata
  - 空 query 返回空列表
  - mock EmbeddingClient 和 VectorStore 时能正确编排调用
  - source 字段标记为 "dense"
"""
import pytest
from src.core.types import RetrievalResult
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.core.query_engine.dense_retriever import DenseRetriever


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


# ── Fake Embedding ────────────────────────────────────────────────────────────

class FakeEmbedding:
    """固定返回 [0.1, 0.2, 0.3] 向量，避免真实 API 调用"""
    def embed(self, texts, trace=None):
        return [[0.1, 0.2, 0.3] for _ in texts]


# ── Fake VectorStore ──────────────────────────────────────────────────────────

class FakeVectorStore:
    """预设召回结果的 VectorStore mock"""
    def __init__(self, results=None):
        self.last_vector = None
        self.last_top_k = None
        self.last_filters = None
        self._results = results if results is not None else [
            {"id": "c001", "score": 0.95, "text": "Azure setup guide", "metadata": {"source_path": "/a.pdf"}},
            {"id": "c002", "score": 0.87, "text": "OpenAI configuration", "metadata": {"source_path": "/b.pdf"}},
        ]

    def query(self, vector, top_k, filters=None, trace=None):
        self.last_vector = vector
        self.last_top_k = top_k
        self.last_filters = filters
        return self._results[:top_k]

    def upsert(self, records, trace=None): pass
    def get_by_ids(self, ids): return []
    def delete_by_metadata(self, f): return 0


# ── 测试 ──────────────────────────────────────────────────────────────────────

def test_retrieve_returns_list():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore())
    results = retriever.retrieve("how to configure Azure")
    assert isinstance(results, list)
    assert len(results) == 2


def test_retrieve_result_fields():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore())
    results = retriever.retrieve("test query")
    r = results[0]
    assert r.chunk_id == "c001"
    assert r.score == pytest.approx(0.95)
    assert r.text == "Azure setup guide"
    assert "source_path" in r.metadata


def test_retrieve_source_marked_dense():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore())
    results = retriever.retrieve("query")
    for r in results:
        assert r.source == "dense"


def test_retrieve_empty_query_returns_empty():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore())
    results = retriever.retrieve("")
    assert results == []


def test_retrieve_whitespace_query_returns_empty():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore())
    results = retriever.retrieve("   ")
    assert results == []


def test_retrieve_passes_top_k_to_store():
    store = FakeVectorStore()
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), store)
    retriever.retrieve("query", top_k=5)
    assert store.last_top_k == 5


def test_retrieve_embedding_called_with_query():
    class TrackingEmbedding:
        called_with = None
        def embed(self, texts, trace=None):
            TrackingEmbedding.called_with = texts
            return [[0.0, 0.0]]

    retriever = DenseRetriever(_make_settings(), TrackingEmbedding(), FakeVectorStore([]))
    retriever.retrieve("my search query")
    assert TrackingEmbedding.called_with == ["my search query"]


def test_retrieve_returns_empty_when_store_empty():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore(results=[]))
    results = retriever.retrieve("query")
    assert results == []


def test_retrieve_result_is_retrieval_result_type():
    retriever = DenseRetriever(_make_settings(), FakeEmbedding(), FakeVectorStore())
    results = retriever.retrieve("query")
    for r in results:
        assert isinstance(r, RetrievalResult)
