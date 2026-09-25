"""
SparseRetriever 单元测试 (tests/unit/test_sparse_retriever.py)
==============================================================
验收标准（DEV_SPEC D3）：
  - 对已构建索引的语料，关键词检索命中预期 chunk_id
  - 返回结果包含完整 text 和 metadata
  - BM25 无命中时返回空列表
  - source 字段标记为 "sparse"
  - keywords 为空时返回空列表
"""
import pytest
from src.core.types import RetrievalResult
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.core.types import Chunk


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_indexed_chunk(cid: str, text: str, idx: int = 0) -> Chunk:
    """创建已做 SparseEncoder 的 Chunk（metadata 含 sparse_vector）"""
    tokens = text.lower().split()
    freq: dict = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    sparse = {t: c / total for t, c in freq.items()}
    return Chunk(
        id=cid, doc_id="doc1", text=text, index=idx,
        metadata={"sparse_vector": sparse, "source_path": "/test.pdf"},
    )


class FakeVectorStore:
    """用字典存记录，模拟 get_by_ids 行为"""
    def __init__(self, records: list):
        self._store = {r["id"]: r for r in records}

    def get_by_ids(self, ids):
        return [self._store[i] for i in ids if i in self._store]

    def query(self, vector, top_k, filters=None, trace=None): return []
    def upsert(self, records, trace=None): pass
    def delete_by_metadata(self, f): return 0


# ── 测试 ──────────────────────────────────────────────────────────────────────

def test_retrieve_hits_expected_chunk(tmp_path):
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = [
        _make_indexed_chunk("c1", "machine learning neural network training", 0),
        _make_indexed_chunk("c2", "python programming language syntax", 1),
        _make_indexed_chunk("c3", "database query optimization index", 2),
    ]
    indexer.build(chunks)

    store = FakeVectorStore([
        {"id": "c1", "text": "machine learning neural network training",
         "metadata": {"source_path": "/a.pdf"}},
        {"id": "c2", "text": "python programming language syntax",
         "metadata": {"source_path": "/b.pdf"}},
        {"id": "c3", "text": "database query optimization index",
         "metadata": {"source_path": "/c.pdf"}},
    ])

    retriever = SparseRetriever(_make_settings(), bm25_indexer=indexer, vector_store=store)
    results = retriever.retrieve(["machine", "learning"], top_k=3)

    assert len(results) > 0
    ids = [r.chunk_id for r in results]
    assert "c1" in ids


def test_retrieve_result_has_text_and_metadata(tmp_path):
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = [_make_indexed_chunk("c1", "hello world test", 0)]
    indexer.build(chunks)

    store = FakeVectorStore([
        {"id": "c1", "text": "hello world test", "metadata": {"source_path": "/x.pdf"}}
    ])
    retriever = SparseRetriever(_make_settings(), bm25_indexer=indexer, vector_store=store)
    results = retriever.retrieve(["hello"])

    assert results[0].text == "hello world test"
    assert "source_path" in results[0].metadata


def test_retrieve_source_marked_sparse(tmp_path):
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = [_make_indexed_chunk("c1", "hello world", 0)]
    indexer.build(chunks)

    store = FakeVectorStore([{"id": "c1", "text": "hello world", "metadata": {}}])
    retriever = SparseRetriever(_make_settings(), bm25_indexer=indexer, vector_store=store)
    results = retriever.retrieve(["hello"])

    assert all(r.source == "sparse" for r in results)


def test_retrieve_empty_keywords_returns_empty(tmp_path):
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    indexer.build([_make_indexed_chunk("c1", "hello", 0)])
    store = FakeVectorStore([{"id": "c1", "text": "hello", "metadata": {}}])
    retriever = SparseRetriever(_make_settings(), bm25_indexer=indexer, vector_store=store)
    assert retriever.retrieve([]) == []


def test_retrieve_unknown_keywords_returns_empty(tmp_path):
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    indexer.build([_make_indexed_chunk("c1", "hello world", 0)])
    store = FakeVectorStore([{"id": "c1", "text": "hello world", "metadata": {}}])
    retriever = SparseRetriever(_make_settings(), bm25_indexer=indexer, vector_store=store)
    results = retriever.retrieve(["completely_unknown_xyzabc"])
    assert results == []


def test_retrieve_result_type(tmp_path):
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = [_make_indexed_chunk("c1", "python machine learning", 0)]
    indexer.build(chunks)
    store = FakeVectorStore([{"id": "c1", "text": "python machine learning", "metadata": {}}])
    retriever = SparseRetriever(_make_settings(), bm25_indexer=indexer, vector_store=store)
    results = retriever.retrieve(["python"])
    for r in results:
        assert isinstance(r, RetrievalResult)
