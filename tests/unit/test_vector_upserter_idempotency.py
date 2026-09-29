"""VectorUpserter 幂等性测试 (tests/unit/test_vector_upserter_idempotency.py)"""
import pytest
from src.ingestion.storage.vector_upserter import _stable_chunk_id, VectorUpserter
from src.core.types import Chunk
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.libs.vector_store.vector_store_factory import register_vector_store
from src.libs.vector_store.base_vector_store import BaseVectorStore


# ── FakeVectorStore（记录 upsert 调用）────────────────────────────────────────

@register_vector_store("fake_upsert")
class FakeUpsertStore(BaseVectorStore):
    def __init__(self, persist_path, **kwargs):
        super().__init__(persist_path)
        self.upserted: list = []

    def upsert(self, records, trace=None):
        self.upserted.extend(records)

    def query(self, vector, top_k, filters=None, trace=None): return []
    def get_by_ids(self, ids): return []
    def delete_by_metadata(self, filter): return 0


def _make_settings():
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="fake_upsert", persist_path="./tmp"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def _make_encoded_chunk(idx: int, text: str = "hello world") -> Chunk:
    import hashlib
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    return Chunk(
        id=f"c{idx}", doc_id="doc1", text=text, index=idx,
        metadata={
            "source_path": "/test.pdf",
            "content_hash": content_hash,
            "dense_vector": [0.1, 0.2, 0.3],
        },
    )


# ── stable_chunk_id 函数测试 ──────────────────────────────────────────────────

def test_stable_id_same_input_same_output():
    id1 = _stable_chunk_id("/doc.pdf", 0, "abc12345")
    id2 = _stable_chunk_id("/doc.pdf", 0, "abc12345")
    assert id1 == id2


def test_stable_id_different_content_different_id():
    id1 = _stable_chunk_id("/doc.pdf", 0, "abc12345")
    id2 = _stable_chunk_id("/doc.pdf", 0, "xyz99999")
    assert id1 != id2


def test_stable_id_different_index_different_id():
    id1 = _stable_chunk_id("/doc.pdf", 0, "abc12345")
    id2 = _stable_chunk_id("/doc.pdf", 1, "abc12345")
    assert id1 != id2


def test_stable_id_is_16_hex_chars():
    sid = _stable_chunk_id("/doc.pdf", 0, "abc12345")
    assert len(sid) == 16
    assert all(c in "0123456789abcdef" for c in sid)


# ── VectorUpserter 测试 ───────────────────────────────────────────────────────

def test_upsert_returns_correct_count():
    upserter = VectorUpserter(_make_settings())
    chunks = [_make_encoded_chunk(i) for i in range(3)]
    count = upserter.upsert(chunks)
    assert count == 3


def test_upsert_empty_returns_zero():
    upserter = VectorUpserter(_make_settings())
    assert upserter.upsert([]) == 0


def test_upsert_same_chunk_twice_same_id():
    """同一内容两次 upsert 产生相同的 stable_id（幂等性核心）"""
    upserter = VectorUpserter(_make_settings())
    chunk = _make_encoded_chunk(0, "stable content")
    upserter.upsert([chunk])
    upserter.upsert([chunk])

    store = upserter._store
    ids1 = [r["id"] for r in store.upserted[:1]]
    ids2 = [r["id"] for r in store.upserted[1:]]
    assert ids1 == ids2


def test_upsert_preserves_text():
    upserter = VectorUpserter(_make_settings())
    chunk = _make_encoded_chunk(0, "unique content text")
    upserter.upsert([chunk])
    record = upserter._store.upserted[0]
    assert record["text"] == "unique content text"


def test_upsert_metadata_excludes_vectors():
    """dense_vector 不应出现在 metadata 里（单独存）"""
    upserter = VectorUpserter(_make_settings())
    chunk = _make_encoded_chunk(0)
    upserter.upsert([chunk])
    record = upserter._store.upserted[0]
    assert "dense_vector" not in record["metadata"]
    assert "dense_vector" in record  # 但在 record 顶层
