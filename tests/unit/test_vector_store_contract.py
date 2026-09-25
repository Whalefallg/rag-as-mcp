"""
VectorStore 契约测试 (tests/unit/test_vector_store_contract.py)
===============================================================
为什么需要这个文件：
  向量存储是 RAG 系统的核心数据层，upsert/query/get_by_ids/delete 四个方法
  被多个上层模块依赖（VectorUpserter/DenseRetriever/SparseRetriever）。
  "契约测试"的意义：不是测 ChromaDB 的具体实现细节，
  而是测"所有实现都必须满足的接口约定"——输入什么格式、输出什么格式。
  这样将来换成 Qdrant 或 Pinecone，只要通过契约测试，上层代码零改动。

验收标准（DEV_SPEC B4）：
  - 工厂能根据 backend 路由到正确的类
  - 未知 backend 抛出 ValueError
  - 契约测试：所有实现都遵守 BaseVectorStore 的接口约定（输入输出 shape 正确）
"""
import pytest
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import register_vector_store, create_vector_store, get_supported_backends
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig


# ── Fake 实现 ─────────────────────────────────────────────────────────────────

@register_vector_store("fake_store")
class FakeVectorStore(BaseVectorStore):
    """内存版 VectorStore，用于测试"""

    def __init__(self, persist_path: str, **kwargs):
        super().__init__(persist_path=persist_path)
        self._data: dict = {}  # id -> record

    def upsert(self, records, trace=None):
        for r in records:
            self._data[r["id"]] = r

    def query(self, vector, top_k, filters=None, trace=None):
        results = list(self._data.values())[:top_k]
        return [{"id": r["id"], "text": r.get("text", ""), "metadata": r.get("metadata", {}), "score": 0.9} for r in results]

    def get_by_ids(self, ids):
        return [self._data[i] for i in ids if i in self._data]

    def delete_by_metadata(self, filter):
        before = len(self._data)
        to_delete = [k for k, v in self._data.items()
                     if all(v.get("metadata", {}).get(fk) == fv for fk, fv in filter.items())]
        for k in to_delete:
            del self._data[k]
        return before - len(self._data)


def _make_settings(backend="fake_store"):
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend=backend, persist_path="./data/db/test"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf", top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


# ── 工厂路由测试 ──────────────────────────────────────────────────────────────

def test_factory_creates_correct_instance():
    store = create_vector_store(_make_settings())
    assert isinstance(store, FakeVectorStore)


def test_factory_passes_persist_path():
    store = create_vector_store(_make_settings())
    assert store.persist_path == "./data/db/test"


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError, match="不支持的 VectorStore backend"):
        create_vector_store(_make_settings(backend="no_such_backend"))


def test_get_supported_backends_includes_fake():
    assert "fake_store" in get_supported_backends()


# ── 契约测试（输入/输出 shape 验证）──────────────────────────────────────────

@pytest.fixture
def store():
    return FakeVectorStore(persist_path="./tmp")


def test_upsert_and_query_roundtrip(store):
    """upsert 后能通过 query 查出数据"""
    records = [
        {"id": "c1", "text": "hello world", "metadata": {"source": "doc.pdf"}, "dense_vector": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "foo bar", "metadata": {"source": "doc.pdf"}, "dense_vector": [0.4, 0.5, 0.6]},
    ]
    store.upsert(records)
    results = store.query(vector=[0.1, 0.2, 0.3], top_k=2)
    assert len(results) == 2
    assert all("id" in r and "text" in r and "score" in r for r in results)


def test_get_by_ids_returns_correct_records(store):
    """get_by_ids 返回对应记录"""
    store.upsert([{"id": "x1", "text": "test", "metadata": {}, "dense_vector": []}])
    results = store.get_by_ids(["x1"])
    assert len(results) == 1
    assert results[0]["id"] == "x1"


def test_get_by_ids_missing_id_skipped(store):
    """不存在的 id 不报错，直接跳过"""
    store.upsert([{"id": "y1", "text": "test", "metadata": {}, "dense_vector": []}])
    results = store.get_by_ids(["y1", "nonexistent"])
    assert len(results) == 1


def test_delete_by_metadata_returns_count(store):
    """delete_by_metadata 返回删除数量"""
    store.upsert([
        {"id": "d1", "text": "a", "metadata": {"source": "old.pdf"}, "dense_vector": []},
        {"id": "d2", "text": "b", "metadata": {"source": "old.pdf"}, "dense_vector": []},
        {"id": "d3", "text": "c", "metadata": {"source": "keep.pdf"}, "dense_vector": []},
    ])
    deleted = store.delete_by_metadata({"source": "old.pdf"})
    assert deleted == 2
    assert len(store.get_by_ids(["d1", "d2", "d3"])) == 1


def test_upsert_idempotent(store):
    """同一 id 重复 upsert 不产生重复记录"""
    record = {"id": "dup", "text": "v1", "metadata": {}, "dense_vector": []}
    store.upsert([record])
    record["text"] = "v2"
    store.upsert([record])
    results = store.get_by_ids(["dup"])
    assert len(results) == 1
    assert results[0]["text"] == "v2"
