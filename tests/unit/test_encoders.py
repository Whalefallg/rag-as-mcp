"""
DenseEncoder / SparseEncoder / BatchProcessor 单元测试
(tests/unit/test_encoders.py)
=====================================================
为什么需要这个文件：
  Ingestion Pipeline 在把 chunk 存入向量库之前，需要为每个 chunk 生成两种向量：
    - DenseEncoder：调用 Embedding 模型，生成语义向量（用于 Dense Retrieval）
    - SparseEncoder：基于词频统计，生成稀疏权重字典（用于 BM25 建索引）
  BatchProcessor 负责把 chunk 列表分批交给上面两个 Encoder，
  避免一次性把几千个 chunk 发给 API 导致超时或 OOM。
  这里的测试确保三个组件的输入输出尺寸正确、批处理顺序稳定。

验收标准（DEV_SPEC C8/C9/C10）：
  - DenseEncoder 输出向量数量与 chunks 数量一致
  - SparseEncoder 空文本返回空字典
  - BatchProcessor batch_size=2 对 5 chunks 分 3 批，顺序稳定
"""
import pytest
from src.core.types import Chunk
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.libs.embedding.embedding_factory import register_embedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.vector_store_factory import register_vector_store
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder, _tokenize
from src.ingestion.embedding.batch_processor import BatchProcessor


# ── Fake Embedding（固定维度，避免真实 API 调用）────────────────────────────

@register_embedding("fake_enc")
class FakeEmbeddingForEncoder(BaseEmbedding):
    DIM = 8
    def embed(self, texts, trace=None):
        self._validate_texts(texts)
        return [[float(i) for i in range(self.DIM)] for _ in texts]


# ── Fake VectorStore（本文件专属，不与其他测试文件冲突）──────────────────────

@register_vector_store("fake_enc_store")
class FakeEncStore(BaseVectorStore):
    def upsert(self, records, trace=None): pass
    def query(self, vector, top_k, filters=None, trace=None): return []
    def get_by_ids(self, ids): return []
    def delete_by_metadata(self, filter): return 0


def _make_settings():
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="fake_enc", model="fake"),
        vector_store=VectorStoreConfig(backend="fake_enc_store", persist_path="./tmp"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def _make_chunks(n: int) -> list:
    return [
        Chunk(id=f"c{i}", doc_id="doc1", text=f"this is chunk number {i}", index=i, metadata={})
        for i in range(n)
    ]


# ── DenseEncoder ──────────────────────────────────────────────────────────────

def test_dense_encoder_count_equals_input():
    encoder = DenseEncoder(_make_settings())
    chunks = _make_chunks(5)
    result = encoder.encode(chunks)
    assert len(result) == 5


def test_dense_encoder_vector_attached_to_metadata():
    encoder = DenseEncoder(_make_settings())
    result = encoder.encode(_make_chunks(1))
    assert "dense_vector" in result[0].metadata
    assert len(result[0].metadata["dense_vector"]) == FakeEmbeddingForEncoder.DIM


def test_dense_encoder_all_same_dim():
    encoder = DenseEncoder(_make_settings())
    result = encoder.encode(_make_chunks(4))
    dims = [len(c.metadata["dense_vector"]) for c in result]
    assert len(set(dims)) == 1  # 所有维度相同


def test_dense_encoder_empty_returns_empty():
    encoder = DenseEncoder(_make_settings())
    assert encoder.encode([]) == []


def test_dense_encoder_preserves_order():
    encoder = DenseEncoder(_make_settings())
    chunks = _make_chunks(3)
    result = encoder.encode(chunks)
    assert [c.index for c in result] == [0, 1, 2]


# ── SparseEncoder ─────────────────────────────────────────────────────────────

def test_sparse_encoder_returns_dict():
    encoder = SparseEncoder()
    chunks = _make_chunks(2)
    result = encoder.encode(chunks)
    for chunk in result:
        assert isinstance(chunk.metadata["sparse_vector"], dict)


def test_sparse_encoder_empty_text_returns_empty_dict():
    encoder = SparseEncoder()
    chunk = Chunk(id="empty", doc_id="d", text="", index=0, metadata={})
    result = encoder.encode([chunk])
    assert result[0].metadata["sparse_vector"] == {}


def test_sparse_encoder_term_weights_sum_to_one():
    encoder = SparseEncoder()
    chunk = Chunk(id="c1", doc_id="d", text="hello world hello", index=0, metadata={})
    result = encoder.encode([chunk])
    weights = result[0].metadata["sparse_vector"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_sparse_encoder_known_term_frequency():
    """hello 出现 2/3，world 出现 1/3"""
    encoder = SparseEncoder()
    chunk = Chunk(id="c1", doc_id="d", text="hello world hello", index=0, metadata={})
    result = encoder.encode([chunk])
    weights = result[0].metadata["sparse_vector"]
    assert abs(weights["hello"] - 2 / 3) < 1e-6
    assert abs(weights["world"] - 1 / 3) < 1e-6


def test_tokenize_lowercases():
    tokens = _tokenize("Hello WORLD")
    assert "hello" in tokens
    assert "world" in tokens


# ── BatchProcessor ────────────────────────────────────────────────────────────

def test_batch_processor_5_chunks_3_batches():
    """batch_size=2 时 5 chunks 分 3 批（2+2+1）"""
    batch_calls = []
    original_encode = DenseEncoder.encode

    def track_encode(self, chunks, trace=None):
        batch_calls.append(len(chunks))
        return original_encode(self, chunks, trace)

    import unittest.mock as mock
    settings = _make_settings()
    processor = BatchProcessor(settings, batch_size=2)

    with mock.patch.object(DenseEncoder, "encode", track_encode):
        result = processor.encode_all(_make_chunks(5))

    assert len(result) == 5
    assert batch_calls == [2, 2, 1]  # 3 批


def test_batch_processor_preserves_order():
    settings = _make_settings()
    processor = BatchProcessor(settings, batch_size=3)
    chunks = _make_chunks(7)
    result = processor.encode_all(chunks)
    assert [c.index for c in result] == list(range(7))


def test_batch_processor_empty_returns_empty():
    settings = _make_settings()
    processor = BatchProcessor(settings)
    assert processor.encode_all([]) == []
