"""
Embedding Factory 单元测试 (tests/unit/test_embedding_factory.py)
================================================================
为什么需要这个文件：
  Embedding 模型在 Ingestion（写入）和 Retrieval（查询）两个阶段都会用到，
  且必须是同一个模型——否则向量空间不一致，相似度计算完全失效。
  测试工厂确保 provider 路由正确，FakeEmbedding 则让其他模块的测试
  不需要真实 API 就能验证"向量维度是否正确、数量是否与 input 一致"。

验收标准（DEV_SPEC B2）：
  - 工厂能根据 provider 路由到正确的类
  - 未知 provider 抛出 ValueError
  - BaseEmbedding 的文本校验逻辑正确
  - FakeEmbedding 的 embed() 返回维度一致的向量
"""
import pytest
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import register_embedding, create_embedding, get_supported_providers
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig


# ── Fake 实现 ─────────────────────────────────────────────────────────────────

@register_embedding("fake_emb")
class FakeEmbedding(BaseEmbedding):
    DIM = 4

    def embed(self, texts, trace=None):
        self._validate_texts(texts)
        return [[0.1 * i for i in range(self.DIM)] for _ in texts]


def _make_settings(provider="fake_emb"):
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider=provider, model="test-emb"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf", top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


# ── 工厂路由测试 ──────────────────────────────────────────────────────────────

def test_factory_creates_correct_instance():
    emb = create_embedding(_make_settings())
    assert isinstance(emb, FakeEmbedding)
    assert emb.model == "test-emb"


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="不支持的 Embedding provider"):
        create_embedding(_make_settings(provider="no_such_provider"))


def test_get_supported_providers_includes_fake():
    assert "fake_emb" in get_supported_providers()


# ── embed 功能测试 ────────────────────────────────────────────────────────────

def test_embed_returns_correct_count():
    emb = FakeEmbedding(model="test")
    texts = ["hello", "world", "foo"]
    vectors = emb.embed(texts)
    assert len(vectors) == 3


def test_embed_returns_correct_dimension():
    emb = FakeEmbedding(model="test")
    vectors = emb.embed(["hello"])
    assert len(vectors[0]) == FakeEmbedding.DIM


# ── 校验测试 ──────────────────────────────────────────────────────────────────

def test_embed_empty_list_raises():
    emb = FakeEmbedding(model="test")
    with pytest.raises(ValueError, match="文本列表不能为空"):
        emb.embed([])


def test_embed_non_string_raises():
    emb = FakeEmbedding(model="test")
    with pytest.raises(ValueError, match="类型错误"):
        emb.embed([123])


def test_embed_none_in_list_raises():
    emb = FakeEmbedding(model="test")
    with pytest.raises(ValueError, match="类型错误"):
        emb.embed([None])
