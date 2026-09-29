"""Splitter Factory 单元测试 (tests/unit/test_splitter_factory.py)"""
import pytest
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import register_splitter, create_splitter, get_supported_methods
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig


# ── Fake 实现 ─────────────────────────────────────────────────────────────────

@register_splitter("fake_split")
class FakeSplitter(BaseSplitter):
    def split_text(self, text, trace=None):
        if not text:
            return []
        # 简单按 chunk_size 截断，方便断言
        return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


def _make_settings(method="fake_split", chunk_size=100, chunk_overlap=20):
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        splitter=SplitterConfig(method=method, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf", top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


# ── 工厂路由测试 ──────────────────────────────────────────────────────────────

def test_factory_creates_correct_instance():
    splitter = create_splitter(_make_settings())
    assert isinstance(splitter, FakeSplitter)


def test_factory_passes_chunk_size():
    splitter = create_splitter(_make_settings(chunk_size=512))
    assert splitter.chunk_size == 512


def test_factory_passes_chunk_overlap():
    splitter = create_splitter(_make_settings(chunk_overlap=50))
    assert splitter.chunk_overlap == 50


def test_factory_unknown_method_raises():
    with pytest.raises(ValueError, match="不支持的 Splitter method"):
        create_splitter(_make_settings(method="nonexistent"))


def test_get_supported_methods_includes_fake():
    assert "fake_split" in get_supported_methods()


# ── split_text 功能测试 ───────────────────────────────────────────────────────

def test_split_text_empty_returns_empty_list():
    splitter = FakeSplitter(chunk_size=100, chunk_overlap=0)
    assert splitter.split_text("") == []


def test_split_text_short_text_returns_one_chunk():
    splitter = FakeSplitter(chunk_size=100, chunk_overlap=0)
    result = splitter.split_text("short text")
    assert len(result) == 1
    assert result[0] == "short text"


def test_split_text_long_text_returns_multiple_chunks():
    splitter = FakeSplitter(chunk_size=10, chunk_overlap=0)
    text = "a" * 35
    result = splitter.split_text(text)
    assert len(result) == 4  # 35 / 10 = 3 full + 1 remainder
