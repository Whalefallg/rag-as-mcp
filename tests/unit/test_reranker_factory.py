"""
Reranker Factory 单元测试 (tests/unit/test_reranker_factory.py)
===============================================================
为什么需要这个文件：
  Reranker 是 RAG 系统中成本最高的组件之一（Cross-Encoder 每次都要过模型）。
  "backend=none 时跳过精排"是必须保证的行为——测试 NoneReranker 的 passthrough 语义，
  确保关闭精排不会影响系统其他部分的正确性。
  降级测试则验证"精排失败时系统仍能返回 RRF 融合结果"，不让精排成为单点故障。

验收标准（DEV_SPEC B5）：
  - backend=none 时返回 NoneReranker，原样返回 candidates
  - 未知 backend 抛出 ValueError
  - NoneReranker 保持原始顺序不变
  - CrossEncoderReranker 和 LLMReranker 降级回退逻辑正确
"""
import pytest
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import create_reranker, get_supported_backends
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.llm_reranker import LLMReranker
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig


def _make_settings(backend="none", model=None):
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf", top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend=backend, model=model),
        raw_config={},
    )


SAMPLE_CANDIDATES = [
    {"id": "c1", "text": "first chunk", "score": 0.9},
    {"id": "c2", "text": "second chunk", "score": 0.7},
    {"id": "c3", "text": "third chunk", "score": 0.5},
]


# ── NoneReranker 测试 ─────────────────────────────────────────────────────────

def test_factory_creates_none_reranker():
    reranker = create_reranker(_make_settings(backend="none"))
    assert isinstance(reranker, NoneReranker)


def test_none_reranker_preserves_order():
    reranker = NoneReranker()
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    assert [r["id"] for r in result] == ["c1", "c2", "c3"]


def test_none_reranker_empty_candidates():
    reranker = NoneReranker()
    assert reranker.rerank("query", []) == []


# ── 工厂错误处理测试 ──────────────────────────────────────────────────────────

def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError, match="不支持的 Reranker backend"):
        create_reranker(_make_settings(backend="no_such_backend"))


def test_factory_error_message_includes_supported_backends():
    try:
        create_reranker(_make_settings(backend="bad"))
    except ValueError as e:
        assert "none" in str(e)


def test_get_supported_backends_includes_none():
    assert "none" in get_supported_backends()


# ── CrossEncoderReranker 降级测试 ─────────────────────────────────────────────

def test_cross_encoder_fallback_on_scorer_failure():
    """scorer 抛出异常时，回退返回原始 candidates"""
    class FailingScorer:
        def predict(self, pairs):
            raise RuntimeError("model load failed")

    reranker = CrossEncoderReranker(scorer=FailingScorer())
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    # 回退：原样返回
    assert [r["id"] for r in result] == ["c1", "c2", "c3"]


def test_cross_encoder_with_mock_scorer():
    """mock scorer 正常工作时，按分数降序排列"""
    class MockScorer:
        def predict(self, pairs):
            class FakeArray(list):
                def tolist(self): return list(self)
            return FakeArray([0.5, 0.7, 0.9])

    reranker = CrossEncoderReranker(scorer=MockScorer())
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    assert result[0]["id"] == "c3"  # 分最高
    assert result[1]["id"] == "c2"
    assert result[2]["id"] == "c1"


def test_cross_encoder_empty_candidates():
    reranker = CrossEncoderReranker(scorer=None)
    assert reranker.rerank("query", []) == []


# ── LLMReranker 测试 ──────────────────────────────────────────────────────────

def test_llm_reranker_with_mock_llm():
    """mock LLM 返回正确 JSON 时，按返回顺序重排"""
    from src.libs.llm.base_llm import ChatResponse

    class MockLLM:
        def chat(self, messages):
            return ChatResponse(content='["c3", "c1", "c2"]', model="test")

    reranker = LLMReranker(llm=MockLLM())
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    assert [r["id"] for r in result] == ["c3", "c1", "c2"]


def test_llm_reranker_fallback_on_invalid_json():
    """LLM 返回无效 JSON 时，回退返回原始顺序"""
    from src.libs.llm.base_llm import ChatResponse

    class BrokenLLM:
        def chat(self, messages):
            return ChatResponse(content="this is not json", model="test")

    reranker = LLMReranker(llm=BrokenLLM())
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    assert [r["id"] for r in result] == ["c1", "c2", "c3"]


def test_llm_reranker_fallback_on_llm_exception():
    """LLM 调用抛出异常时，回退返回原始顺序"""
    class ExplodingLLM:
        def chat(self, messages):
            raise RuntimeError("network error")

    reranker = LLMReranker(llm=ExplodingLLM())
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    assert [r["id"] for r in result] == ["c1", "c2", "c3"]


def test_llm_reranker_ids_not_in_result_appended_to_end():
    """LLM 只返回部分 id 时，剩余的追加到末尾"""
    from src.libs.llm.base_llm import ChatResponse

    class PartialLLM:
        def chat(self, messages):
            return ChatResponse(content='["c3"]', model="test")

    reranker = LLMReranker(llm=PartialLLM())
    result = reranker.rerank("query", SAMPLE_CANDIDATES)
    assert result[0]["id"] == "c3"
    # c1, c2 追加到末尾（顺序不定，但都在）
    remaining_ids = {r["id"] for r in result[1:]}
    assert remaining_ids == {"c1", "c2"}
