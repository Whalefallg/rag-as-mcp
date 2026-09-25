"""
MetadataEnricher 契约测试 (tests/unit/test_metadata_enricher_contract.py)
=========================================================================
为什么需要这个文件：
  检索结果的 metadata（title/summary/tags）直接影响用户体验：
  UI 展示结果时需要显示标题和摘要，而不是裸露的原始文本。
  LLM 模式提取的 metadata 质量最高，但 LLM 可能返回格式错误的 JSON。
  契约测试确保无论 LLM 成功还是失败，title/summary/tags 字段一定存在且非空，
  上层代码可以无防御性地直接访问这三个字段。

验收标准（DEV_SPEC C6）：
  - 规则模式输出必须包含非空的 title/summary/tags
  - LLM 模式：mock LLM 返回有效 JSON 时 enriched_by="llm"
  - 降级：LLM 失败时回退到规则结果，enriched_by="rule"
  - LLM 返回无效 JSON 时降级到规则模式
"""
import pytest
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher


def _make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(id=f"c{idx}", doc_id="doc1", text=text, index=idx, metadata={})


def _make_settings_llm_on():
    from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={"ingestion": {"metadata_enricher": {"use_llm": True}}},
    )


def _mock_llm(json_response: str):
    from src.libs.llm.base_llm import ChatResponse
    class MockLLM:
        def chat(self, messages):
            return ChatResponse(content=json_response, model="mock")
    return MockLLM()


# ── 规则模式 ──────────────────────────────────────────────────────────────────

def test_rule_mode_title_non_empty():
    enricher = MetadataEnricher()
    chunk = _make_chunk("This is a meaningful paragraph about machine learning.")
    result = enricher.transform([chunk])
    assert result[0].metadata.get("title", "") != ""


def test_rule_mode_summary_non_empty():
    enricher = MetadataEnricher()
    chunk = _make_chunk("Summary text here. It has multiple sentences.")
    result = enricher.transform([chunk])
    assert result[0].metadata.get("summary", "") != ""


def test_rule_mode_tags_list():
    enricher = MetadataEnricher()
    chunk = _make_chunk("machine learning deep neural network training optimization")
    result = enricher.transform([chunk])
    tags = result[0].metadata.get("tags", None)
    assert tags is not None
    assert isinstance(tags, list)


def test_rule_mode_enriched_by_marker():
    enricher = MetadataEnricher()
    chunk = _make_chunk("some text")
    result = enricher.transform([chunk])
    assert result[0].metadata["enriched_by"] == "rule"


def test_rule_title_uses_first_line():
    enricher = MetadataEnricher()
    chunk = _make_chunk("First Line Title\nSecond line content here")
    result = enricher.transform([chunk])
    assert "First Line Title" in result[0].metadata["title"]


# ── LLM 模式 ──────────────────────────────────────────────────────────────────

def test_llm_mode_valid_json():
    settings = _make_settings_llm_on()
    llm = _mock_llm('{"title": "LLM Title", "summary": "LLM Summary", "tags": ["a", "b"]}')
    enricher = MetadataEnricher(settings=settings, llm=llm)
    chunk = _make_chunk("some text")
    result = enricher.transform([chunk])
    assert result[0].metadata["title"] == "LLM Title"
    assert result[0].metadata["enriched_by"] == "llm"


def test_llm_mode_fallback_on_invalid_json():
    settings = _make_settings_llm_on()
    llm = _mock_llm("this is not json")
    enricher = MetadataEnricher(settings=settings, llm=llm)
    chunk = _make_chunk("some text")
    result = enricher.transform([chunk])
    assert result[0].metadata["enriched_by"] == "rule"


def test_llm_mode_fallback_on_exception():
    settings = _make_settings_llm_on()
    class FailingLLM:
        def chat(self, messages):
            raise RuntimeError("API error")
    enricher = MetadataEnricher(settings=settings, llm=FailingLLM())
    chunk = _make_chunk("some text")
    result = enricher.transform([chunk])
    assert result[0].metadata["enriched_by"] == "rule"


def test_llm_mode_missing_fields_fallback():
    settings = _make_settings_llm_on()
    llm = _mock_llm('{"title": "Only Title"}')  # 缺 summary 和 tags
    enricher = MetadataEnricher(settings=settings, llm=llm)
    chunk = _make_chunk("some text")
    result = enricher.transform([chunk])
    assert result[0].metadata["enriched_by"] == "rule"


# ── 批量 ──────────────────────────────────────────────────────────────────────

def test_multiple_chunks_all_enriched():
    enricher = MetadataEnricher()
    chunks = [_make_chunk(f"content for chunk {i}", i) for i in range(4)]
    result = enricher.transform(chunks)
    assert len(result) == 4
    for r in result:
        assert "title" in r.metadata
        assert "summary" in r.metadata


def test_empty_list():
    enricher = MetadataEnricher()
    assert enricher.transform([]) == []
