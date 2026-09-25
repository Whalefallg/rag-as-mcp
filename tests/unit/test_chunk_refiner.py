"""
ChunkRefiner 单元测试 (tests/unit/test_chunk_refiner.py)
========================================================
为什么需要这个文件：
  原始 PDF 解析出来的文本充满噪音：HTML 标签、分隔线、页码行、多余空格。
  这些噪音直接进向量库会降低检索质量——Embedding 模型会浪费编码容量在无意义符号上。
  ChunkRefiner 的规则模式做基础清洗，LLM 模式做更深度的语义整理。
  降级测试验证 LLM 失败时自动回退到规则结果，保证整个 Pipeline 不会卡在这一步。

验收标准（DEV_SPEC C5）：
  - 规则模式去除 HTML 注释、分隔线、页码行、多余空白
  - 代码块内容不被破坏
  - LLM 模式：mock LLM 调用成功时 refined_by="llm"
  - 降级：LLM 失败时 refined_by="rule"
  - 单个 chunk 处理异常不影响其他 chunk
"""
import pytest
from src.core.types import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner


def _make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(id=f"c{idx}", doc_id="doc1", text=text, index=idx, metadata={})


# ── 规则模式测试 ──────────────────────────────────────────────────────────────

def test_rule_removes_html_comments():
    refiner = ChunkRefiner()
    chunk = _make_chunk("Before <!-- hidden --> After")
    result = refiner.transform([chunk])
    assert "<!--" not in result[0].text
    assert "Before" in result[0].text
    assert "After" in result[0].text


def test_rule_removes_html_tags():
    refiner = ChunkRefiner()
    chunk = _make_chunk("<p>Hello</p> <br/> World")
    result = refiner.transform([chunk])
    assert "<p>" not in result[0].text
    assert "Hello" in result[0].text


def test_rule_removes_separator_lines():
    refiner = ChunkRefiner()
    chunk = _make_chunk("Title\n---\nContent")
    result = refiner.transform([chunk])
    assert "---" not in result[0].text
    assert "Title" in result[0].text
    assert "Content" in result[0].text


def test_rule_removes_page_number_line():
    refiner = ChunkRefiner()
    chunk = _make_chunk("Content here\nPage 5\nMore content")
    result = refiner.transform([chunk])
    assert result[0].text.count("Page 5") == 0


def test_rule_collapses_excessive_newlines():
    refiner = ChunkRefiner()
    chunk = _make_chunk("Line 1\n\n\n\n\nLine 2")
    result = refiner.transform([chunk])
    # 最多两个连续换行
    assert "\n\n\n" not in result[0].text


def test_rule_collapses_spaces():
    refiner = ChunkRefiner()
    chunk = _make_chunk("Word1    Word2    Word3")
    result = refiner.transform([chunk])
    assert "   " not in result[0].text


def test_rule_preserves_clean_text():
    refiner = ChunkRefiner()
    text = "This is clean text with normal formatting."
    chunk = _make_chunk(text)
    result = refiner.transform([chunk])
    assert result[0].text == text


def test_rule_refined_by_marker():
    refiner = ChunkRefiner()
    chunk = _make_chunk("some text")
    result = refiner.transform([chunk])
    assert result[0].metadata["refined_by"] == "rule"


# ── LLM 模式测试 ──────────────────────────────────────────────────────────────

def _make_mock_llm(response_text: str):
    from src.libs.llm.base_llm import ChatResponse
    class MockLLM:
        def chat(self, messages):
            return ChatResponse(content=response_text, model="mock")
    return MockLLM()


def test_llm_mode_uses_llm_response(monkeypatch):
    settings = _make_settings_with_llm()
    mock_llm = _make_mock_llm("LLM cleaned text")
    refiner = ChunkRefiner(settings=settings, llm=mock_llm)
    chunk = _make_chunk("noisy text <!-- comment -->")
    result = refiner.transform([chunk])
    assert result[0].text == "LLM cleaned text"
    assert result[0].metadata["refined_by"] == "llm"


def test_llm_mode_fallback_on_exception(monkeypatch):
    settings = _make_settings_with_llm()
    class FailingLLM:
        def chat(self, messages):
            raise RuntimeError("API down")
    refiner = ChunkRefiner(settings=settings, llm=FailingLLM())
    chunk = _make_chunk("some text <!-- comment -->")
    result = refiner.transform([chunk])
    # 降级到规则结果
    assert result[0].metadata["refined_by"] == "rule"
    assert "<!--" not in result[0].text


def test_llm_mode_fallback_on_empty_response(monkeypatch):
    settings = _make_settings_with_llm()
    mock_llm = _make_mock_llm("")  # 空响应
    refiner = ChunkRefiner(settings=settings, llm=mock_llm)
    chunk = _make_chunk("text")
    result = refiner.transform([chunk])
    assert result[0].metadata["refined_by"] == "rule"


# ── 批量处理与异常隔离 ────────────────────────────────────────────────────────

def test_multiple_chunks_processed():
    refiner = ChunkRefiner()
    chunks = [_make_chunk(f"text {i}", i) for i in range(5)]
    result = refiner.transform(chunks)
    assert len(result) == 5


def test_chunk_order_preserved():
    refiner = ChunkRefiner()
    chunks = [_make_chunk(f"chunk {i}", i) for i in range(3)]
    result = refiner.transform(chunks)
    assert [c.index for c in result] == [0, 1, 2]


def test_empty_chunks_list():
    refiner = ChunkRefiner()
    assert refiner.transform([]) == []


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_settings_with_llm():
    from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={"ingestion": {"chunk_refiner": {"use_llm": True}}},
    )
