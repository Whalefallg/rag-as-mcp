"""
get_document_summary Tool 单元测试 (tests/unit/test_get_document_summary.py)
============================================================================
验收标准 (DEV_SPEC E5)：
  - 对不存在的 doc_id 返回规范错误提示
  - 存在时返回结构化信息（title/chunk_count/tags 等）
  - doc_id 为空时返回错误
  - ChromaDB 不可用时优雅降级
"""
import pytest
from unittest.mock import patch, MagicMock

from src.core.settings import (
    Settings, LLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
)
from src.mcp_server.tools.get_document_summary import (
    execute, _aggregate_metadata, _format_summary, _basename,
)


def _make_settings() -> Settings:
    from src.core.settings import SplitterConfig
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="k"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="/tmp/test"),
        splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


class TestExecute:
    def test_empty_doc_id_returns_error(self):
        content = execute({"doc_id": ""}, _make_settings())
        assert content[0]["type"] == "text"
        assert "错误" in content[0]["text"]

    def test_missing_doc_id_returns_error(self):
        content = execute({}, _make_settings())
        assert "错误" in content[0]["text"]

    def test_not_found_returns_helpful_message(self):
        """ChromaDB 没有该文档时返回友好提示"""
        with patch("src.mcp_server.tools.get_document_summary._get_from_chroma", return_value={}):
            content = execute({"doc_id": "missing.pdf"}, _make_settings())

        text = content[0]["text"]
        assert "未找到" in text
        assert "missing.pdf" in text

    def test_found_document_returns_summary(self):
        """找到文档时返回 Markdown 摘要"""
        mock_summary = {
            "doc_id": "data/documents/default/guide.pdf",
            "title": "用户指南",
            "chunk_count": 15,
            "page_count": 3,
            "pages": [1, 2, 3],
            "summaries": ["这是一份用户指南"],
            "tags": ["教程", "配置"],
            "first_chunk_text": "本文档介绍了系统的安装步骤",
            "source_path": "data/documents/default/guide.pdf",
        }
        with patch("src.mcp_server.tools.get_document_summary._get_from_chroma", return_value=mock_summary):
            content = execute({"doc_id": "data/documents/default/guide.pdf"}, _make_settings())

        text = content[0]["text"]
        assert "用户指南" in text
        assert "15" in text
        assert "教程" in text

    def test_chroma_unavailable_returns_not_found(self):
        """ChromaDB 抛异常时优雅降级"""
        with patch("src.mcp_server.tools.get_document_summary._get_from_chroma",
                   side_effect=Exception("chroma error")):
            content = execute({"doc_id": "any.pdf"}, _make_settings())
        assert content[0]["type"] == "text"


class TestAggregateMetadata:
    def test_basic_aggregation(self):
        metadatas = [
            {"title": "标题A", "summary": "摘要1", "tags": ["t1"], "source_path": "doc.pdf"},
            {"title": "标题A", "summary": "摘要2", "tags": ["t2"], "source_path": "doc.pdf", "page": "2"},
        ]
        documents = ["内容1", "内容2"]
        result = _aggregate_metadata("doc.pdf", metadatas, documents)
        assert result["chunk_count"] == 2
        assert result["title"] == "标题A"
        assert "摘要1" in result["summaries"]
        assert "t1" in result["tags"]
        assert "t2" in result["tags"]

    def test_page_count(self):
        metadatas = [{"page": "1"}, {"page": "2"}, {"page": "3"}]
        result = _aggregate_metadata("doc.pdf", metadatas, [])
        assert result["page_count"] == 3

    def test_deduplicates_titles(self):
        metadatas = [
            {"title": "同一标题"},
            {"title": "同一标题"},
        ]
        result = _aggregate_metadata("doc.pdf", metadatas, [])
        assert result["summaries"].count("同一标题") <= 1 or result["title"] == "同一标题"

    def test_fallback_title_from_path(self):
        """无 title 时使用文件名作为标题"""
        result = _aggregate_metadata("data/docs/guide.pdf", [{}], [])
        assert result["title"] == "guide.pdf"

    def test_tags_from_comma_string(self):
        """tags 可以是逗号分隔的字符串"""
        metadatas = [{"tags": "ai,ml,rag"}]
        result = _aggregate_metadata("doc.pdf", metadatas, [])
        assert "ai" in result["tags"]
        assert "ml" in result["tags"]


class TestFormatSummary:
    def test_contains_title(self):
        summary = {
            "doc_id": "doc.pdf", "title": "测试文档",
            "chunk_count": 5, "page_count": 0, "pages": [],
            "summaries": [], "tags": [], "first_chunk_text": "",
            "source_path": "doc.pdf",
        }
        text = _format_summary(summary)
        assert "测试文档" in text

    def test_contains_chunk_count(self):
        summary = {
            "doc_id": "doc.pdf", "title": "Doc",
            "chunk_count": 42, "page_count": 0, "pages": [],
            "summaries": [], "tags": [], "first_chunk_text": "",
            "source_path": "doc.pdf",
        }
        assert "42" in _format_summary(summary)

    def test_tags_shown_when_present(self):
        summary = {
            "doc_id": "doc.pdf", "title": "Doc",
            "chunk_count": 1, "page_count": 0, "pages": [],
            "summaries": [], "tags": ["RAG", "LLM"], "first_chunk_text": "",
            "source_path": "doc.pdf",
        }
        text = _format_summary(summary)
        assert "RAG" in text
        assert "LLM" in text


class TestBasename:
    def test_unix_path(self):
        assert _basename("data/documents/default/guide.pdf") == "guide.pdf"

    def test_windows_path(self):
        assert _basename(r"data\documents\guide.pdf") == "guide.pdf"

    def test_empty_string(self):
        assert _basename("") == ""

    def test_filename_only(self):
        assert _basename("guide.pdf") == "guide.pdf"
