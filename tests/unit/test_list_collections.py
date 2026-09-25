"""
list_collections Tool 单元测试 (tests/unit/test_list_collections.py)
====================================================================
验收标准 (DEV_SPEC E4)：
  - 对 fixtures 目录结构能返回集合名列表
  - Chroma 不可用时回退到文件系统扫描
  - 无集合时返回友好提示
"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from src.core.settings import (
    Settings, LLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
)
from src.mcp_server.tools.list_collections import execute


def _make_settings(persist_path: str = "/tmp/test_chroma") -> Settings:
    from src.core.settings import SplitterConfig
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="k"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path=persist_path),
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


class TestListCollections:
    def test_returns_text_content(self):
        """始终返回 text 类型的内容"""
        settings = _make_settings()
        with patch("src.mcp_server.tools.list_collections.chromadb", None, create=True):
            # Chroma 不可用，应回退到文件系统
            with tempfile.TemporaryDirectory() as tmpdir:
                os.makedirs(os.path.join(tmpdir, "docs", "default"), exist_ok=True)
                with patch("src.mcp_server.tools.list_collections.Path") as MockPath:
                    MockPath.return_value.exists.return_value = False
                    content = execute({}, settings)
        assert len(content) >= 1
        assert content[0]["type"] == "text"

    def test_no_collections_returns_friendly_message(self):
        """Chroma 失败且无目录时返回友好提示"""
        settings = _make_settings()

        with patch("src.mcp_server.tools.list_collections._get_from_chroma", return_value=[]):
            with patch("src.mcp_server.tools.list_collections._get_from_filesystem", return_value=[]):
                content = execute({}, settings)

        assert "暂无" in content[0]["text"] or "请先" in content[0]["text"]

    def test_chroma_collections_returned(self):
        """当 _get_from_chroma 有结果时直接使用"""
        settings = _make_settings()
        mock_collections = [
            {"name": "default", "chunk_count": 42, "document_count": 3},
            {"name": "research", "chunk_count": 100, "document_count": 10},
        ]

        with patch("src.mcp_server.tools.list_collections._get_from_chroma", return_value=mock_collections):
            content = execute({}, settings)

        text = content[0]["text"]
        assert "default" in text
        assert "research" in text
        assert "42" in text

    def test_filesystem_fallback_when_chroma_empty(self):
        """Chroma 返回空列表时回退到文件系统"""
        settings = _make_settings()

        with patch("src.mcp_server.tools.list_collections._get_from_chroma", return_value=[]):
            with patch("src.mcp_server.tools.list_collections._get_from_filesystem",
                       return_value=[{"name": "my_docs", "chunk_count": 0, "document_count": 5}]):
                content = execute({}, settings)

        assert "my_docs" in content[0]["text"]

    def test_collection_count_shown(self):
        """集合数量显示在输出中"""
        settings = _make_settings()
        mock_collections = [{"name": "col1", "chunk_count": 10, "document_count": 1}]

        with patch("src.mcp_server.tools.list_collections._get_from_chroma", return_value=mock_collections):
            content = execute({}, settings)

        assert "1 个集合" in content[0]["text"] or "col1" in content[0]["text"]
