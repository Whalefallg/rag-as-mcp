"""
ConfigService 单元测试 (tests/unit/test_config_service.py)
===========================================================
验收标准 (DEV_SPEC G1)：
  - load() 成功时返回 SystemConfig 含所有组件 card
  - load() 失败时返回 load_error 非空的 SystemConfig（不抛异常）
  - ComponentCard 含 category/provider/model
  - get_collection_stats() Chroma 不可用时返回 error 字段（不抛异常）
"""
import pytest
from unittest.mock import patch, MagicMock
from src.observability.dashboard.services.config_service import (
    ConfigService, SystemConfig, ComponentCard
)


def _make_settings_mock():
    s = MagicMock()
    s.llm.provider = "openai"
    s.llm.model = "gpt-4o"
    s.llm.azure_endpoint = None
    s.embedding.provider = "openai"
    s.embedding.model = "text-embedding-3-small"
    s.vector_store.backend = "chroma"
    s.vector_store.persist_path = "/tmp/test_chroma"
    s.retrieval.sparse_backend = "bm25"
    s.retrieval.fusion_algorithm = "rrf"
    s.retrieval.top_k_dense = 20
    s.retrieval.top_k_sparse = 20
    s.retrieval.top_k_final = 10
    s.rerank.backend = "none"
    s.rerank.model = None
    s.rerank.top_m = 30
    s.splitter.method = "recursive"
    s.splitter.chunk_size = 512
    s.splitter.chunk_overlap = 64
    return s


class TestConfigServiceLoad:
    def test_success_returns_system_config(self):
        svc = ConfigService()
        mock_settings = _make_settings_mock()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   return_value=mock_settings):
            config = svc.load()
        assert isinstance(config, SystemConfig)
        assert config.load_error is None

    def test_three_component_cards(self):
        svc = ConfigService()
        mock_settings = _make_settings_mock()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   return_value=mock_settings):
            config = svc.load()
        categories = [c.category for c in config.components]
        assert "LLM" in categories
        assert "Embedding" in categories
        assert "VectorStore" in categories

    def test_llm_card_fields(self):
        svc = ConfigService()
        mock_settings = _make_settings_mock()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   return_value=mock_settings):
            config = svc.load()
        llm_card = next(c for c in config.components if c.category == "LLM")
        assert llm_card.provider == "openai"
        assert llm_card.model == "gpt-4o"

    def test_retrieval_config_present(self):
        svc = ConfigService()
        mock_settings = _make_settings_mock()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   return_value=mock_settings):
            config = svc.load()
        assert config.retrieval["top_k_final"] == 10
        assert config.retrieval["fusion_algorithm"] == "rrf"

    def test_failure_returns_load_error_not_raises(self):
        svc = ConfigService(settings_path="/nonexistent/settings.yaml")
        config = svc.load()
        assert isinstance(config, SystemConfig)
        assert config.load_error is not None
        assert len(config.components) == 0

    def test_settings_path_in_output(self):
        svc = ConfigService(settings_path="config/settings.yaml")
        mock_settings = _make_settings_mock()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   return_value=mock_settings):
            config = svc.load()
        assert "settings.yaml" in config.settings_path


class TestGetCollectionStats:
    def test_chromadb_unavailable_returns_error_not_raises(self):
        svc = ConfigService()
        mock_settings = _make_settings_mock()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   return_value=mock_settings):
            with patch("src.observability.dashboard.services.config_service.chromadb",
                       None, create=True):
                stats = svc.get_collection_stats()
        assert "error" in stats or stats["total_collections"] == 0

    def test_returns_dict_with_collections_key(self):
        svc = ConfigService()
        with patch("src.observability.dashboard.services.config_service.load_settings",
                   side_effect=Exception("no config")):
            stats = svc.get_collection_stats()
        assert "collections" in stats
        assert "total_collections" in stats


class TestComponentCard:
    def test_fields_accessible(self):
        card = ComponentCard(
            category="LLM", provider="openai", model="gpt-4o"
        )
        assert card.category == "LLM"
        assert card.provider == "openai"
        assert card.model == "gpt-4o"
        assert card.details == {}

    def test_with_details(self):
        card = ComponentCard(
            category="VectorStore", provider="chroma",
            model="/tmp/chroma",
            details={"persist_path": "/tmp/chroma"},
        )
        assert card.details["persist_path"] == "/tmp/chroma"
