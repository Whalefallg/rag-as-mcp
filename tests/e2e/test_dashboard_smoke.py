"""
Dashboard 冒烟测试 (tests/e2e/test_dashboard_smoke.py)
=======================================================
验收标准 (DEV_SPEC I2)：
  - 六个页面的 render() 函数均可被调用、不抛 Python 异常
  - 每个页面的服务层依赖均用 Mock 替代（无需真实 ChromaDB/File）
  - 各 service 返回空数据时页面不崩溃（空状态覆盖）
  - 各 service 返回错误时页面不崩溃（错误状态覆盖）

  测试策略：
    patch sys.modules["streamlit"] 为 MagicMock，
    直接调用各页面 render() 函数验证"不抛异常"这一核心约束。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).parent.parent.parent


# ── Streamlit mock ────────────────────────────────────────────────────────────

def _make_st_mock() -> MagicMock:
    st = MagicMock()
    # columns() must return exactly N MagicMock items matching the argument
    def _columns(n, *args, **kwargs):
        if isinstance(n, (list, tuple)):
            n = len(n)
        return [MagicMock() for _ in range(int(n))]
    st.columns.side_effect = _columns
    st.tabs.return_value = [MagicMock(), MagicMock()]
    ctx = MagicMock()
    ctx.__enter__ = lambda s: s
    ctx.__exit__ = MagicMock(return_value=False)
    st.expander.return_value = ctx
    st.form.return_value = ctx
    st.spinner.return_value = ctx
    st.session_state = {}
    st.navigation.return_value = MagicMock()
    return st


def _patch_st(st_mock):
    return patch.dict(sys.modules, {"streamlit": st_mock})


def _make_settings():
    from src.core.settings import (
        Settings, LLMConfig, EmbeddingConfig,
        VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig,
    )
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(
            provider="openai", model="text-embedding-3-small", api_key="k"
        ),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="/tmp/tc"),
        splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10,
        ),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


# ── 各页面冒烟测试 ────────────────────────────────────────────────────────────

class TestOverviewPageSmoke:
    def test_render_with_error_config(self):
        from src.observability.dashboard.services.config_service import (
            ConfigService, SystemConfig,
        )
        svc = MagicMock(spec=ConfigService)
        svc.load.return_value = SystemConfig(
            components=[], retrieval={}, rerank={}, splitter={},
            settings_path="/tmp/s.yaml", load_error="no config",
        )
        svc.get_collection_stats.return_value = {"collections": [], "total_collections": 0}
        st = _make_st_mock()
        with _patch_st(st):
            from src.observability.dashboard.pages import overview
            importlib.reload(overview)
            overview.render(svc)

    def test_render_with_valid_config(self):
        from src.observability.dashboard.services.config_service import (
            ConfigService, SystemConfig, ComponentCard,
        )
        svc = MagicMock(spec=ConfigService)
        svc.load.return_value = SystemConfig(
            components=[
                ComponentCard("LLM", "openai", "gpt-4o"),
                ComponentCard("Embedding", "openai", "text-embedding-3-small"),
                ComponentCard("VectorStore", "chroma", "/tmp/chroma"),
            ],
            retrieval={"sparse_backend": "bm25", "fusion_algorithm": "rrf",
                       "top_k_dense": 20, "top_k_sparse": 20, "top_k_final": 10},
            rerank={"backend": "none", "model": "—", "top_m": 30},
            splitter={"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
            settings_path="/tmp/s.yaml",
        )
        svc.get_collection_stats.return_value = {
            "collections": [{"name": "default", "chunk_count": 42}],
            "total_collections": 1,
        }
        st = _make_st_mock()
        with _patch_st(st):
            from src.observability.dashboard.pages import overview
            importlib.reload(overview)
            overview.render(svc)


class TestDataBrowserPageSmoke:
    def test_render_empty_collections(self):
        from src.observability.dashboard.services.data_service import DataService
        ds = MagicMock(spec=DataService)
        ds.list_collections.return_value = []
        st = _make_st_mock()
        with _patch_st(st):
            from src.observability.dashboard.pages import data_browser
            importlib.reload(data_browser)
            data_browser.render(ds)

    def test_render_with_documents(self):
        from src.observability.dashboard.services.data_service import DataService, DocumentInfo
        ds = MagicMock(spec=DataService)
        ds.list_collections.return_value = ["default"]
        ds.list_documents.return_value = [
            DocumentInfo("data/doc.pdf", "default", 10, 0, "doc.pdf"),
        ]
        st = _make_st_mock()
        st.selectbox.return_value = "default"
        st.text_input.return_value = ""
        st.session_state = {}
        with _patch_st(st):
            from src.observability.dashboard.pages import data_browser
            importlib.reload(data_browser)
            data_browser.render(ds)


class TestIngestionTracesPageSmoke:
    def test_render_no_trace_file(self):
        from src.observability.dashboard.services.trace_service import TraceService
        ts = MagicMock(spec=TraceService)
        ts.exists.return_value = False
        ts.file_path.return_value = "/tmp/t.jsonl"
        st = _make_st_mock()
        with _patch_st(st):
            from src.observability.dashboard.pages import ingestion_traces
            importlib.reload(ingestion_traces)
            ingestion_traces.render(ts)

    def test_render_empty_traces(self):
        from src.observability.dashboard.services.trace_service import TraceService
        ts = MagicMock(spec=TraceService)
        ts.exists.return_value = True
        ts.file_path.return_value = "/tmp/t.jsonl"
        ts.list.return_value = []
        st = _make_st_mock()
        st.session_state = {}
        with _patch_st(st):
            from src.observability.dashboard.pages import ingestion_traces
            importlib.reload(ingestion_traces)
            ingestion_traces.render(ts)


class TestQueryTracesPageSmoke:
    def test_render_no_trace_file(self):
        from src.observability.dashboard.services.trace_service import TraceService
        ts = MagicMock(spec=TraceService)
        ts.exists.return_value = False
        ts.file_path.return_value = "/tmp/t.jsonl"
        st = _make_st_mock()
        with _patch_st(st):
            from src.observability.dashboard.pages import query_traces
            importlib.reload(query_traces)
            query_traces.render(ts)

    def test_render_empty_traces(self):
        from src.observability.dashboard.services.trace_service import TraceService
        ts = MagicMock(spec=TraceService)
        ts.exists.return_value = True
        ts.file_path.return_value = "/tmp/t.jsonl"
        ts.list.return_value = []
        st = _make_st_mock()
        st.text_input.return_value = ""
        st.session_state = {}
        with _patch_st(st):
            from src.observability.dashboard.pages import query_traces
            importlib.reload(query_traces)
            query_traces.render(ts)


class TestEvaluationPageSmoke:
    def test_render_placeholder(self):
        st = _make_st_mock()
        with _patch_st(st):
            from src.observability.dashboard.pages import evaluation
            importlib.reload(evaluation)
            evaluation.render()

    def test_evaluation_panel_no_run(self):
        settings = _make_settings()
        st = _make_st_mock()
        st.text_input.return_value = "tests/fixtures/golden_test_set.json"
        st.selectbox.return_value = "local"
        st.button.return_value = False
        st.session_state = {}
        with _patch_st(st):
            from src.observability.dashboard.pages import evaluation_panel
            importlib.reload(evaluation_panel)
            evaluation_panel.render(settings)


class TestAllPagesNoException:
    """六页面空数据下集成冒烟"""

    def test_all_six_pages(self):
        from src.observability.dashboard.services.config_service import (
            ConfigService, SystemConfig,
        )
        from src.observability.dashboard.services.trace_service import TraceService
        from src.observability.dashboard.services.data_service import DataService

        config_svc = MagicMock(spec=ConfigService)
        config_svc.load.return_value = SystemConfig(
            components=[], retrieval={}, rerank={}, splitter={},
            settings_path="/tmp/s.yaml",
        )
        config_svc.get_collection_stats.return_value = {
            "collections": [], "total_collections": 0
        }
        trace_svc = MagicMock(spec=TraceService)
        trace_svc.exists.return_value = False
        trace_svc.file_path.return_value = "/tmp/t.jsonl"
        trace_svc.list.return_value = []

        data_svc = MagicMock(spec=DataService)
        data_svc.list_collections.return_value = []
        data_svc.list_documents.return_value = []

        settings = _make_settings()
        st = _make_st_mock()
        st.text_input.return_value = ""
        st.selectbox.return_value = "default"
        st.button.return_value = False
        st.session_state = {}

        with _patch_st(st):
            import src.observability.dashboard.pages.overview as p1
            import src.observability.dashboard.pages.data_browser as p2
            import src.observability.dashboard.pages.ingestion_traces as p4
            import src.observability.dashboard.pages.query_traces as p5
            import src.observability.dashboard.pages.evaluation as p6
            import src.observability.dashboard.pages.evaluation_panel as p6b
            for mod in [p1, p2, p4, p5, p6, p6b]:
                importlib.reload(mod)
            p1.render(config_svc)
            p2.render(data_svc)
            p4.render(trace_svc)
            p5.render(trace_svc)
            p6.render()
            p6b.render(settings)
