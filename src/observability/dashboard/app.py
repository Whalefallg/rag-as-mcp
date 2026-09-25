"""
Dashboard 入口 (src/observability/dashboard/app.py)
====================================================
为什么需要这个文件：
  Streamlit 多页面应用需要一个统一的入口来注册所有页面、初始化共享服务。
  app.py 使用 st.navigation() 把六个页面串联成一个有序导航，
  共享服务（ConfigService / DataService / TraceService）在这里实例化后
  通过函数参数传入各页面，避免每个页面各自重建连接。

  启动方式：
    streamlit run src/observability/dashboard/app.py
  或通过脚本：
    python scripts/start_dashboard.py
"""
import sys
from pathlib import Path

import streamlit as st

# 确保项目根目录在 sys.path
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

st.set_page_config(
    page_title="RAG AS MCP Dashboard",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 共享服务（只初始化一次）──────────────────────────────────────────────────

@st.cache_resource
def _get_config_service():
    from src.observability.dashboard.services.config_service import ConfigService
    return ConfigService()


@st.cache_resource
def _get_trace_service():
    from src.observability.dashboard.services.trace_service import TraceService
    return TraceService()


@st.cache_resource
def _get_data_service():
    from src.observability.dashboard.services.config_service import ConfigService
    from src.observability.dashboard.services.data_service import DataService
    cfg = ConfigService()
    config = cfg.load()
    if config.load_error:
        return None
    from src.core.settings import load_settings
    settings = load_settings()
    return DataService(settings)


@st.cache_resource
def _get_settings():
    try:
        from src.core.settings import load_settings
        return load_settings()
    except Exception:
        return None


# ── 页面定义 ─────────────────────────────────────────────────────────────────

def _page_overview():
    from src.observability.dashboard.pages.overview import render
    render(_get_config_service())


def _page_data_browser():
    ds = _get_data_service()
    if ds is None:
        st.error("DataService 初始化失败，请检查配置文件。")
        return
    from src.observability.dashboard.pages.data_browser import render
    render(ds)


def _page_ingestion_manager():
    ds = _get_data_service()
    settings = _get_settings()
    if ds is None or settings is None:
        st.error("服务初始化失败，请检查配置文件。")
        return
    from src.observability.dashboard.pages.ingestion_manager import render
    render(ds, settings)


def _page_ingestion_traces():
    from src.observability.dashboard.pages.ingestion_traces import render
    render(_get_trace_service())


def _page_query_traces():
    from src.observability.dashboard.pages.query_traces import render
    render(_get_trace_service())


def _page_evaluation():
    settings = _get_settings()
    if settings is None:
        st.error("配置文件加载失败，请检查 config/settings.yaml。")
        return
    from src.observability.dashboard.pages.evaluation_panel import render
    render(settings)


# ── 导航注册 ──────────────────────────────────────────────────────────────────

pages = st.navigation([
    st.Page(_page_overview,          title="📊 系统总览",       url_path="overview"),
    st.Page(_page_data_browser,      title="🗂️ 数据浏览器",    url_path="data-browser"),
    st.Page(_page_ingestion_manager, title="📥 Ingestion 管理", url_path="ingestion"),
    st.Page(_page_ingestion_traces,  title="⏱️ Ingestion 追踪", url_path="ingestion-traces"),
    st.Page(_page_query_traces,      title="🔍 Query 追踪",     url_path="query-traces"),
    st.Page(_page_evaluation,        title="📈 评估面板",        url_path="evaluation"),
])

pages.run()
