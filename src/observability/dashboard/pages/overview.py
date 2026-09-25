"""
系统总览页面 (src/observability/dashboard/pages/overview.py)
=============================================================
为什么需要这个文件：
  运行前先看"系统健康状态"——当前配置了哪些组件、知识库里有多少数据，
  是最常见的运维需求。Overview 页面一屏展示所有关键信息，无需进入命令行。

  动态感知：所有组件信息来自 ConfigService，切换 provider 后页面自动更新。
"""
import streamlit as st


def render(config_service) -> None:
    st.title("📊 系统总览")
    st.caption("当前系统配置与知识库统计概览")

    config = config_service.load()

    if config.load_error:
        st.error(f"⚠️ 配置加载失败：{config.load_error}")
        st.info("请确认 `config/settings.yaml` 存在且格式正确。")
        return

    # ── 组件配置卡片 ──────────────────────────────────────────────────────
    st.subheader("🔧 可插拔组件")
    cols = st.columns(len(config.components))
    for col, card in zip(cols, config.components):
        with col:
            st.metric(label=card.category, value=card.provider)
            st.caption(f"模型：`{card.model}`")
            for k, v in card.details.items():
                if v and v != "—":
                    st.caption(f"{k}：`{v}`")

    st.divider()

    # ── 检索与重排配置 ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🔍 检索配置")
        r = config.retrieval
        st.markdown(f"""
- **Sparse Backend**：`{r.get('sparse_backend', '—')}`
- **融合算法**：`{r.get('fusion_algorithm', '—')}`
- **Dense Top-K**：{r.get('top_k_dense', '—')}
- **Sparse Top-K**：{r.get('top_k_sparse', '—')}
- **最终 Top-K**：{r.get('top_k_final', '—')}
""")
    with c2:
        st.subheader("🎯 重排配置")
        rr = config.rerank
        st.markdown(f"""
- **Backend**：`{rr.get('backend', '—')}`
- **Model**：`{rr.get('model', '—')}`
- **Top-M**：{rr.get('top_m', '—')}
""")
    with c3:
        st.subheader("✂️ 切分配置")
        sp = config.splitter
        st.markdown(f"""
- **方法**：`{sp.get('method', '—')}`
- **Chunk Size**：{sp.get('chunk_size', '—')}
- **Chunk Overlap**：{sp.get('chunk_overlap', '—')}
""")

    st.divider()

    # ── 知识库统计 ────────────────────────────────────────────────────────
    st.subheader("🗄️ 知识库统计")
    stats = config_service.get_collection_stats()

    if stats.get("error"):
        st.warning(f"无法连接 ChromaDB：{stats['error']}")
    elif not stats["collections"]:
        st.info("知识库为空，请先使用 Ingestion 页面摄取文档。")
    else:
        cols = st.columns(min(len(stats["collections"]), 4))
        for col, coll in zip(cols, stats["collections"]):
            with col:
                st.metric(
                    label=f"📁 {coll['name']}",
                    value=f"{coll['chunk_count']} chunks",
                )

    st.divider()
    st.caption(f"配置文件路径：`{config.settings_path}`")
