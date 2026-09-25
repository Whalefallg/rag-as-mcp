"""
Query 追踪页面 (src/observability/dashboard/pages/query_traces.py)
===================================================================
为什么需要这个文件：
  调试 RAG 效果时最常见的问题是"为什么没找到我想要的文档"。
  Query 追踪页面展示：
    - Dense vs Sparse 各自召回了多少条
    - 融合后的分布
    - Rerank 前后排名变化
  这让工程师无需添加任何日志代码，直接在 Dashboard 里定位问题。
"""
import streamlit as st


def render(trace_service) -> None:
    st.title("🔍 Query 追踪")
    st.caption("查看查询历史、Dense/Sparse 召回对比与 Rerank 排名变化")

    if not trace_service.exists():
        st.info(
            "暂无追踪记录。\n\n"
            "通过 MCP Client 执行一次查询后，此处将自动显示追踪数据。\n"
            f"日志文件路径：`{trace_service.file_path()}`"
        )
        return

    # ── 搜索过滤 ──────────────────────────────────────────────────────────
    keyword = st.text_input("🔍 按 Query 关键词搜索", placeholder="输入关键词…")
    records = trace_service.list(trace_type="query", keyword=keyword or None)

    if not records:
        msg = "暂无 Query 追踪记录。" if not keyword else f"未找到包含 `{keyword}` 的查询记录。"
        st.info(msg)
        return

    # ── 历史列表 ──────────────────────────────────────────────────────────
    st.subheader(f"📋 查询历史（{len(records)} 条）")

    for rec in records:
        query_preview = rec.user_query[:60] + ("…" if len(rec.user_query) > 60 else "")
        elapsed = f"{rec.total_elapsed_ms:.0f} ms" if rec.total_elapsed_ms else "—"
        label = f"💬 {query_preview}  ·  {elapsed}  ·  `{rec.trace_id[:8]}…`"
        if st.button(label, key=f"qt_{rec.trace_id}"):
            st.session_state["selected_query_trace"] = rec.trace_id

    selected_id = st.session_state.get("selected_query_trace")
    if not selected_id:
        st.info("点击上方任意记录查看详情。")
        return

    rec = trace_service.get(selected_id)
    if rec is None:
        return

    # ── 详情 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("查询详情")

    st.markdown(f"**Query**：{rec.user_query}")
    st.caption(f"集合：`{rec.collection}`  ·  trace_id：`{rec.trace_id}`")
    st.caption(f"时间：{rec.started_at}")

    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("总耗时", f"{rec.total_elapsed_ms:.0f} ms" if rec.total_elapsed_ms else "—")
    with meta_cols[1]:
        dense_stage = rec.stage_by_name("dense_retrieval")
        st.metric("Dense 召回", dense_stage.get("result_count", "—") if dense_stage else "—")
    with meta_cols[2]:
        sparse_stage = rec.stage_by_name("sparse_retrieval")
        st.metric("Sparse 召回", sparse_stage.get("result_count", "—") if sparse_stage else "—")

    # ── Dense vs Sparse 对比 ──────────────────────────────────────────────
    _render_retrieval_comparison(rec)

    # ── Rerank 信息 ───────────────────────────────────────────────────────
    rerank_stage = rec.stage_by_name("rerank")
    if rerank_stage:
        st.subheader("🎯 Rerank")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Backend", rerank_stage.get("backend", "—"))
        with rc2:
            st.metric("结果数", rerank_stage.get("result_count", "—"))
        with rc3:
            fallback = rerank_stage.get("fallback", False)
            st.metric("降级", "是 ⚠️" if fallback else "否 ✅")
        if rerank_stage.get("error"):
            st.error(f"Rerank 错误：{rerank_stage['error']}")

    # ── 全量阶段耗时 ──────────────────────────────────────────────────────
    stages_with_dur = [s for s in rec.stages if s.get("duration_ms") is not None]
    if stages_with_dur:
        st.subheader("⏱️ 阶段耗时")
        _render_stage_bar(stages_with_dur)


def _render_retrieval_comparison(rec) -> None:
    """Dense vs Sparse 并列对比卡片"""
    dense = rec.stage_by_name("dense_retrieval")
    sparse = rec.stage_by_name("sparse_retrieval")
    fusion = rec.stage_by_name("fusion")

    if not dense and not sparse:
        return

    st.subheader("📊 召回对比")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Dense（语义）**")
        if dense:
            status = dense.get("status", "?")
            icon = "✅" if status == "ok" else "⚠️"
            st.metric("召回数", dense.get("result_count", "—"),
                      delta=f"{dense.get('duration_ms', 0):.0f} ms")
            st.caption(f"{icon} 状态：{status}")
            if dense.get("error"):
                st.error(dense["error"])
        else:
            st.caption("未执行")

    with c2:
        st.markdown("**Sparse（BM25）**")
        if sparse:
            status = sparse.get("status", "?")
            icon = "✅" if status == "ok" else "⚠️"
            st.metric("召回数", sparse.get("result_count", "—"),
                      delta=f"{sparse.get('duration_ms', 0):.0f} ms")
            st.caption(f"{icon} 状态：{status}")
            if sparse.get("error"):
                st.error(sparse["error"])
        else:
            st.caption("未执行")

    with c3:
        st.markdown("**融合（RRF）**")
        if fusion:
            st.metric("融合结果", fusion.get("result_count", "—"),
                      delta=f"{fusion.get('duration_ms', 0):.0f} ms")
            st.caption(f"算法：{fusion.get('algorithm', 'rrf')}")
        else:
            st.caption("未执行")


def _render_stage_bar(stages: list) -> None:
    try:
        import pandas as pd
        data = {s["stage"]: s["duration_ms"] for s in stages if s.get("duration_ms")}
        df = pd.DataFrame.from_dict({"耗时 (ms)": data}, orient="index").T
        st.bar_chart(df, use_container_width=True)
    except ImportError:
        for s in stages:
            dur = s.get("duration_ms", 0)
            bar = "█" * max(1, int(dur / 20))
            st.text(f"{s['stage']:24s} {bar} {dur:.0f} ms")
