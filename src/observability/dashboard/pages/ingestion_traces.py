"""
Ingestion 追踪页面 (src/observability/dashboard/pages/ingestion_traces.py)
==========================================================================
为什么需要这个文件：
  摄取后需要诊断"哪一步最慢"——是 Embedding 调用超时、还是写 Chroma 慢。
  瀑布图让各阶段耗时一目了然，快速定位性能瓶颈。
"""
import streamlit as st


def render(trace_service) -> None:
    st.title("⏱️ Ingestion 追踪")
    st.caption("查看摄取历史与各阶段耗时瀑布图")

    if not trace_service.exists():
        st.info(
            "暂无追踪记录。\n\n"
            "执行一次摄取后，此处将自动显示追踪数据。\n"
            f"日志文件路径：`{trace_service.file_path()}`"
        )
        return

    records = trace_service.list(trace_type="ingestion")
    if not records:
        st.info("暂无 Ingestion 追踪记录。")
        return

    # ── 历史列表 ──────────────────────────────────────────────────────────
    st.subheader(f"📋 摄取历史（最近 {len(records)} 条）")

    selected_id = None
    for rec in records:
        src = rec.source_path.replace("\\", "/").rsplit("/", 1)[-1]
        elapsed = f"{rec.total_elapsed_ms:.0f} ms" if rec.total_elapsed_ms else "—"
        label = f"📄 {src}  ·  {elapsed}  ·  `{rec.trace_id[:8]}…`"
        if st.button(label, key=f"it_{rec.trace_id}"):
            selected_id = rec.trace_id
            st.session_state["selected_ingestion_trace"] = rec.trace_id

    selected_id = st.session_state.get("selected_ingestion_trace")
    if not selected_id:
        st.info("点击上方任意记录查看详情。")
        return

    rec = trace_service.get(selected_id)
    if rec is None:
        return

    # ── 详情 & 瀑布图 ─────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"详情：`{rec.trace_id[:16]}…`")

    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("总耗时", f"{rec.total_elapsed_ms:.0f} ms" if rec.total_elapsed_ms else "—")
    with meta_cols[1]:
        st.metric("集合", rec.collection)
    with meta_cols[2]:
        st.metric("阶段数", len(rec.stages))

    st.caption(f"文件：`{rec.source_path}`")
    st.caption(f"开始时间：{rec.started_at}")

    # 瀑布图（横向条形图）
    stages_with_duration = [
        s for s in rec.stages
        if s.get("duration_ms") is not None
    ]
    if stages_with_duration:
        st.subheader("⏱️ 阶段耗时瀑布图")
        _render_waterfall(stages_with_duration)

    # 阶段详情表
    st.subheader("📊 阶段详情")
    for s in rec.stages:
        status_icon = "❌" if s.get("error") else "✅"
        dur = f"{s['duration_ms']:.1f} ms" if s.get("duration_ms") else "—"
        with st.expander(f"{status_icon} {s['stage']}  ·  {dur}", expanded=False):
            detail = {k: v for k, v in s.items()
                      if k not in ("stage", "duration_ms")}
            if detail:
                st.json(detail)
            else:
                st.caption("无附加数据")


def _render_waterfall(stages: list) -> None:
    """用 st.bar_chart 渲染阶段耗时条形图"""
    try:
        import pandas as pd
        data = {
            s["stage"]: s["duration_ms"]
            for s in stages
            if s.get("duration_ms") is not None
        }
        df = pd.DataFrame.from_dict(
            {"耗时 (ms)": data}, orient="index"
        ).T
        st.bar_chart(df, use_container_width=True)
    except ImportError:
        # pandas 不可用时回退到文本展示
        for s in stages:
            dur = s.get("duration_ms", 0)
            bar = "█" * max(1, int(dur / 50))
            st.text(f"{s['stage']:20s} {bar} {dur:.0f} ms")
