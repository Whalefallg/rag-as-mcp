"""
评估面板页面 (src/observability/dashboard/pages/evaluation_panel.py)
====================================================================
为什么需要这个文件：
  替换 Phase G 的占位提示，提供完整的 RAG 评估交互界面：
    1. 选择 golden test set 和评估器类型
    2. 一键触发评估（调用 EvalRunner）
    3. 展示 Hit Rate / MRR / Precision@K 等指标
    4. 逐条查看检索命中详情
    5. 历史评估结果对比

  离线优先：默认 local evaluator，无需 API Key 即可在演示环境中运行。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st


def render(settings) -> None:
    st.title("📈 评估面板")
    st.caption("基于 Golden Test Set 量化 RAG 检索质量（Hit Rate / MRR / Precision@K）")

    # ── 配置区 ────────────────────────────────────────────────────────────
    with st.expander("⚙️ 评估配置", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            test_set_path = st.text_input(
                "Golden Test Set 路径",
                value="tests/fixtures/golden_test_set.json",
            )
        with c2:
            evaluator_type = st.selectbox(
                "评估器",
                ["local", "ragas", "composite"],
                help="local=离线（无需API），ragas=需要LLM API，composite=组合",
            )
        with c3:
            collection = st.text_input("集合名称", value="default")

    # ── 测试集预览 ────────────────────────────────────────────────────────
    if Path(test_set_path).exists():
        try:
            data = json.loads(Path(test_set_path).read_text(encoding="utf-8"))
            cases = data.get("test_cases", [])
            st.caption(f"测试集：{len(cases)} 条用例  ·  `{test_set_path}`")
        except Exception:
            st.warning("无法解析测试集文件，请检查 JSON 格式。")
            cases = []
    else:
        st.warning(f"测试集文件不存在：`{test_set_path}`")
        cases = []

    # ── 运行评估 ──────────────────────────────────────────────────────────
    run_btn = st.button(
        "🚀 运行评估",
        disabled=(not cases),
        type="primary",
    )

    if run_btn:
        _run_evaluation(test_set_path, evaluator_type, collection, settings)

    # ── 历史结果 ──────────────────────────────────────────────────────────
    if "eval_history" in st.session_state and st.session_state["eval_history"]:
        st.divider()
        _render_history()


def _run_evaluation(test_set_path: str, evaluator_type: str, collection: str, settings) -> None:
    """执行评估并展示结果"""
    from src.observability.evaluation.eval_runner import EvalRunner

    evaluator = _build_evaluator(evaluator_type, settings)
    if evaluator is None:
        return

    with st.spinner("评估中…"):
        try:
            runner = EvalRunner(
                settings=settings,
                evaluator=evaluator,
                collection=collection,
            )
            report = runner.run(test_set_path)
        except Exception as exc:
            st.error(f"评估失败：{exc}")
            return

    # ── 指标汇总 ──────────────────────────────────────────────────────────
    st.subheader("📊 指标汇总")
    cols = st.columns(min(len(report.summary), 4) or 1)
    metric_items = list(report.summary.items())
    for i, col in enumerate(cols):
        if i < len(metric_items):
            k, v = metric_items[i]
            col.metric(label=k, value=f"{v:.4f}")

    st.caption(
        f"用例总数：{report.total_cases}  ·  "
        f"失败：{report.failed_cases}  ·  "
        f"耗时：{report.elapsed_ms:.0f} ms"
    )

    # ── 逐条结果 ──────────────────────────────────────────────────────────
    st.subheader("🔍 逐条结果")
    for r in report.per_query:
        status = "✅" if r.error is None else "❌"
        metrics_str = "  |  ".join(
            f"{k}={v:.3f}" for k, v in r.metrics.items()
            if not k.startswith("_")
        )
        query_preview = r.query[:70] + ("…" if len(r.query) > 70 else "")
        with st.expander(f"{status} {query_preview}  ·  {metrics_str}", expanded=False):
            if r.error:
                st.error(f"错误：{r.error}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**检索到的文档**")
                for src in r.retrieved_sources[:5]:
                    st.caption(f"• {src}")
            with col2:
                st.markdown("**期望文档**")
                for src in r.expected_sources:
                    hit = src in r.retrieved_sources
                    icon = "✅" if hit else "❌"
                    st.caption(f"{icon} {src}")

    # 保存到历史记录
    history = st.session_state.get("eval_history", [])
    history.insert(0, {
        "evaluator": evaluator_type,
        "summary": report.summary,
        "total": report.total_cases,
        "elapsed_ms": report.elapsed_ms,
    })
    st.session_state["eval_history"] = history[:10]  # 最多保留 10 条


def _build_evaluator(name: str, settings):
    """构建评估器，Ragas 不可用时优雅降级"""
    if name == "local":
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        return LocalRetrievalEvaluator(k=settings.retrieval.top_k_final)
    elif name == "ragas":
        try:
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            return RagasEvaluator()
        except ImportError as exc:
            st.error(str(exc))
            return None
    elif name == "composite":
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator
        evaluators = [LocalRetrievalEvaluator(k=settings.retrieval.top_k_final)]
        try:
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            evaluators.append(RagasEvaluator())
        except ImportError:
            st.warning("Ragas 未安装，composite 模式仅使用 local evaluator。")
        return CompositeEvaluator(evaluators)
    return None


def _render_history() -> None:
    """展示历史评估结果对比"""
    st.subheader("📅 历史评估记录")
    history = st.session_state["eval_history"]
    try:
        import pandas as pd
        rows = []
        for h in history:
            row = {"评估器": h["evaluator"], "用例数": h["total"],
                   "耗时(ms)": f"{h['elapsed_ms']:.0f}"}
            row.update({k: f"{v:.4f}" for k, v in h["summary"].items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        for i, h in enumerate(history):
            summary_str = "  ".join(f"{k}={v:.4f}" for k, v in h["summary"].items())
            st.caption(f"#{i+1} [{h['evaluator']}] {summary_str}")
