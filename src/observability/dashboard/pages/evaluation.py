"""
评估占位页面 (src/observability/dashboard/pages/evaluation.py)
==============================================================
为什么需要这个文件：
  DEV_SPEC 要求阶段 G 完成后此页面显示"评估模块尚未启用"的占位提示，
  Phase H 实现 RagasEvaluator 后再替换为完整功能。
  占位页面的价值：保持六页面架构完整，让 Dashboard 导航不出现空洞。
"""
import streamlit as st


def render() -> None:
    st.title("📈 评估面板")
    st.info(
        "**评估模块尚未启用。**\n\n"
        "此页面将在 **Phase H** 实现以下功能：\n"
        "- Faithfulness（忠实度）评估\n"
        "- Answer Relevancy（答案相关性）评估\n"
        "- Context Precision（上下文精准度）评估\n"
        "- 黄金测试集回归对比\n\n"
        "完成 Phase H 后，运行 `pytest tests/e2e/test_evaluation.py` 触发评估流程。"
    )
    st.caption("依赖：`pip install ragas`（Phase H 添加到 requirements.txt）")
