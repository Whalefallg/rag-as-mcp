"""Offline evaluation guidance page."""
import streamlit as st


def render() -> None:
    st.title("📈 评估面板")
    st.info(
        "此轻量页面未启用交互式评估。请使用 "
        "`python scripts/evaluate.py --mode compare` 运行离线检索评估。"
    )
    st.caption("LLM-based evaluation requires `pip install -e '.[evaluation]'`.")
