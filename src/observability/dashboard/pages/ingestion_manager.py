"""
Ingestion 管理页面 (src/observability/dashboard/pages/ingestion_manager.py)
==========================================================================
为什么需要这个文件：
  提供文件上传 → 触发摄取 → 实时进度 → 完成后浏览结果的完整闭环。
  进度条由 Pipeline 的 on_progress 回调驱动，在同一个 Streamlit request 内
  同步更新（Streamlit 的 rerun 机制天然支持）。
"""
import tempfile
from pathlib import Path

import streamlit as st


def render(data_service, settings) -> None:
    st.title("📥 Ingestion 管理")
    st.caption("上传文档、触发摄取、管理已有文档")

    tab_upload, tab_manage = st.tabs(["📤 上传摄取", "🗑️ 文档管理"])

    # ── Tab 1: 上传并摄取 ─────────────────────────────────────────────────
    with tab_upload:
        collections = data_service.list_collections() or ["default"]
        col_name = st.selectbox("目标集合", collections + ["（新建）default"],
                                key="ingest_collection")
        if col_name == "（新建）default":
            col_name = st.text_input("输入新集合名称", value="default")

        force = st.checkbox("强制重新摄取（跳过去重检查）", value=False)
        uploaded = st.file_uploader(
            "选择 PDF 文件（支持多文件）",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("🚀 开始摄取", disabled=not uploaded):
            _run_ingestion(uploaded, col_name, force, settings)

    # ── Tab 2: 文档管理 ───────────────────────────────────────────────────
    with tab_manage:
        _render_document_list(data_service, settings)


def _run_ingestion(uploaded_files, collection: str, force: bool, settings) -> None:
    """执行摄取并展示实时进度"""
    from src.ingestion.pipeline import IngestionPipeline

    progress_bar = st.progress(0, text="准备中…")
    status_text = st.empty()
    results = []

    _STAGE_LABELS = {
        "load": "加载文档",
        "split": "切分 Chunk",
        "transform": "清洗 & 增强",
        "encode": "向量编码",
        "upsert": "写入向量库",
        "store_images": "存储图片",
        "integrity": "完整性检查",
    }
    _STAGE_WEIGHTS = {
        "integrity": 0.05, "load": 0.10, "split": 0.15,
        "transform": 0.30, "encode": 0.25, "upsert": 0.10, "store_images": 0.05,
    }
    _stage_order = list(_STAGE_WEIGHTS.keys())
    _stage_done: list = []

    def _on_progress(stage: str, current: int, total: int) -> None:
        if current == total and stage not in _stage_done:
            _stage_done.append(stage)
        done_weight = sum(_STAGE_WEIGHTS.get(s, 0) for s in _stage_done)
        pct = min(done_weight, 1.0)
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(pct, text=f"{label}… ({current}/{total})")

    pipeline = IngestionPipeline(
        settings=settings,
        collection=collection,
        on_progress=_on_progress,
    )

    for uf in uploaded_files:
        _stage_done.clear()
        status_text.info(f"处理：{uf.name}")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        try:
            result = pipeline.run(tmp_path, force=force)
            result["filename"] = uf.name
            results.append(result)
        except Exception as exc:
            results.append({"filename": uf.name, "error": str(exc)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    progress_bar.progress(1.0, text="完成！")
    status_text.empty()

    # 结果汇报
    st.subheader("摄取结果")
    for r in results:
        if r.get("error"):
            st.error(f"❌ {r['filename']}：{r['error']}")
        elif r.get("skipped"):
            st.warning(f"⏭️ {r['filename']}：已跳过（{r.get('reason', '')}）")
        else:
            st.success(
                f"✅ {r['filename']}：{r.get('chunk_count', 0)} chunks，"
                f"{r.get('image_count', 0)} 张图片"
            )


def _render_document_list(data_service, settings) -> None:
    """展示文档列表并提供删除功能"""
    collections = data_service.list_collections()
    if not collections:
        st.info("知识库为空。")
        return

    selected = st.selectbox("集合", collections, key="manage_collection")
    docs = data_service.list_documents(selected)

    if not docs:
        st.info(f"集合 `{selected}` 中暂无文档。")
        return

    st.markdown(f"共 **{len(docs)}** 份文档")
    for doc in docs:
        basename = doc.source_path.replace("\\", "/").rsplit("/", 1)[-1]
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"📄 **{basename}**  ·  {doc.chunk_count} chunks")
            st.caption(f"`{doc.source_path}`")
        with c2:
            if st.button("删除", key=f"del_{doc.source_path}", type="secondary"):
                _delete_document(doc.source_path, selected, settings)
                st.rerun()


def _delete_document(source_path: str, collection: str, settings) -> None:
    try:
        from src.ingestion.document_manager import DocumentManager
        dm = DocumentManager.from_settings(settings)
        result = dm.delete_document(source_path, collection)
        if result.success:
            st.success(f"已删除：{source_path}")
        else:
            st.error(f"删除失败：{result.error}")
    except Exception as exc:
        st.error(f"删除出错：{exc}")
