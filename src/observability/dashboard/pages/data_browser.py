"""
数据浏览器页面 (src/observability/dashboard/pages/data_browser.py)
===================================================================
为什么需要这个文件：
  摄取完文档后需要验证"数据是否正确进库"——chunk 内容是否正常、
  metadata 字段是否完整、图片是否关联正确。
  数据浏览器提供集合→文档→Chunk 的三级下钻视图。
"""
import streamlit as st


def render(data_service) -> None:
    st.title("🗂️ 数据浏览器")
    st.caption("浏览已摄取的文档、Chunk 详情与关联图片")

    # ── 集合选择 ──────────────────────────────────────────────────────────
    collections = data_service.list_collections()
    if not collections:
        st.info("知识库为空，请先在「Ingestion 管理」页面摄取文档。")
        return

    selected_col = st.selectbox("选择集合", collections)

    # ── 文档列表 ──────────────────────────────────────────────────────────
    docs = data_service.list_documents(selected_col)
    if not docs:
        st.info(f"集合 `{selected_col}` 中暂无文档。")
        return

    st.subheader(f"📄 文档列表（共 {len(docs)} 份）")

    # 搜索框
    search = st.text_input("🔍 按文件名搜索", placeholder="输入文件名关键词…")
    if search:
        docs = [d for d in docs if search.lower() in d.source_path.lower()]

    for doc in docs:
        basename = doc.source_path.replace("\\", "/").rsplit("/", 1)[-1]
        with st.expander(f"📄 {basename}  ·  {doc.chunk_count} chunks", expanded=False):
            st.caption(f"路径：`{doc.source_path}`")
            st.caption(f"集合：`{doc.collection}`")
            if doc.ingested_at:
                st.caption(f"摄取时间：{doc.ingested_at}")

            if st.button("加载 Chunk 详情", key=f"load_{doc.source_path}"):
                st.session_state[f"chunks_{doc.source_path}"] = \
                    data_service.get_chunks(doc.source_path, selected_col)

            chunks = st.session_state.get(f"chunks_{doc.source_path}", [])
            if chunks:
                st.markdown(f"**共 {len(chunks)} 个 Chunk**")
                for i, chunk in enumerate(chunks[:50]):   # 最多展示 50 个
                    with st.expander(f"Chunk {i+1}  `{chunk.chunk_id[:16]}…`",
                                     expanded=False):
                        st.text_area("内容", chunk.text, height=120,
                                     key=f"txt_{chunk.chunk_id}", disabled=True)
                        # Metadata 表格
                        meta_display = {
                            k: v for k, v in chunk.metadata.items()
                            if k not in ("dense_vector",)   # 跳过大字段
                        }
                        if meta_display:
                            st.json(meta_display, expanded=False)
                        # 图片预览
                        for img_id in chunk.metadata.get("image_refs", []):
                            img_path = data_service.get_image_path(img_id)
                            if img_path:
                                st.image(img_path, caption=img_id, width=300)
                if len(chunks) > 50:
                    st.caption(f"仅展示前 50 个 Chunk，共 {len(chunks)} 个。")
