"""
DocumentChunker 单元测试 (tests/unit/test_document_chunker.py)
==============================================================
为什么需要这个文件：
  DocumentChunker 是 Ingestion Pipeline 中最关键的一步——
  它决定了向量库里存储的基本单元的形状：Chunk ID 必须稳定（防止重复摄取产生新 ID），
  metadata 必须完整继承（检索时需要知道 chunk 来自哪个文件）。
  图片分发逻辑验证 image_refs 正确归属到包含占位符的 chunk，
  确保多模态内容在切分后不会丢失。

验收标准（DEV_SPEC C4）：
  - Chunk ID 唯一且确定性（重复切分同一 Document 产生相同 ID）
  - 元数据继承：Chunk.metadata 包含 Document.metadata 的所有字段
  - chunk_index 正确（从 0 开始连续）
  - 图片分发：含 [IMAGE: id] 的 chunk 有 image_refs，无占位符的 chunk 无 image_refs
  - 空文档返回空列表
"""
import pytest
from src.core.types import Document, Chunk
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
from src.libs.splitter.splitter_factory import register_splitter
from src.libs.splitter.base_splitter import BaseSplitter
from src.ingestion.chunking.document_chunker import DocumentChunker


# ── Fake Splitter（固定切分，隔离外部依赖）────────────────────────────────────

@register_splitter("test_chunker")
class FixedSplitter(BaseSplitter):
    """按 '\n---\n' 切分，方便测试中精确控制 chunk 内容"""
    def split_text(self, text, trace=None):
        if not text or not text.strip():
            return []
        parts = [p.strip() for p in text.split("\n---\n") if p.strip()]
        return parts


def _make_settings():
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        splitter=SplitterConfig(method="test_chunker", chunk_size=1000, chunk_overlap=0),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def _make_doc(text: str, metadata: dict = None) -> Document:
    return Document(
        id="testdoc001",
        source="/path/test.pdf",
        text=text,
        metadata={"source_path": "/path/test.pdf", "doc_type": "pdf", **(metadata or {})}
    )


# ── 基本功能测试 ──────────────────────────────────────────────────────────────

def test_split_produces_chunks():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("chunk one\n---\nchunk two\n---\nchunk three")
    chunks = chunker.split_document(doc)
    assert len(chunks) == 3


def test_empty_document_returns_empty_list():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("")
    chunks = chunker.split_document(doc)
    assert chunks == []


def test_chunk_index_starts_at_zero():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("a\n---\nb\n---\nc")
    chunks = chunker.split_document(doc)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunk_index_in_metadata():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("a\n---\nb")
    chunks = chunker.split_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


# ── ID 唯一性与确定性 ─────────────────────────────────────────────────────────

def test_chunk_ids_are_unique():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("a\n---\nb\n---\nc")
    chunks = chunker.split_document(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_deterministic():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("hello world\n---\nfoo bar")
    chunks1 = chunker.split_document(doc)
    chunks2 = chunker.split_document(doc)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_chunk_id_format():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("only one chunk")
    chunks = chunker.split_document(doc)
    chunk_id = chunks[0].id
    # 格式：{doc_id}_{index:04d}_{hash8}
    parts = chunk_id.split("_")
    assert len(parts) == 3
    assert parts[1] == "0000"
    assert len(parts[2]) == 8


# ── 元数据继承 ────────────────────────────────────────────────────────────────

def test_metadata_inherits_source_path():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("content", {"title": "Test Doc"})
    chunks = chunker.split_document(doc)
    for chunk in chunks:
        assert chunk.metadata["source_path"] == "/path/test.pdf"
        assert chunk.metadata["title"] == "Test Doc"


def test_doc_id_propagated():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("a\n---\nb")
    chunks = chunker.split_document(doc)
    for chunk in chunks:
        assert chunk.doc_id == "testdoc001"


# ── 图片引用分发 ──────────────────────────────────────────────────────────────

def test_image_refs_distributed_to_correct_chunk():
    chunker = DocumentChunker(_make_settings())
    text = "text with image [IMAGE: img_001_0_0]\n---\ntext without image"
    doc = _make_doc(text, metadata={
        "images": [{"image_id": "img_001_0_0", "path": "/p.png", "page": 0, "seq": 0}]
    })
    chunks = chunker.split_document(doc)
    assert len(chunks) == 2
    # 第一个 chunk 有图片引用
    assert "image_refs" in chunks[0].metadata
    assert "img_001_0_0" in chunks[0].metadata["image_refs"]
    assert len(chunks[0].metadata["images"]) == 1
    # 第二个 chunk 没有图片引用
    assert "image_refs" not in chunks[1].metadata


def test_chunk_without_image_has_no_images_field():
    chunker = DocumentChunker(_make_settings())
    doc = _make_doc("no images here", metadata={"images": []})
    chunks = chunker.split_document(doc)
    assert "image_refs" not in chunks[0].metadata
    assert "images" not in chunks[0].metadata


def test_multiple_images_in_one_chunk():
    chunker = DocumentChunker(_make_settings())
    text = "img1 [IMAGE: img1] and img2 [IMAGE: img2]"
    doc = _make_doc(text, metadata={
        "images": [
            {"image_id": "img1", "path": "/1.png", "page": 0, "seq": 0},
            {"image_id": "img2", "path": "/2.png", "page": 0, "seq": 1},
        ]
    })
    chunks = chunker.split_document(doc)
    assert set(chunks[0].metadata["image_refs"]) == {"img1", "img2"}
