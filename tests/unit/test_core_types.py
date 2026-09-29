"""核心数据类型测试 (tests/unit/test_core_types.py)"""
import pytest
from dataclasses import asdict
from src.core.types import Document, Chunk, ChunkRecord, ImageRef, DocumentInfo, DeleteResult, IngestionProgress


def test_document_asdict():
    doc = Document(id="abc", source="/path/doc.pdf", text="hello", metadata={"source_path": "/path/doc.pdf"})
    d = asdict(doc)
    assert d["id"] == "abc"
    assert d["metadata"]["source_path"] == "/path/doc.pdf"


def test_chunk_asdict():
    chunk = Chunk(id="abc_0001_deadbeef", doc_id="abc", text="hello", index=0,
                  metadata={"source_path": "/a.pdf", "chunk_index": 0})
    d = asdict(chunk)
    assert d["index"] == 0
    assert d["metadata"]["chunk_index"] == 0


def test_chunk_record_has_vectors():
    record = ChunkRecord(
        id="abc", text="hello", metadata={},
        dense_vector=[0.1, 0.2], sparse_vector={"hello": 0.5}
    )
    d = asdict(record)
    assert d["dense_vector"] == [0.1, 0.2]
    assert d["sparse_vector"]["hello"] == 0.5


def test_image_ref_optional_fields():
    ref = ImageRef(image_id="img1", file_path="/p.png", page=0, seq=0)
    assert ref.width is None
    assert ref.caption is None


def test_document_default_metadata():
    doc = Document(id="x", source="/x.pdf", text="")
    assert doc.metadata == {}


def test_chunk_default_metadata():
    chunk = Chunk(id="x", doc_id="y", text="", index=0)
    assert chunk.metadata == {}


def test_chunk_record_default_vectors():
    record = ChunkRecord(id="x", text="", metadata={})
    assert record.dense_vector == []
    assert record.sparse_vector == {}


def test_ingestion_progress_fields():
    p = IngestionProgress(stage="load", current=1, total=5, message="Loading")
    assert p.stage == "load"
    assert p.total == 5


def test_delete_result_fields():
    r = DeleteResult(source="/a.pdf", chunks_deleted=3, bm25_removed=True, images_deleted=2, history_removed=True)
    assert r.chunks_deleted == 3
