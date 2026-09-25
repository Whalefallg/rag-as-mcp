"""Regression tests for Batch A1 document replacement semantics."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.pipeline import IngestionPipeline, PipelineError
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore


def _chunk(chunk_id: str, source: str = "/fake/doc.pdf") -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="doc",
        text=chunk_id,
        index=0,
        metadata={
            "source_path": source,
            "dense_vector": [0.1, 0.2],
            "sparse_vector": {chunk_id: 1.0},
        },
    )


def _pipeline(chunks):
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._settings = object()
    pipeline._collection = "A"
    pipeline._legacy_progress = lambda progress: None
    pipeline._on_progress = None

    pipeline._integrity = MagicMock()
    pipeline._integrity.compute_sha256.return_value = "hash-v2"
    pipeline._integrity.should_skip.return_value = False

    document = SimpleNamespace(id="doc", metadata={"images": []})
    pipeline._loader = MagicMock()
    pipeline._loader.load.return_value = document

    pipeline._chunker = MagicMock()
    pipeline._chunker.split_document.return_value = chunks

    pipeline._refiner = MagicMock()
    pipeline._refiner.transform.side_effect = lambda current, trace: current
    pipeline._enricher = MagicMock()
    pipeline._enricher.transform.side_effect = lambda current, trace: current
    pipeline._captioner = MagicMock()
    pipeline._captioner.transform.side_effect = lambda current, trace: current
    pipeline._batch = MagicMock()
    pipeline._batch.encode_all.side_effect = lambda current, trace: current

    pipeline._upserter = MagicMock()
    pipeline._upserter.list_source_ids.return_value = []
    pipeline._bm25 = MagicMock()
    pipeline._img_store = MagicMock()
    return pipeline


def test_pipeline_replaces_only_stale_chunk_ids():
    pipeline = _pipeline([_chunk("keep"), _chunk("new")])
    pipeline._upserter.list_source_ids.return_value = ["old", "keep"]

    order = []
    pipeline._upserter.upsert.side_effect = lambda *a, **k: order.append("vector_upsert")
    pipeline._bm25.update.side_effect = lambda *a, **k: order.append("bm25_update")
    pipeline._bm25.remove_chunks.side_effect = lambda *a, **k: order.append("bm25_cleanup")
    pipeline._upserter.delete_ids.side_effect = lambda *a, **k: order.append("vector_cleanup")

    result = pipeline.run(
        "/fake/doc.pdf",
        trace=TraceContext(trace_type="ingestion"),
        collect_trace=False,
    )

    pipeline._bm25.remove_chunks.assert_called_once_with(["old"], collection="A")
    pipeline._upserter.delete_ids.assert_called_once_with(["old"], collection="A")
    assert order == ["vector_upsert", "bm25_update", "bm25_cleanup", "vector_cleanup"]
    assert result["stale_chunk_count"] == 1


def test_pipeline_does_not_delete_old_data_when_bm25_write_fails():
    pipeline = _pipeline([_chunk("new")])
    pipeline._upserter.list_source_ids.return_value = ["old"]
    pipeline._bm25.update.side_effect = RuntimeError("bm25 write failed")

    with pytest.raises(PipelineError, match="upsert"):
        pipeline.run(
            "/fake/doc.pdf",
            trace=TraceContext(trace_type="ingestion"),
            collect_trace=False,
        )

    pipeline._bm25.remove_chunks.assert_not_called()
    pipeline._upserter.delete_ids.assert_not_called()
    pipeline._integrity.mark_failed.assert_called_once_with(
        "hash-v2", "pipeline_error", collection="A"
    )


def test_pipeline_does_not_delete_vector_stale_ids_if_bm25_cleanup_fails():
    pipeline = _pipeline([_chunk("new")])
    pipeline._upserter.list_source_ids.return_value = ["old"]
    pipeline._bm25.remove_chunks.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(PipelineError, match="upsert"):
        pipeline.run(
            "/fake/doc.pdf",
            trace=TraceContext(trace_type="ingestion"),
            collect_trace=False,
        )

    pipeline._upserter.delete_ids.assert_not_called()


def test_pipeline_retry_converges_after_partial_new_write():
    pipeline = _pipeline([_chunk("new")])
    # 第一次 Chroma 已写入 new，但 BM25 失败；重试时 Chroma 中是 old + new。
    pipeline._upserter.list_source_ids.side_effect = [
        ["old"],
        ["old", "new"],
    ]
    pipeline._bm25.update.side_effect = [RuntimeError("first failure"), None]

    with pytest.raises(PipelineError):
        pipeline.run(
            "/fake/doc.pdf",
            trace=TraceContext(trace_type="ingestion"),
            collect_trace=False,
        )

    result = pipeline.run(
        "/fake/doc.pdf",
        trace=TraceContext(trace_type="ingestion"),
        collect_trace=False,
    )

    pipeline._bm25.remove_chunks.assert_called_once_with(["old"], collection="A")
    pipeline._upserter.delete_ids.assert_called_once_with(["old"], collection="A")
    assert result["stale_chunk_count"] == 1


def test_pipeline_encode_failure_never_starts_replacement():
    pipeline = _pipeline([_chunk("new")])
    pipeline._batch.encode_all.side_effect = RuntimeError("encode failed")

    with pytest.raises(PipelineError, match="encode"):
        pipeline.run(
            "/fake/doc.pdf",
            trace=TraceContext(trace_type="ingestion"),
            collect_trace=False,
        )

    pipeline._upserter.list_source_ids.assert_not_called()
    pipeline._upserter.upsert.assert_not_called()


def test_bm25_remove_chunks_is_collection_scoped(tmp_path):
    indexer = BM25Indexer(str(tmp_path / "bm25"))
    indexer.update([_chunk("shared")], collection="A")
    indexer.update([_chunk("shared")], collection="B")

    assert indexer.remove_chunks(["shared"], collection="A") == 1
    assert indexer.query({"shared": 1.0}, collection="A") == []
    assert indexer.query({"shared": 1.0}, collection="B")[0]["chunk_id"] == "shared"


def test_integrity_success_replaces_previous_version_for_same_path(tmp_path):
    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    checker.mark_success("hash-v1", "/doc.pdf", collection="A")
    checker.mark_success("hash-v2", "/doc.pdf", collection="A")

    assert checker.should_skip("hash-v1", collection="A") is False
    assert checker.should_skip("hash-v2", collection="A") is True


def test_chroma_delete_by_ids_deletes_only_existing_ids():
    class FakeCollection:
        def __init__(self):
            self.deleted = None

        def get(self, ids, include):
            return {"ids": [item for item in ids if item != "missing"]}

        def delete(self, ids):
            self.deleted = ids

    collection = FakeCollection()
    store = ChromaStore.__new__(ChromaStore)
    store.get_or_create_collection = lambda name="default": collection

    count = ChromaStore.delete_by_ids(
        store, ["a", "missing", "b"], collection_name="A"
    )

    assert count == 2
    assert collection.deleted == ["a", "b"]
