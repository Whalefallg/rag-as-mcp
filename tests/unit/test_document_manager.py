"""
DocumentManager 单元测试 (tests/unit/test_document_manager.py)
==============================================================
验收标准 (DEV_SPEC G2)：
  - list_documents 返回已摄入文档列表
  - delete_document 协调删除四个存储
  - 删除后 list 不包含已删除文档
  - 任一存储失败时 DeleteResult.success=False 且 error 有描述
  - get_collection_stats 返回正确统计
"""
import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.document_manager import DocumentManager, DocumentInfo, DeleteResult
from src.core.types import RetrievalResult


def _make_result(source_path: str, chunk_id: str = "c1", title: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text="text",
        score=0.9,
        metadata={"source_path": source_path, "title": title},
    )


def _make_dm(chroma_results=None, fail_chroma=False, fail_bm25=False,
             fail_images=False, fail_integrity=False) -> DocumentManager:
    chroma = MagicMock()
    if fail_chroma:
        chroma.get_by_metadata.side_effect = RuntimeError("chroma error")
        chroma.delete_by_metadata.side_effect = RuntimeError("chroma error")
    else:
        chroma.get_by_metadata.return_value = chroma_results or []
        chroma.delete_by_metadata.return_value = len(chroma_results) if chroma_results else 0
    chroma.list_collections.return_value = ["default"]

    bm25 = MagicMock()
    if fail_bm25:
        bm25.remove_document.side_effect = RuntimeError("bm25 error")

    images = MagicMock()
    if fail_images:
        images.delete_by_source.side_effect = RuntimeError("images error")
    else:
        images.delete_by_source.return_value = 0

    integrity = MagicMock()
    if fail_integrity:
        integrity.remove_record.side_effect = RuntimeError("integrity error")

    return DocumentManager(
        chroma_store=chroma,
        bm25_indexer=bm25,
        image_storage=images,
        file_integrity=integrity,
    )


class TestListDocuments:
    def test_returns_list_of_document_info(self):
        results = [
            _make_result("data/docs/a.pdf", "c1"),
            _make_result("data/docs/a.pdf", "c2"),
            _make_result("data/docs/b.pdf", "c3"),
        ]
        dm = _make_dm(chroma_results=results)
        docs = dm.list_documents("default")
        assert len(docs) == 2
        sources = {d.source_path for d in docs}
        assert "data/docs/a.pdf" in sources
        assert "data/docs/b.pdf" in sources

    def test_chunk_count_aggregated(self):
        results = [_make_result("doc.pdf", f"c{i}") for i in range(5)]
        dm = _make_dm(chroma_results=results)
        docs = dm.list_documents("default")
        assert docs[0].chunk_count == 5

    def test_empty_collection_returns_empty(self):
        dm = _make_dm(chroma_results=[])
        docs = dm.list_documents("default")
        assert docs == []

    def test_chroma_failure_returns_empty(self):
        dm = _make_dm(fail_chroma=True)
        docs = dm.list_documents("default")
        assert docs == []

    def test_sorted_by_source_path(self):
        results = [
            _make_result("z_doc.pdf", "c1"),
            _make_result("a_doc.pdf", "c2"),
        ]
        dm = _make_dm(chroma_results=results)
        docs = dm.list_documents("default")
        assert docs[0].source_path == "a_doc.pdf"
        assert docs[1].source_path == "z_doc.pdf"


class TestDeleteDocument:
    def test_successful_delete_returns_success_true(self):
        results = [_make_result("doc.pdf", "c1"), _make_result("doc.pdf", "c2")]
        dm = _make_dm(chroma_results=results)
        result = dm.delete_document("doc.pdf", "default")
        assert result.success is True
        assert result.deleted_chunks == 2

    def test_calls_all_four_stores(self):
        dm = _make_dm()
        dm.delete_document("doc.pdf", "default")
        dm._chroma.delete_by_metadata.assert_called_once()
        dm._bm25.remove_document.assert_called_once_with("doc.pdf")
        dm._images.delete_by_source.assert_called_once_with("doc.pdf")
        dm._integrity.remove_record.assert_called_once_with("doc.pdf")

    def test_chroma_failure_returns_error(self):
        dm = _make_dm(fail_chroma=True)
        result = dm.delete_document("doc.pdf", "default")
        assert result.success is False
        assert "chroma" in result.error

    def test_bm25_failure_returns_error(self):
        dm = _make_dm(fail_bm25=True)
        result = dm.delete_document("doc.pdf", "default")
        assert result.success is False
        assert "bm25" in result.error

    def test_partial_failure_accumulates_errors(self):
        dm = _make_dm(fail_bm25=True, fail_images=True)
        result = dm.delete_document("doc.pdf", "default")
        assert result.success is False
        assert "bm25" in result.error
        assert "images" in result.error

    def test_integrity_deleted_last(self):
        """FileIntegrity 应最后删除"""
        call_order = []
        dm = _make_dm()
        dm._chroma.delete_by_metadata.side_effect = lambda **kw: call_order.append("chroma") or 1
        dm._bm25.remove_document.side_effect = lambda p: call_order.append("bm25")
        dm._images.delete_by_source.side_effect = lambda p: call_order.append("images") or 0
        dm._integrity.remove_record.side_effect = lambda p: call_order.append("integrity")

        dm.delete_document("doc.pdf", "default")
        assert call_order[-1] == "integrity"


class TestCollectionStats:
    def test_returns_stats_list(self):
        results = [_make_result("a.pdf", "c1"), _make_result("b.pdf", "c2")]
        dm = _make_dm(chroma_results=results)
        stats = dm.get_collection_stats("default")
        assert len(stats) == 1
        assert stats[0].collection == "default"
        assert stats[0].chunk_count == 2
        assert stats[0].document_count == 2

    def test_chroma_failure_returns_zero_stats(self):
        dm = _make_dm(fail_chroma=True)
        stats = dm.get_collection_stats("default")
        assert stats[0].chunk_count == 0
