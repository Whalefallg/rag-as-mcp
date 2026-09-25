"""
关键抽象契约测试 (tests/unit/test_contracts.py)
================================================
验收标准 (DEV_SPEC I4)：
  VectorStore / Reranker / Evaluator / DocumentManager 接口形状契约

  VectorStore 测试使用 Mock ChromaDB client（chromadb 可能未安装）。
  Reranker factory 测试通过传入完整 Settings 对象调用。
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from src.core.types import RetrievalResult


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_settings(rerank_backend="none"):
    from src.core.settings import (
        Settings, LLMConfig, EmbeddingConfig,
        VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig,
    )
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="k"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="/tmp/tc"),
        splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend=rerank_backend),
        raw_config={},
    )


def _make_mock_chroma_client():
    """构造 mock chromadb client + collection，无需真实安装"""
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_col.query.return_value = {
        "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col
    mock_client.list_collections.return_value = []
    return mock_client, mock_col


# ── VectorStore 契约（mock chromadb）─────────────────────────────────────────

class TestVectorStoreContract:
    """使用 mock chromadb client 测试接口契约，不需要真实安装"""

    @pytest.fixture
    def store_and_col(self, tmp_path):
        from src.libs.vector_store.chroma_store import ChromaStore
        store = ChromaStore(persist_path=str(tmp_path / "chroma"))
        mock_client, mock_col = _make_mock_chroma_client()
        store._client = mock_client
        return store, mock_col

    def test_delete_returns_int(self, store_and_col):
        store, mock_col = store_and_col
        mock_col.get.return_value = {"ids": []}
        result = store.delete_by_metadata(filters={"source_path": "no.pdf"}, collection="d")
        assert isinstance(result, int)

    def test_delete_no_match_returns_zero(self, store_and_col):
        store, mock_col = store_and_col
        mock_col.get.return_value = {"ids": []}
        assert store.delete_by_metadata(filters={"source_path": "ghost.pdf"}, collection="d") == 0

    def test_delete_empty_filter_returns_zero(self, store_and_col):
        store, _ = store_and_col
        assert store.delete_by_metadata(filters={}, collection="default") == 0

    def test_delete_legacy_filter_kwarg(self, store_and_col):
        store, mock_col = store_and_col
        mock_col.get.return_value = {"ids": []}
        result = store.delete_by_metadata(filter={"source_path": "x.pdf"}, collection_name="d")
        assert isinstance(result, int)

    def test_delete_with_match_returns_count(self, store_and_col):
        store, mock_col = store_and_col
        mock_col.get.return_value = {"ids": ["c1", "c2"]}
        result = store.delete_by_metadata(filters={"source_path": "del.pdf"}, collection="d")
        assert result == 2

    def test_get_by_metadata_returns_list(self, store_and_col):
        store, _ = store_and_col
        assert isinstance(store.get_by_metadata(filters={}, collection="default"), list)

    def test_list_collections_returns_list(self, store_and_col):
        store, _ = store_and_col
        assert isinstance(store.list_collections(), list)

    def test_upsert_calls_chroma_upsert(self, store_and_col):
        store, mock_col = store_and_col
        store.upsert([{
            "id": "c1", "text": "hello",
            "metadata": {"source_path": "t.pdf", "collection": "ct"},
            "dense_vector": [0.1] * 384,
        }])
        mock_col.upsert.assert_called_once()

    def test_get_by_ids_returns_list(self, store_and_col):
        store, mock_col = store_and_col
        mock_col.get.return_value = {
            "ids": ["c1"], "documents": ["text"], "metadatas": [{}]
        }
        results = store.get_by_ids(["c1"], collection_name="ct")
        assert isinstance(results, list)
        assert len(results) == 1 and results[0]["id"] == "c1"

    def test_delete_after_upsert_returns_nonzero(self, store_and_col):
        store, mock_col = store_and_col
        store.upsert([{
            "id": "del_c1", "text": "bye",
            "metadata": {"source_path": "del.pdf", "collection": "del_col"},
            "dense_vector": [0.2] * 384,
        }])
        mock_col.get.return_value = {"ids": ["del_c1"]}
        count = store.delete_by_metadata(filters={"source_path": "del.pdf"}, collection="del_col")
        assert count == 1


# ── Reranker 契约 ─────────────────────────────────────────────────────────────

class TestRerankerContract:

    def test_rerank_returns_list(self):
        from src.core.query_engine.reranker import CoreReranker
        mock_be = MagicMock()
        mock_be.rerank.return_value = [0, 1]
        r = CoreReranker(_make_settings(), reranker=mock_be)
        result = r.rerank("q", [RetrievalResult("c1", 0.9, "t", {}),
                                 RetrievalResult("c2", 0.8, "t", {})])
        assert isinstance(result, list)

    def test_rerank_empty_input_returns_empty(self):
        from src.core.query_engine.reranker import CoreReranker
        mock_be = MagicMock()
        mock_be.rerank.return_value = []
        r = CoreReranker(_make_settings(), reranker=mock_be)
        assert r.rerank("q", []) == []

    def test_rerank_count_lte_input(self):
        from src.core.query_engine.reranker import CoreReranker
        mock_be = MagicMock()
        mock_be.rerank.return_value = [0]
        r = CoreReranker(_make_settings(), reranker=mock_be)
        candidates = [RetrievalResult(f"c{i}", 0.9, "t", {}) for i in range(5)]
        assert len(r.rerank("q", candidates)) <= 5

    def test_rerank_results_are_retrieval_result_instances(self):
        from src.core.query_engine.reranker import CoreReranker
        mock_be = MagicMock()
        mock_be.rerank.return_value = [1, 0]
        r = CoreReranker(_make_settings(), reranker=mock_be)
        candidates = [RetrievalResult("c1", 0.9, "t1", {}),
                      RetrievalResult("c2", 0.8, "t2", {})]
        for item in r.rerank("q", candidates):
            assert isinstance(item, RetrievalResult)

    def test_rerank_fallback_returns_retrieval_results(self):
        from src.core.query_engine.reranker import CoreReranker
        mock_be = MagicMock()
        mock_be.rerank.side_effect = RuntimeError("down")
        r = CoreReranker(_make_settings(), reranker=mock_be)
        candidates = [RetrievalResult("c1", 0.9, "t", {})]
        result = r.rerank("q", candidates)
        assert len(result) >= 1
        assert all(isinstance(x, RetrievalResult) for x in result)

    def test_reranker_factory_none_backend(self):
        """create_reranker 接受 Settings 对象，backend=none 应正常创建"""
        from src.libs.reranker.reranker_factory import create_reranker
        reranker = create_reranker(_make_settings("none"))
        assert reranker is not None

    def test_reranker_factory_unknown_backend_raises(self):
        from src.libs.reranker.reranker_factory import create_reranker
        with pytest.raises(ValueError):
            create_reranker(_make_settings("__nonexistent__"))


# ── Evaluator 契约 ────────────────────────────────────────────────────────────

class TestEvaluatorContract:

    def _chunk(self, cid, source=""):
        return {"id": cid, "text": "t", "score": 0.9, "metadata": {"source_path": source}}

    def test_local_required_keys_present(self):
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        ev = LocalRetrievalEvaluator(k=5)
        result = ev.evaluate("q", [self._chunk("c1")],
                             ground_truth={"expected_chunk_ids": ["c1"]})
        for key in ("hit_rate", "mrr", "precision_at_k"):
            assert key in result and isinstance(result[key], float)

    def test_local_values_in_unit_range(self):
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        ev = LocalRetrievalEvaluator(k=10)
        result = ev.evaluate("q", [self._chunk(f"c{i}") for i in range(5)],
                             ground_truth={"expected_chunk_ids": ["c2"]})
        for k in ("hit_rate", "mrr", "precision_at_k"):
            assert 0.0 <= result[k] <= 1.0

    def test_empty_input_returns_zeros(self):
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        ev = LocalRetrievalEvaluator(k=10)
        result = ev.evaluate("q", [], ground_truth=None)
        assert isinstance(result, dict)
        assert result["hit_rate"] == 0.0

    def test_composite_merges_keys(self):
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        from src.libs.evaluator.base_evaluator import BaseEvaluator

        class ExtraEv(BaseEvaluator):
            def evaluate(self, query, retrieved_chunks, **kw):
                return {"custom_score": 0.75}

        comp = CompositeEvaluator([LocalRetrievalEvaluator(k=5), ExtraEv()])
        result = comp.evaluate("q", [])
        assert "hit_rate" in result and "custom_score" in result

    def test_factory_create_local(self):
        from src.libs.evaluator.evaluator_factory import create_evaluator
        import src.observability.evaluation.local_retrieval_evaluator  # noqa
        ev = create_evaluator("local")
        assert isinstance(ev.evaluate("q", []), dict)

    def test_factory_unknown_raises(self):
        from src.libs.evaluator.evaluator_factory import create_evaluator
        with pytest.raises(ValueError):
            create_evaluator("__nonexistent__")


# ── DocumentManager 契约 ──────────────────────────────────────────────────────

class TestDocumentManagerContract:

    def _make_dm(self, fail=False):
        from src.ingestion.document_manager import DocumentManager
        chroma = MagicMock()
        if fail:
            chroma.get_by_metadata.side_effect = RuntimeError("fail")
            chroma.delete_by_metadata.side_effect = RuntimeError("fail")
        else:
            chroma.get_by_metadata.return_value = []
            chroma.delete_by_metadata.return_value = 0
        chroma.list_collections.return_value = ["default"]
        bm25 = MagicMock()
        images = MagicMock()
        images.delete_by_source.return_value = 0
        integrity = MagicMock()
        return DocumentManager(chroma, bm25, images, integrity)

    def test_list_documents_returns_list(self):
        assert isinstance(self._make_dm().list_documents("default"), list)

    def test_list_documents_empty_on_failure(self):
        assert self._make_dm(fail=True).list_documents("default") == []

    def test_delete_returns_delete_result(self):
        from src.ingestion.document_manager import DeleteResult
        result = self._make_dm().delete_document("doc.pdf", "default")
        assert isinstance(result, DeleteResult)
        assert hasattr(result, "success")
        assert hasattr(result, "source_path")
        assert hasattr(result, "deleted_chunks")

    def test_delete_success_true_on_no_error(self):
        assert self._make_dm().delete_document("d.pdf", "default").success is True

    def test_delete_success_false_on_error(self):
        result = self._make_dm(fail=True).delete_document("d.pdf", "default")
        assert result.success is False
        assert result.error is not None

    def test_delete_source_path_preserved(self):
        result = self._make_dm().delete_document("specific/path/doc.pdf", "default")
        assert result.source_path == "specific/path/doc.pdf"

    def test_get_collection_stats_returns_list(self):
        from src.ingestion.document_manager import CollectionStats
        stats = self._make_dm().get_collection_stats("default")
        assert isinstance(stats, list)
        for s in stats:
            assert isinstance(s, CollectionStats)
