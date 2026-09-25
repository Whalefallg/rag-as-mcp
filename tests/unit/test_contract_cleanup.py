"""
Regression tests for runtime capability contracts.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import (
    EmbeddingConfig,
    LLMConfig,
    RerankConfig,
    RetrievalConfig,
    Settings,
    SplitterConfig,
    VectorStoreConfig,
    validate_settings,
)
from src.core.query_engine.fusion import (
    RRFusion,
    create_fusion,
    get_supported_algorithms,
)
from src.core.types import RetrievalResult
from src.ingestion.document_manager import DocumentManager


def _settings(
    *,
    llm="openai",
    embedding="openai",
    vector="chroma",
    splitter="recursive",
    sparse="bm25",
    fusion="rrf",
    rerank="none",
    top_m=30,
    top_k_final=10,
):
    return Settings(
        llm=LLMConfig(provider=llm, model="m"),
        embedding=EmbeddingConfig(provider=embedding, model="e"),
        vector_store=VectorStoreConfig(
            backend=vector,
            persist_path="./tmp",
        ),
        splitter=SplitterConfig(
            method=splitter,
            chunk_size=512,
            chunk_overlap=64,
        ),
        retrieval=RetrievalConfig(
            sparse_backend=sparse,
            fusion_algorithm=fusion,
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=top_k_final,
        ),
        rerank=RerankConfig(
            backend=rerank,
            top_m=top_m,
        ),
        raw_config={},
    )


def test_validate_rejects_unregistered_vector_backend():
    with pytest.raises(ValueError, match="vector_store backend"):
        validate_settings(_settings(vector="qdrant"))


def test_validate_rejects_unregistered_splitter_method():
    with pytest.raises(ValueError, match="splitter method"):
        validate_settings(_settings(splitter="semantic"))


def test_validate_rejects_unknown_sparse_backend():
    with pytest.raises(ValueError, match="sparse backend"):
        validate_settings(_settings(sparse="not-real"))


def test_validate_rejects_unknown_fusion_algorithm():
    with pytest.raises(ValueError, match="fusion algorithm"):
        validate_settings(_settings(fusion="weighted-sum"))


def test_validate_rejects_nonpositive_top_m():
    with pytest.raises(ValueError, match="top_m"):
        validate_settings(_settings(top_m=0))


def test_validate_rejects_nonpositive_final_top_k():
    with pytest.raises(ValueError, match="top_k_final"):
        validate_settings(_settings(top_k_final=0))


def test_fusion_factory_reports_real_capability():
    assert get_supported_algorithms() == ["rrf"]
    assert isinstance(create_fusion("rrf"), RRFusion)
    with pytest.raises(ValueError, match="fusion algorithm"):
        create_fusion("missing")


def test_query_tool_uses_top_m_as_rerank_candidate_window():
    from src.mcp_server.tools import query_knowledge_hub as tool

    calls = {}

    class Hybrid:
        def search(self, **kwargs):
            calls["search_top_k"] = kwargs["top_k"]
            return [
                RetrievalResult(
                    chunk_id=f"c{i}",
                    score=1.0,
                    text=f"text {i}",
                    metadata={},
                )
                for i in range(kwargs["top_k"])
            ]

    class Reranker:
        def rerank(self, **kwargs):
            calls["rerank_top_k"] = kwargs["top_k"]
            return kwargs["candidates"][: kwargs["top_k"]]

    class Assembler:
        def assemble(self, results, max_images=3):
            return []

    class Builder:
        def build(self, results, query, image_contents):
            calls["final_count"] = len(results)
            return [{"type": "text", "text": "ok"}]

    components = {
        "hybrid_search": Hybrid(),
        "reranker": Reranker(),
        "assembler": Assembler(),
        "builder": Builder(),
    }
    settings = _settings(rerank="cross_encoder", top_m=30)

    with patch.object(tool, "_get_components", return_value=components):
        result = tool.execute(
            {"query": "hello", "top_k": 5},
            settings,
        )

    assert result[0]["text"] == "ok"
    assert calls["search_top_k"] == 30
    assert calls["rerank_top_k"] == 5
    assert calls["final_count"] == 5


def test_query_tool_does_not_expand_candidates_when_rerank_disabled():
    from src.mcp_server.tools import query_knowledge_hub as tool

    calls = {}

    class Hybrid:
        def search(self, **kwargs):
            calls["search_top_k"] = kwargs["top_k"]
            return []

    class Assembler:
        def assemble(self, results, max_images=3):
            return []

    class Builder:
        def build(self, results, query, image_contents):
            return [{"type": "text", "text": "ok"}]

    components = {
        "hybrid_search": Hybrid(),
        "reranker": MagicMock(),
        "assembler": Assembler(),
        "builder": Builder(),
    }

    with patch.object(tool, "_get_components", return_value=components):
        tool.execute(
            {"query": "hello", "top_k": 7},
            _settings(rerank="none", top_m=30),
        )

    assert calls["search_top_k"] == 7


def _result(source, collection, cid):
    return RetrievalResult(
        chunk_id=cid,
        score=1.0,
        text="text",
        metadata={
            "source_path": source,
            "collection": collection,
        },
    )


def test_list_documents_keeps_same_source_in_separate_collections():
    chroma = MagicMock()
    chroma.list_collections.return_value = ["A", "B"]

    def by_metadata(*, filters, collection, limit):
        return [_result("/same.pdf", collection, f"{collection}-1")]

    chroma.get_by_metadata.side_effect = by_metadata

    dm = DocumentManager(
        chroma_store=chroma,
        bm25_indexer=MagicMock(),
        image_storage=MagicMock(),
        file_integrity=MagicMock(),
    )

    docs = dm.list_documents()
    assert len(docs) == 2
    assert {(d.collection, d.source_path) for d in docs} == {
        ("A", "/same.pdf"),
        ("B", "/same.pdf"),
    }


def test_delete_result_explicitly_reports_best_effort_partial_failure():
    chroma = MagicMock()
    chroma.delete_by_metadata.return_value = 2

    bm25 = MagicMock()
    bm25.remove_document.side_effect = RuntimeError("bm25 down")

    images = MagicMock()
    images.delete_by_source.return_value = 1

    integrity = MagicMock()
    integrity.remove_record.return_value = True

    dm = DocumentManager(
        chroma_store=chroma,
        bm25_indexer=bm25,
        image_storage=images,
        file_integrity=integrity,
    )

    result = dm.delete_document("/doc.pdf", "default")

    assert result.success is False
    assert result.details["mode"] == "best_effort"
    assert set(result.details["successful_stores"]) == {
        "chroma",
        "images",
        "integrity",
    }
    assert result.details["failed_stores"] == ["bm25"]
    assert "bm25" in result.error
