"""
Regression tests for Phase 1A correctness hardening.

Covers:
  - Dense/Sparse physical collection routing
  - HybridSearch propagation of collection
  - BM25 incremental updates and collection isolation
  - VectorStore/BM25 chunk-id alignment
  - FileIntegrity collection scoping and legacy-schema migration
"""
import sqlite3
from types import SimpleNamespace

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.types import Chunk, ProcessedQuery
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


class _Embedding:
    def embed(self, texts, trace=None):
        return [[0.1, 0.2] for _ in texts]


def test_dense_routes_non_default_collection_to_store():
    class Store:
        def __init__(self):
            self.kwargs = None

        def query(self, **kwargs):
            self.kwargs = kwargs
            return []

    store = Store()
    retriever = DenseRetriever(None, _Embedding(), store)
    retriever.retrieve(
        "query",
        collection="knowledge_b",
        filters={"source_path": "/doc.pdf"},
    )

    assert store.kwargs["collection_name"] == "knowledge_b"
    assert store.kwargs["filters"] == {"source_path": "/doc.pdf"}


def test_sparse_routes_bm25_and_hydration_to_same_collection():
    class BM25:
        def __init__(self):
            self.collection = None

        def query(self, terms, top_k=10, collection="default"):
            self.collection = collection
            return [{"chunk_id": "c1", "score": 2.0}]

    class Store:
        def __init__(self):
            self.collection = None

        def get_by_ids(self, ids, collection_name="default"):
            self.collection = collection_name
            return [{"id": "c1", "text": "hit", "metadata": {}}]

    bm25 = BM25()
    store = Store()
    retriever = SparseRetriever(
        None, bm25_indexer=bm25, vector_store=store
    )
    results = retriever.retrieve(
        ["needle"], collection="knowledge_b"
    )

    assert bm25.collection == "knowledge_b"
    assert store.collection == "knowledge_b"
    assert [result.chunk_id for result in results] == ["c1"]


def test_hybrid_search_propagates_collection_to_sparse():
    class Processor:
        def process(self, query, filters=None):
            return ProcessedQuery(
                original=query,
                keywords=["query"],
                filters=filters or {},
            )

    class Dense:
        def __init__(self):
            self.collection = None

        def retrieve(self, **kwargs):
            self.collection = kwargs["collection"]
            return []

    class Sparse:
        def __init__(self):
            self.collection = None

        def retrieve(self, **kwargs):
            self.collection = kwargs.get("collection", "default")
            return []

    settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            top_k_dense=10,
            top_k_sparse=10,
            top_k_final=5,
        )
    )
    dense = Dense()
    sparse = Sparse()
    search = HybridSearch(
        settings,
        query_processor=Processor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    assert search.search("query", collection="knowledge_b") == []
    assert dense.collection == "knowledge_b"
    assert sparse.collection == "knowledge_b"


def _chunk(cid, source, terms):
    return Chunk(
        id=cid,
        doc_id="doc",
        text=" ".join(terms),
        index=0,
        metadata={
            "source_path": source,
            "sparse_vector": {term: 1.0 for term in terms},
        },
    )


def test_bm25_update_preserves_previous_documents(tmp_path):
    indexer = BM25Indexer(str(tmp_path / "bm25"))
    indexer.update(
        [_chunk("a1", "/a.pdf", ["alpha"])],
        collection="A",
    )
    indexer.update(
        [_chunk("b1", "/b.pdf", ["beta"])],
        collection="A",
    )

    assert indexer.query(
        {"alpha": 1.0}, collection="A"
    )[0]["chunk_id"] == "a1"
    assert indexer.query(
        {"beta": 1.0}, collection="A"
    )[0]["chunk_id"] == "b1"


def test_bm25_same_chunk_id_is_isolated_by_collection(tmp_path):
    indexer = BM25Indexer(str(tmp_path / "bm25"))
    indexer.update(
        [_chunk("shared", "/doc.pdf", ["alpha"])],
        collection="A",
    )
    indexer.update(
        [_chunk("shared", "/doc.pdf", ["beta"])],
        collection="B",
    )

    assert indexer.query(
        {"alpha": 1.0}, collection="A"
    )[0]["chunk_id"] == "shared"
    assert indexer.query({"alpha": 1.0}, collection="B") == []
    assert indexer.query(
        {"beta": 1.0}, collection="B"
    )[0]["chunk_id"] == "shared"


def test_vector_upserter_uses_business_chunk_id():
    class Store:
        def __init__(self):
            self.records = []

        def upsert(self, records, trace=None):
            self.records.extend(records)

    store = Store()
    upserter = VectorUpserter.__new__(VectorUpserter)
    upserter._store = store

    chunk = Chunk(
        id="doc_0000_deadbeef",
        doc_id="doc",
        text="content",
        index=0,
        metadata={
            "source_path": "/doc.pdf",
            "content_hash": "deadbeef" * 8,
            "dense_vector": [0.1, 0.2],
        },
    )

    upserter.upsert([chunk], collection="A")
    assert store.records[0]["id"] == chunk.id


def test_integrity_is_scoped_by_collection(tmp_path):
    checker = SQLiteIntegrityChecker(
        str(tmp_path / "history.db")
    )
    checker.mark_success(
        "hash-1", "/doc.pdf", collection="A"
    )

    assert checker.should_skip("hash-1", collection="A") is True
    assert checker.should_skip("hash-1", collection="B") is False

    checker.mark_success(
        "hash-1", "/doc.pdf", collection="B"
    )
    assert checker.should_skip("hash-1", collection="B") is True

    checker.remove_record("/doc.pdf", collection="A")
    assert checker.should_skip("hash-1", collection="A") is False
    assert checker.should_skip("hash-1", collection="B") is True


def test_integrity_migrates_legacy_schema_to_default_collection(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE ingestion_history (
                file_hash   TEXT PRIMARY KEY,
                file_path   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                error_msg   TEXT,
                ingested_at TEXT,
                updated_at  TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO ingestion_history (
                file_hash, file_path, status,
                ingested_at, updated_at
            )
            VALUES (
                'legacy-hash', '/legacy.pdf', 'success',
                '2026-01-01', '2026-01-01'
            )
            """
        )

    checker = SQLiteIntegrityChecker(str(db))
    assert checker.should_skip(
        "legacy-hash", collection="default"
    ) is True
    assert checker.should_skip(
        "legacy-hash", collection="other"
    ) is False
