"""
Regression tests for collection-aware image lifecycle.
"""
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.types import RetrievalResult
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter


PNG_A = b"\x89PNG\r\n\x1a\n" + b"A" * 20
PNG_B = b"\x89PNG\r\n\x1a\n" + b"B" * 20


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(
        storage_root=str(tmp_path / "images"),
        db_path=str(tmp_path / "image_index.db"),
    )


def test_same_image_id_can_exist_in_two_collections(storage):
    path_a = storage.save(
        "shared", PNG_A, collection="A", source_path="/doc.pdf"
    )
    path_b = storage.save(
        "shared", PNG_B, collection="B", source_path="/doc.pdf"
    )

    assert path_a != path_b
    assert Path(storage.get_path("shared", collection="A")).read_bytes() == PNG_A
    assert Path(storage.get_path("shared", collection="B")).read_bytes() == PNG_B


def test_delete_by_source_is_collection_isolated(storage):
    storage.save(
        "a1", PNG_A, collection="A", source_path="/doc.pdf"
    )
    storage.save(
        "b1", PNG_B, collection="B", source_path="/doc.pdf"
    )

    assert storage.delete_by_source("/doc.pdf", collection="A") == 1
    assert storage.get_path("a1", collection="A") is None
    assert storage.get_path("b1", collection="B") is not None


def test_legacy_schema_migrates_and_falls_back_to_doc_hash(tmp_path):
    db = tmp_path / "legacy.db"
    root = tmp_path / "images"
    root.mkdir()
    old_file = root / "legacy.png"
    old_file.write_bytes(PNG_A)

    import hashlib
    source = "/legacy.pdf"
    doc_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE image_index (
                image_id   TEXT PRIMARY KEY,
                file_path  TEXT NOT NULL,
                collection TEXT,
                doc_hash   TEXT,
                page_num   INTEGER,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO image_index (
                image_id, file_path, collection,
                doc_hash, page_num, created_at
            )
            VALUES (?, ?, 'default', ?, 1, '2026-01-01')
            """,
            ("legacy", str(old_file), doc_hash),
        )

    migrated = ImageStorage(storage_root=str(root), db_path=str(db))
    rows = migrated.list_by_source(source, collection="default")
    assert [row["image_id"] for row in rows] == ["legacy"]


def _document(source_path, images):
    return SimpleNamespace(
        id="doc-hash",
        metadata={
            "source_path": source_path,
            "images": images,
        },
    )


def test_pipeline_image_replacement_removes_stale_images(tmp_path):
    store = ImageStorage(
        storage_root=str(tmp_path / "stored"),
        db_path=str(tmp_path / "index.db"),
    )
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._img_store = store
    pipeline._collection = "A"

    source = "/doc.pdf"
    p1 = tmp_path / "i1.png"
    p2 = tmp_path / "i2.png"
    p3 = tmp_path / "i3.png"
    p1.write_bytes(PNG_A)
    p2.write_bytes(PNG_A)
    p3.write_bytes(PNG_B)

    v1 = _document(source, [
        {"image_id": "i1", "path": str(p1), "page": 1},
        {"image_id": "i2", "path": str(p2), "page": 2},
    ])
    assert pipeline._replace_images(v1, source) == 0

    v2 = _document(source, [
        {"image_id": "i2", "path": str(p2), "page": 2},
        {"image_id": "i3", "path": str(p3), "page": 3},
    ])
    assert pipeline._replace_images(v2, source) == 1

    ids = {
        row["image_id"]
        for row in store.list_by_source(source, collection="A")
    }
    assert ids == {"i2", "i3"}
    assert store.get_path("i1", collection="A") is None


def test_pipeline_image_write_failure_keeps_old_images(tmp_path):
    store = ImageStorage(
        storage_root=str(tmp_path / "stored"),
        db_path=str(tmp_path / "index.db"),
    )
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._img_store = store
    pipeline._collection = "A"

    source = "/doc.pdf"
    old = tmp_path / "old.png"
    old.write_bytes(PNG_A)
    v1 = _document(source, [
        {"image_id": "old", "path": str(old), "page": 1},
    ])
    pipeline._replace_images(v1, source)

    broken = _document(source, [
        {
            "image_id": "new",
            "path": str(tmp_path / "missing.png"),
            "page": 1,
        },
    ])
    with pytest.raises(FileNotFoundError):
        pipeline._replace_images(broken, source)

    assert store.get_path("old", collection="A") is not None


def test_vector_upserter_persists_image_refs_as_json_string():
    class Store:
        def __init__(self):
            self.records = []

        def upsert(self, records, trace=None):
            self.records.extend(records)

    from src.core.types import Chunk

    store = Store()
    upserter = VectorUpserter.__new__(VectorUpserter)
    upserter._store = store
    chunk = Chunk(
        id="c1",
        doc_id="doc",
        text="with image",
        index=0,
        metadata={
            "source_path": "/doc.pdf",
            "dense_vector": [0.1],
            "image_refs": ["img1", "img2"],
        },
    )

    upserter.upsert([chunk], collection="A")
    assert store.records[0]["metadata"]["image_refs"] == '["img1", "img2"]'


def test_multimodal_assembler_uses_result_collection(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    path_a.write_bytes(PNG_A)
    path_b.write_bytes(PNG_B)

    class Storage:
        def __init__(self):
            self.calls = []

        def get_path(self, image_id, collection=None):
            self.calls.append((image_id, collection))
            return str(path_a if collection == "A" else path_b)

    storage = Storage()
    assembler = MultimodalAssembler(image_storage=storage)
    result = RetrievalResult(
        chunk_id="c1",
        score=1.0,
        text="x",
        metadata={
            "collection": "A",
            "image_refs": '["shared"]',
        },
    )

    output = assembler.assemble([result])
    assert len(output) == 1
    assert storage.calls == [("shared", "A")]


def test_multimodal_dedup_key_includes_collection(tmp_path):
    path = tmp_path / "img.png"
    path.write_bytes(PNG_A)

    class Storage:
        def get_path(self, image_id, collection=None):
            return str(path)

    assembler = MultimodalAssembler(image_storage=Storage())
    a = RetrievalResult(
        "a", 1.0, "a",
        {"collection": "A", "image_refs": '["shared"]'},
    )
    b = RetrievalResult(
        "b", 0.9, "b",
        {"collection": "B", "image_refs": '["shared"]'},
    )

    assert len(assembler.assemble([a, b])) == 2
