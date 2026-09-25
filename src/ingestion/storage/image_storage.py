"""
ImageStorage 实现 (src/ingestion/storage/image_storage.py)
===========================================================
图片文件存储 + SQLite 索引。

关键语义：
  - image_id 只在 collection 内唯一；数据库主键是 (image_id, collection)。
  - 每条图片记录显式保存 source_path，支持按文档精确清理。
  - 旧数据库自动迁移；旧记录没有 source_path 时可通过 doc_hash 回退定位。
  - 文档更新采用“先写新图片，再删除 stale 图片”的可重试收敛语义。
"""
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


class ImageStorage:
    """图片文件存储 + SQLite 索引管理。"""

    _TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS image_index (
        image_id    TEXT NOT NULL,
        collection  TEXT NOT NULL DEFAULT 'default',
        file_path   TEXT NOT NULL,
        doc_hash    TEXT,
        source_path TEXT,
        page_num    INTEGER,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (image_id, collection)
    );
    """

    _INDEX_SCHEMA = """
    CREATE INDEX IF NOT EXISTS idx_image_collection
        ON image_index(collection);
    CREATE INDEX IF NOT EXISTS idx_image_doc_hash
        ON image_index(doc_hash);
    CREATE INDEX IF NOT EXISTS idx_image_source_collection
        ON image_index(source_path, collection);
    """

    def __init__(
        self,
        storage_root: str = "data/images",
        db_path: str = "data/db/image_index.db",
    ):
        self._storage_root = Path(storage_root)
        self._db_path = db_path
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(Path(self._db_path).parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """初始化 schema，并迁移旧版 image_id 全局主键。"""
        with self._connect() as conn:
            exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'image_index'
                """
            ).fetchone()

            if not exists:
                conn.execute(self._TABLE_SCHEMA)
                conn.executescript(self._INDEX_SCHEMA)
                return

            columns = conn.execute("PRAGMA table_info(image_index)").fetchall()
            names = {row["name"] for row in columns}
            pk_rows = [row for row in columns if row["pk"]]
            pk_cols = [
                row["name"]
                for row in sorted(pk_rows, key=lambda row: row["pk"])
            ]

            needs_migration = (
                "source_path" not in names
                or "collection" not in names
                or pk_cols != ["image_id", "collection"]
            )

            if needs_migration:
                conn.execute("ALTER TABLE image_index RENAME TO image_index_legacy")
                conn.execute(self._TABLE_SCHEMA)

                collection_expr = (
                    "COALESCE(collection, 'default')"
                    if "collection" in names
                    else "'default'"
                )
                source_expr = "source_path" if "source_path" in names else "NULL"
                doc_hash_expr = "doc_hash" if "doc_hash" in names else "NULL"
                page_expr = "page_num" if "page_num" in names else "NULL"
                created_expr = (
                    "created_at" if "created_at" in names else f"'{self._now()}'"
                )

                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO image_index (
                        image_id, collection, file_path, doc_hash,
                        source_path, page_num, created_at
                    )
                    SELECT
                        image_id, {collection_expr}, file_path, {doc_hash_expr},
                        {source_expr}, {page_expr}, {created_expr}
                    FROM image_index_legacy
                    """
                )
                conn.execute("DROP TABLE image_index_legacy")

            conn.executescript(self._INDEX_SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _doc_hash_for_source(source_path: str) -> str:
        return hashlib.sha256(source_path.encode()).hexdigest()[:16]

    def save(
        self,
        image_id: str,
        image_bytes: bytes,
        collection: str = "default",
        doc_hash: Optional[str] = None,
        source_path: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> str:
        """保存图片文件并写入 collection-scoped 索引。"""
        img_dir = self._storage_root / collection
        img_dir.mkdir(parents=True, exist_ok=True)
        file_path = img_dir / f"{image_id}.png"
        file_path.write_bytes(image_bytes)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_index (
                    image_id, collection, file_path, doc_hash,
                    source_path, page_num, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, collection) DO UPDATE SET
                    file_path   = excluded.file_path,
                    doc_hash    = excluded.doc_hash,
                    source_path = excluded.source_path,
                    page_num    = excluded.page_num
                """,
                (
                    image_id,
                    collection,
                    str(file_path),
                    doc_hash,
                    source_path,
                    page_num,
                    self._now(),
                ),
            )
        return str(file_path)

    def get_path(
        self,
        image_id: str,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """
        查询图片路径。

        指定 collection 时精确查询。未指定时仅在结果唯一时返回；
        若多 collection 存在同名 image_id，则优先 default，否则返回 None。
        """
        with self._connect() as conn:
            if collection is not None:
                row = conn.execute(
                    """
                    SELECT file_path
                    FROM image_index
                    WHERE image_id = ? AND collection = ?
                    """,
                    (image_id, collection),
                ).fetchone()
                return row["file_path"] if row else None

            rows = conn.execute(
                """
                SELECT file_path, collection
                FROM image_index
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchall()

        if len(rows) == 1:
            return rows[0]["file_path"]

        defaults = [row for row in rows if row["collection"] == "default"]
        if len(defaults) == 1:
            return defaults[0]["file_path"]
        return None

    def list_by_collection(self, collection: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_index WHERE collection = ?",
                (collection,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_by_source(
        self,
        source_path: str,
        collection: str = "default",
    ) -> List[dict]:
        """
        查询指定文档在 collection 内的图片。

        旧 schema 没有 source_path，因此 source_path 为 NULL 时用
        PdfLoader 的稳定 doc_hash 规则回退匹配。
        """
        doc_hash = self._doc_hash_for_source(source_path)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM image_index
                WHERE collection = ?
                  AND (
                    source_path = ?
                    OR (source_path IS NULL AND doc_hash = ?)
                  )
                ORDER BY image_id
                """,
                (collection, source_path, doc_hash),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_by_ids(
        self,
        image_ids: List[str],
        collection: str = "default",
    ) -> int:
        """精确删除指定 collection 中的图片 IDs。"""
        if not image_ids:
            return 0

        unique_ids = list(dict.fromkeys(image_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        params = [collection, *unique_ids]

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT image_id, file_path
                FROM image_index
                WHERE collection = ?
                  AND image_id IN ({placeholders})
                """,
                params,
            ).fetchall()

            for row in rows:
                Path(row["file_path"]).unlink(missing_ok=True)

            conn.execute(
                f"""
                DELETE FROM image_index
                WHERE collection = ?
                  AND image_id IN ({placeholders})
                """,
                params,
            )
        return len(rows)

    def delete_by_doc(
        self,
        doc_hash: str,
        collection: Optional[str] = None,
    ) -> int:
        """按 doc_hash 删除；collection=None 保留旧的跨集合行为。"""
        with self._connect() as conn:
            if collection is None:
                rows = conn.execute(
                    """
                    SELECT image_id, collection, file_path
                    FROM image_index
                    WHERE doc_hash = ?
                    """,
                    (doc_hash,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT image_id, collection, file_path
                    FROM image_index
                    WHERE doc_hash = ? AND collection = ?
                    """,
                    (doc_hash, collection),
                ).fetchall()

            for row in rows:
                Path(row["file_path"]).unlink(missing_ok=True)
                conn.execute(
                    """
                    DELETE FROM image_index
                    WHERE image_id = ? AND collection = ?
                    """,
                    (row["image_id"], row["collection"]),
                )
        return len(rows)

    def delete_by_source(
        self,
        source_path: str,
        collection: str = "default",
    ) -> int:
        """按 source_path 精确删除指定 collection 的图片。"""
        rows = self.list_by_source(source_path, collection=collection)
        return self.delete_by_ids(
            [row["image_id"] for row in rows],
            collection=collection,
        )

    def list_images(self, collection: Optional[str] = None) -> list:
        with self._connect() as conn:
            if collection:
                rows = conn.execute(
                    "SELECT * FROM image_index WHERE collection = ?",
                    (collection,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM image_index").fetchall()
        return [dict(row) for row in rows]
