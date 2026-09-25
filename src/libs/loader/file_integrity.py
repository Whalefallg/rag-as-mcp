"""
文件完整性检查 (src/libs/loader/file_integrity.py)
===================================================
基于 SHA256 + SQLite 的摄取幂等记录。

同一个文件允许进入多个 collection，因此幂等键必须是
(file_hash, collection)，而不是全局 file_hash。
"""
import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FileIntegrityChecker(ABC):
    """文件完整性检查抽象基类。"""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        pass

    @abstractmethod
    def should_skip(
        self,
        file_hash: str,
        collection: str = "default",
    ) -> bool:
        pass

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        metadata: Optional[dict] = None,
        collection: str = "default",
    ) -> None:
        pass

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        collection: str = "default",
    ) -> None:
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的文件完整性检查实现（WAL 模式）。"""

    _TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS ingestion_history (
        file_hash   TEXT NOT NULL,
        collection  TEXT NOT NULL DEFAULT 'default',
        file_path   TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'pending',
        error_msg   TEXT,
        ingested_at TEXT,
        updated_at  TEXT NOT NULL,
        PRIMARY KEY (file_hash, collection)
    );
    """

    _INDEX_SCHEMA = """
    CREATE INDEX IF NOT EXISTS idx_status
        ON ingestion_history(status);
    CREATE INDEX IF NOT EXISTS idx_collection
        ON ingestion_history(collection);
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        self._db_path = db_path
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(Path(self._db_path).parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """
        初始化 schema，并把旧版 file_hash PRIMARY KEY 表迁移到复合主键。

        旧记录归入 default collection，保证升级后原有默认知识库仍可识别。
        """
        with self._connect() as conn:
            exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'ingestion_history'
                """
            ).fetchone()

            if not exists:
                conn.execute(self._TABLE_SCHEMA)
                conn.executescript(self._INDEX_SCHEMA)
                return

            columns = conn.execute(
                "PRAGMA table_info(ingestion_history)"
            ).fetchall()
            names = {row["name"] for row in columns}
            pk_rows = [row for row in columns if row["pk"]]
            pk_cols = [
                row["name"]
                for row in sorted(pk_rows, key=lambda row: row["pk"])
            ]

            if (
                "collection" not in names
                or pk_cols != ["file_hash", "collection"]
            ):
                conn.execute(
                    "ALTER TABLE ingestion_history "
                    "RENAME TO ingestion_history_legacy"
                )
                conn.execute(self._TABLE_SCHEMA)

                if "collection" in names:
                    collection_expr = (
                        "COALESCE(collection, 'default')"
                    )
                else:
                    collection_expr = "'default'"

                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO ingestion_history (
                        file_hash, collection, file_path, status,
                        error_msg, ingested_at, updated_at
                    )
                    SELECT
                        file_hash, {collection_expr}, file_path, status,
                        error_msg, ingested_at, updated_at
                    FROM ingestion_history_legacy
                    """
                )
                conn.execute("DROP TABLE ingestion_history_legacy")

            conn.executescript(self._INDEX_SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def compute_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def should_skip(
        self,
        file_hash: str,
        collection: str = "default",
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status
                FROM ingestion_history
                WHERE file_hash = ? AND collection = ?
                """,
                (file_hash, collection),
            ).fetchone()
        return row is not None and row["status"] == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        metadata: Optional[dict] = None,
        collection: str = "default",
    ) -> None:
        now = self._now()
        with self._connect() as conn:
            # 同一路径在同一 collection 中只保留当前成功版本。
            # 这样 v1 -> v2 成功后，如果文件又回退到 v1，不会被旧 hash 错误跳过。
            conn.execute(
                """
                DELETE FROM ingestion_history
                WHERE file_path = ?
                  AND collection = ?
                  AND file_hash <> ?
                """,
                (file_path, collection, file_hash),
            )
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, collection, file_path,
                    status, ingested_at, updated_at
                )
                VALUES (?, ?, ?, 'success', ?, ?)
                ON CONFLICT(file_hash, collection) DO UPDATE SET
                    file_path   = excluded.file_path,
                    status      = 'success',
                    error_msg   = NULL,
                    ingested_at = excluded.ingested_at,
                    updated_at  = excluded.updated_at
                """,
                (file_hash, collection, file_path, now, now),
            )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        collection: str = "default",
    ) -> None:
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, collection, file_path,
                    status, error_msg, updated_at
                )
                VALUES (?, ?, '', 'failed', ?, ?)
                ON CONFLICT(file_hash, collection) DO UPDATE SET
                    status     = 'failed',
                    error_msg  = excluded.error_msg,
                    updated_at = excluded.updated_at
                """,
                (file_hash, collection, error_msg, now),
            )

    def get_record(
        self,
        file_hash: str,
        collection: str = "default",
    ) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM ingestion_history
                WHERE file_hash = ? AND collection = ?
                """,
                (file_hash, collection),
            ).fetchone()
        return dict(row) if row else None

    def remove_record(
        self,
        file_path: str,
        collection: Optional[str] = None,
    ) -> bool:
        """
        删除摄取历史。

        collection=None 保留旧行为（删除该路径所有 collection 的记录）；
        指定 collection 时只删除该知识库的记录。
        """
        with self._connect() as conn:
            if collection is None:
                cur = conn.execute(
                    "DELETE FROM ingestion_history WHERE file_path = ?",
                    (file_path,),
                )
            else:
                cur = conn.execute(
                    """
                    DELETE FROM ingestion_history
                    WHERE file_path = ? AND collection = ?
                    """,
                    (file_path, collection),
                )
            return cur.rowcount > 0

    def list_processed(
        self,
        status: str = "success",
        collection: Optional[str] = None,
    ) -> list:
        with self._connect() as conn:
            if collection is None:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM ingestion_history
                    WHERE status = ?
                    ORDER BY ingested_at DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM ingestion_history
                    WHERE status = ? AND collection = ?
                    ORDER BY ingested_at DESC
                    """,
                    (status, collection),
                ).fetchall()
        return [dict(row) for row in rows]
