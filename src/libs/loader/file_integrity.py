"""
文件完整性检查 (src/libs/loader/file_integrity.py)
===================================================
为什么需要这个文件：
  重复摄取是 RAG 系统最常见的运营问题——同一文件摄取两次会产生重复向量。
  SQLiteIntegrityChecker 用 SHA256 哈希 + SQLite 记录解决：
  成功摄取的文件下次会被跳过，失败的文件允许重试（幂等核心语义）。

本文件实现基于 SHA256 的文件去重与摄取历史管理。

类说明:
  - FileIntegrityChecker    : 抽象基类，定义"计算 hash / 是否跳过 / 标记结果"三个接口，
                              方便后续替换 Redis / PostgreSQL 等后端。

  - SQLiteIntegrityChecker  : 默认实现，使用本地 SQLite（WAL 模式）持久化摄取历史。
                              compute_sha256()：读文件按 64KB 分块流式计算，避免大文件 OOM。
                              should_skip()    ：hash 存在且状态为 success 时返回 True，
                                                 状态为 failed 时允许重试。
                              mark_success()   ：写入/更新记录，status=success。
                              mark_failed()    ：写入/更新记录，status=failed，记录错误原因。
"""
import hashlib
import sqlite3
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FileIntegrityChecker(ABC):
    """文件完整性检查抽象基类"""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """计算文件的 SHA256 哈希值"""
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """hash 已成功摄取时返回 True，表示可跳过"""
        pass

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, metadata: Optional[dict] = None) -> None:
        """将此 hash 标记为摄取成功"""
        pass

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """将此 hash 标记为摄取失败"""
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的文件完整性检查实现（WAL 模式，支持并发）"""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS ingestion_history (
        file_hash   TEXT PRIMARY KEY,
        file_path   TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'pending',
        error_msg   TEXT,
        ingested_at TEXT,
        updated_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status);
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        self._db_path = db_path
        self._ensure_db()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """创建连接，自动开启 WAL 模式"""
        os.makedirs(Path(self._db_path).parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """初始化数据库 schema"""
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def compute_sha256(self, path: str) -> str:
        """分块流式计算 SHA256，避免大文件 OOM（64KB/块）"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """hash 对应记录存在且 status=success 时返回 True"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        return row is not None and row["status"] == "success"

    def mark_success(self, file_hash: str, file_path: str, metadata: Optional[dict] = None) -> None:
        """写入/更新为 success 状态"""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, status, ingested_at, updated_at)
                VALUES (?, ?, 'success', ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path   = excluded.file_path,
                    status      = 'success',
                    error_msg   = NULL,
                    ingested_at = excluded.ingested_at,
                    updated_at  = excluded.updated_at
                """,
                (file_hash, file_path, now, now),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """写入/更新为 failed 状态，保留错误原因"""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, status, error_msg, updated_at)
                VALUES (?, '', 'failed', ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status     = 'failed',
                    error_msg  = excluded.error_msg,
                    updated_at = excluded.updated_at
                """,
                (file_hash, error_msg, now),
            )

    def get_record(self, file_hash: str) -> Optional[dict]:
        """调试用：返回完整记录"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        return dict(row) if row else None

    def remove_record(self, file_path: str) -> bool:
        """按 file_path 删除完整性记录，返回是否找到并删除。"""
        with self._connect() as conn:
            cur = conn.execute(
                'DELETE FROM ingestion_history WHERE file_path = ?',
                (file_path,),
            )
            return cur.rowcount > 0

    def list_processed(self, status: str = 'success') -> list:
        """返回指定状态的所有记录列表，供 DocumentManager 查询。"""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM ingestion_history WHERE status = ? ORDER BY ingested_at DESC',
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]
