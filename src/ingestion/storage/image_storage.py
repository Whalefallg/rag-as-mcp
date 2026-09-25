"""
ImageStorage 实现 (src/ingestion/storage/image_storage.py)
===========================================================
为什么需要这个文件：
  多模态 RAG 的图片需要持久化（不能只在内存中），删除文档时要能级联清理。
  ImageStorage 用文件系统存图片字节，用 SQLite 存 image_id→path 的映射，
  这样 ImageCaptioner 按 image_id 找到图片路径，DocumentManager 删除文档时
  也能精确清理所有关联图片，不留孤儿文件占用磁盘。

本文件实现图片文件存储和索引映射管理。

类说明:
  - ImageStorage : 保存图片文件到 data/images/{collection}/ 目录，
                   并用 SQLite 维护 image_id → file_path 的持久化索引映射。
                   save()         : 写入图片文件（bytes），更新索引表。
                   get_path()     : 按 image_id 查询文件路径。
                   list_by_collection(): 按 collection 批量查询。
                   delete_by_doc() : 按 doc_hash 批量删除图片和索引记录。
                   数据库：data/db/image_index.db（WAL 模式，支持并发）。
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


class ImageStorage:
    """图片文件存储 + SQLite 索引管理"""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS image_index (
        image_id   TEXT PRIMARY KEY,
        file_path  TEXT NOT NULL,
        collection TEXT,
        doc_hash   TEXT,
        page_num   INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection);
    CREATE INDEX IF NOT EXISTS idx_doc_hash   ON image_index(doc_hash);
    """

    def __init__(
        self,
        storage_root: str = "data/images",
        db_path: str = "data/db/image_index.db",
    ):
        self._storage_root = Path(storage_root)
        self._db_path = db_path
        self._ensure_db()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(Path(self._db_path).parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def save(
        self,
        image_id: str,
        image_bytes: bytes,
        collection: str = "default",
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> str:
        """
        保存图片文件并写入索引，返回保存后的文件路径。

        文件路径约定：data/images/{collection}/{image_id}.png
        """
        img_dir = self._storage_root / collection
        img_dir.mkdir(parents=True, exist_ok=True)
        file_path = img_dir / f"{image_id}.png"
        file_path.write_bytes(image_bytes)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path  = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash   = excluded.doc_hash,
                    page_num   = excluded.page_num
                """,
                (image_id, str(file_path), collection, doc_hash, page_num, self._now()),
            )
        return str(file_path)

    def get_path(self, image_id: str) -> Optional[str]:
        """按 image_id 查询文件路径，不存在时返回 None"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        return row["file_path"] if row else None

    def list_by_collection(self, collection: str) -> List[dict]:
        """查询某 collection 下的所有图片记录"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_index WHERE collection = ?",
                (collection,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_by_doc(self, doc_hash: str) -> int:
        """删除某文档的所有图片（文件 + 索引），返回删除数量"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path FROM image_index WHERE doc_hash = ?",
                (doc_hash,),
            ).fetchall()
            count = len(rows)
            for row in rows:
                fp = Path(row["file_path"])
                if fp.exists():
                    fp.unlink()
            conn.execute(
                "DELETE FROM image_index WHERE doc_hash = ?",
                (doc_hash,),
            )
        return count

    def delete_by_source(self, source_path: str) -> int:
        """按 source_path 删除关联图片文件和索引记录，返回删除数量。"""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT image_id, file_path FROM image_index WHERE file_path LIKE ?',
                (f'%{source_path}%',),
            ).fetchall()
            if not rows:
                # fallback: 按 doc_hash 查找不到，尝试路径模糊匹配所有记录
                rows = conn.execute(
                    'SELECT image_id, file_path FROM image_index',
                ).fetchall()
                rows = [r for r in rows if source_path in (r["file_path"] or "")]
            count = len(rows)
            for row in rows:
                fp = Path(row["file_path"])
                if fp.exists():
                    fp.unlink(missing_ok=True)
                conn.execute(
                    'DELETE FROM image_index WHERE image_id = ?',
                    (row["image_id"],),
                )
        return count

    def list_images(self, collection: str = None) -> list:
        """列出图片记录，供 DataService 使用。"""
        with self._connect() as conn:
            if collection:
                rows = conn.execute(
                    'SELECT * FROM image_index WHERE collection = ?', (collection,)
                ).fetchall()
            else:
                rows = conn.execute('SELECT * FROM image_index').fetchall()
        return [dict(r) for r in rows]
