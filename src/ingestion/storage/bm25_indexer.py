"""
BM25Indexer 实现 (src/ingestion/storage/bm25_indexer.py)
=========================================================
BM25 稀疏检索索引。索引持久化到本地 JSON，并支持增量更新。

关键语义：
  - Pipeline 摄取新文档时使用 update()，不能用 build() 覆盖已有文档。
  - collection 是检索隔离边界；同一个 chunk_id 可以同时存在于多个 collection。
  - BM25 对外返回业务 Chunk.id，与 VectorStore 的记录 ID 保持一致。
  - query() 的 N / avg_dl / df 均按 collection 计算，其他 collection 不影响排名。
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from src.core.types import Chunk


class BM25Indexer:
    """BM25 倒排索引构建、增量更新与持久化。"""

    K1 = 1.5
    B = 0.75
    _KEY_SEPARATOR = "\x1f"

    def __init__(self, index_dir: str = "data/db/bm25"):
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._postings_path = self._index_dir / "postings.json"
        self._meta_path = self._index_dir / "meta.json"

        self._postings: Dict[str, Any] = {}
        self._chunk_meta: Dict[str, Any] = {}
        self._loaded = False

    @classmethod
    def _index_key(cls, collection: str, chunk_id: str) -> str:
        return f"{collection}{cls._KEY_SEPARATOR}{chunk_id}"

    @staticmethod
    def _posting_key(posting: Dict[str, Any]) -> str:
        # 兼容旧索引：旧格式把业务 chunk_id 直接存在 posting["chunk_id"]。
        return posting.get("index_key") or posting.get("chunk_id", "")

    @staticmethod
    def _meta_collection(meta: Dict[str, Any]) -> str:
        # 旧索引没有 collection 字段，按 default 处理。
        return meta.get("collection", "default")

    def build(self, chunks: List[Chunk], collection: str = "default") -> None:
        """
        从头重建整个 BM25 索引（破坏性覆盖）。

        日常摄取应调用 update()；build() 仅用于显式全量重建。
        """
        self._postings = {}
        self._chunk_meta = {}
        self._loaded = True
        self._append_chunks(chunks, collection)
        self._recompute_idf()
        self._save()

    def update(self, chunks: List[Chunk], collection: str = "default") -> None:
        """
        增量更新指定 collection。

        已存在的同 collection + chunk_id 会先移除全部旧 term posting 再写入，
        因此即使新版本删除了某个 term，也不会留下旧 posting。
        """
        self._load_if_needed()
        if not chunks:
            return

        keys_to_replace = {
            self._index_key(collection, chunk.id)
            for chunk in chunks
        }
        # 兼容旧版 default 索引（旧 key 就是裸 chunk_id）。
        if collection == "default":
            keys_to_replace.update(chunk.id for chunk in chunks)

        for key in keys_to_replace:
            self._chunk_meta.pop(key, None)

        for term in list(self._postings.keys()):
            postings = [
                p for p in self._postings[term]["postings"]
                if self._posting_key(p) not in keys_to_replace
            ]
            if postings:
                self._postings[term]["postings"] = postings
            else:
                del self._postings[term]

        self._append_chunks(chunks, collection)
        self._recompute_idf()
        self._save()

    def _append_chunks(self, chunks: List[Chunk], collection: str) -> None:
        for chunk in chunks:
            sparse: Dict[str, float] = chunk.metadata.get("sparse_vector", {})
            doc_length = len(sparse)
            index_key = self._index_key(collection, chunk.id)
            self._chunk_meta[index_key] = {
                "chunk_id": chunk.id,
                "doc_length": doc_length,
                "source": chunk.metadata.get("source_path", ""),
                "collection": collection,
            }

            for term, tf in sparse.items():
                entry = self._postings.setdefault(
                    term, {"idf": 0.0, "postings": []}
                )
                entry["postings"].append({
                    "index_key": index_key,
                    "tf": tf,
                    "doc_length": doc_length,
                })

    def query(
        self,
        query_terms: Dict[str, float],
        top_k: int = 10,
        collection: str = "default",
    ) -> List[Dict[str, Any]]:
        """在指定 collection 内执行 BM25 检索。"""
        self._load_if_needed()

        eligible_meta = {
            key: meta
            for key, meta in self._chunk_meta.items()
            if self._meta_collection(meta) == collection
        }
        if not eligible_meta:
            return []

        avg_dl = (
            sum(meta["doc_length"] for meta in eligible_meta.values())
            / len(eligible_meta)
        )
        scores: Dict[str, float] = {}

        for term in query_terms:
            entry = self._postings.get(term)
            if not entry:
                continue

            eligible_postings = []
            for posting in entry["postings"]:
                key = self._posting_key(posting)
                if key in eligible_meta:
                    eligible_postings.append((key, posting))

            if not eligible_postings:
                continue

            n = len(eligible_meta)
            df = len({key for key, _ in eligible_postings})
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            for key, posting in eligible_postings:
                tf = posting["tf"]
                dl = posting["doc_length"]
                tf_score = (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / avg_dl)
                )
                scores[key] = scores.get(key, 0.0) + idf * tf_score

        ranked = sorted(scores.items(), key=lambda item: -item[1])
        results = []
        for index_key, score in ranked[:top_k]:
            meta = eligible_meta[index_key]
            results.append({
                "chunk_id": meta.get("chunk_id", index_key),
                "score": score,
            })
        return results

    def remove_document(
        self,
        source_path: str,
        collection: str = "default",
    ) -> int:
        """删除指定 collection 中属于 source_path 的所有 posting。"""
        self._load_if_needed()
        to_remove = {
            key
            for key, meta in self._chunk_meta.items()
            if (
                meta.get("source") == source_path
                and self._meta_collection(meta) == collection
            )
        }
        if not to_remove:
            return 0

        for key in to_remove:
            del self._chunk_meta[key]

        for term in list(self._postings.keys()):
            postings = [
                p for p in self._postings[term]["postings"]
                if self._posting_key(p) not in to_remove
            ]
            if postings:
                self._postings[term]["postings"] = postings
            else:
                del self._postings[term]

        self._recompute_idf()
        self._save()
        return len(to_remove)

    def _recompute_idf(self) -> None:
        """
        保留持久化 idf 字段用于兼容/调试。

        query() 会按 collection 重新计算 idf，避免其他 collection 改变排名。
        """
        n = len(self._chunk_meta)
        for entry in self._postings.values():
            df = len(entry["postings"])
            entry["idf"] = (
                math.log((n - df + 0.5) / (df + 0.5) + 1)
                if n and df
                else 0.0
            )

    def _save(self) -> None:
        self._postings_path.write_text(
            json.dumps(self._postings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._meta_path.write_text(
            json.dumps(self._chunk_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        if self._postings_path.exists():
            self._postings = json.loads(
                self._postings_path.read_text(encoding="utf-8")
            )
        if self._meta_path.exists():
            self._chunk_meta = json.loads(
                self._meta_path.read_text(encoding="utf-8")
            )
        self._loaded = True
