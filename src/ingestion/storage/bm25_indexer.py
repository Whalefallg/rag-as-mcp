"""
BM25Indexer 实现 (src/ingestion/storage/bm25_indexer.py)
=========================================================
为什么需要这个文件：
  BM25 是 Sparse Retrieval 的核心算法——对精确关键词匹配（代码名、型号、专有名词）
  比 Dense Retrieval 更准。索引必须持久化到磁盘，重启后才能继续查询。
  BM25Indexer 接收 SparseEncoder 输出的词频，计算 IDF，构建倒排索引并序列化到文件。
  query() 实现完整的 BM25 打分，SparseRetriever 直接调用，无需重建索引。
  设计要点：BM25 公式中 k1 和 b 参数的含义？
    → k1 控制词频饱和速度（默认 1.5），b 控制文档长度归一化强度（默认 0.75）。

本文件实现 BM25 倒排索引的构建与持久化。

类说明:
  - BM25Indexer : 接收 SparseEncoder 输出的 term weights，计算 IDF，构建倒排索引，
                  并将索引序列化到 data/db/bm25/ 目录（JSON 格式，支持增量更新与重建）。

                  数据结构：
                    - postings: {term: {idf, postings: [{chunk_id, tf, doc_length}]}}
                    - chunk_meta: {chunk_id: {doc_length, source}}

                  IDF 公式：log((N - df + 0.5) / (df + 0.5) + 1)
                            （+1 防止 IDF 为负，Robertson-Walker 变种）

                  query() 方法实现 BM25 检索（k1=1.5, b=0.75），
                  供 D 阶段的 SparseRetriever 直接调用。
"""
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.types import Chunk


class BM25Indexer:
    """BM25 倒排索引构建与持久化"""

    # BM25 超参数
    K1 = 1.5
    B = 0.75

    def __init__(self, index_dir: str = "data/db/bm25"):
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._postings_path = self._index_dir / "postings.json"
        self._meta_path = self._index_dir / "meta.json"

        # 内存索引（懒加载）
        self._postings: Dict[str, Any] = {}      # term → {idf, postings: [...]}
        self._chunk_meta: Dict[str, Any] = {}    # chunk_id → {doc_length, source}
        self._loaded = False

    # ── 构建接口 ──────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """
        从头构建完整索引（覆盖写入）。

        Args:
            chunks: 已完成 SparseEncoder 的 Chunk 列表（metadata 含 sparse_vector）。
        """
        raw_postings: Dict[str, List[dict]] = {}  # term → [{chunk_id, tf, doc_length}]
        chunk_meta: Dict[str, Any] = {}

        for chunk in chunks:
            sparse: Dict[str, float] = chunk.metadata.get("sparse_vector", {})
            doc_length = len(sparse)
            chunk_meta[chunk.id] = {
                "doc_length": doc_length,
                "source": chunk.metadata.get("source_path", ""),
            }
            for term, tf in sparse.items():
                if term not in raw_postings:
                    raw_postings[term] = []
                raw_postings[term].append({
                    "chunk_id": chunk.id,
                    "tf": tf,
                    "doc_length": doc_length,
                })

        n = len(chunks)
        postings: Dict[str, Any] = {}
        for term, term_postings in raw_postings.items():
            df = len(term_postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            postings[term] = {"idf": idf, "postings": term_postings}

        self._postings = postings
        self._chunk_meta = chunk_meta
        self._loaded = True
        self._save()

    def update(self, chunks: List[Chunk]) -> None:
        """增量更新：向现有索引追加新 chunk（重复 chunk_id 会覆盖）"""
        self._load_if_needed()
        n_existing = len(self._chunk_meta)

        for chunk in chunks:
            sparse: Dict[str, float] = chunk.metadata.get("sparse_vector", {})
            doc_length = len(sparse)
            self._chunk_meta[chunk.id] = {
                "doc_length": doc_length,
                "source": chunk.metadata.get("source_path", ""),
            }
            for term, tf in sparse.items():
                if term not in self._postings:
                    self._postings[term] = {"idf": 0.0, "postings": []}
                # 移除旧记录（若存在），再追加新记录
                self._postings[term]["postings"] = [
                    p for p in self._postings[term]["postings"]
                    if p["chunk_id"] != chunk.id
                ]
                self._postings[term]["postings"].append({
                    "chunk_id": chunk.id,
                    "tf": tf,
                    "doc_length": doc_length,
                })

        # 重算 IDF
        n = len(self._chunk_meta)
        for term, entry in self._postings.items():
            df = len(entry["postings"])
            entry["idf"] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        self._save()

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def query(self, query_terms: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25 检索，返回 top_k 个候选 chunk。

        Args:
            query_terms: 查询词的 tf 字典（由 SparseEncoder 计算）。
            top_k: 返回结果数量。
        Returns:
            [{chunk_id, score}] 按 BM25 分数降序排列。
        """
        self._load_if_needed()
        if not self._chunk_meta:
            return []

        avg_dl = sum(v["doc_length"] for v in self._chunk_meta.values()) / len(self._chunk_meta)
        scores: Dict[str, float] = {}

        for term in query_terms:
            if term not in self._postings:
                continue
            idf = self._postings[term]["idf"]
            for posting in self._postings[term]["postings"]:
                cid = posting["chunk_id"]
                tf = posting["tf"]
                dl = posting["doc_length"]
                # BM25 公式
                tf_score = (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / avg_dl)
                )
                scores[cid] = scores.get(cid, 0.0) + idf * tf_score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"chunk_id": cid, "score": score} for cid, score in ranked[:top_k]]

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._postings_path.write_text(
            json.dumps(self._postings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._meta_path.write_text(
            json.dumps(self._chunk_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        if self._postings_path.exists():
            self._postings = json.loads(self._postings_path.read_text(encoding="utf-8"))
        if self._meta_path.exists():
            self._chunk_meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._loaded = True

    def remove_document(self, source_path: str) -> int:
        """删除属于指定文档（source_path）的所有 posting 记录，返回删除的 chunk 数。"""
        self._load_if_needed()
        to_remove = {
            cid for cid, meta in self._chunk_meta.items()
            if meta.get('source') == source_path
        }
        if not to_remove:
            return 0
        for cid in to_remove:
            del self._chunk_meta[cid]
        for term in list(self._postings.keys()):
            self._postings[term]['postings'] = [
                p for p in self._postings[term]['postings']
                if p['chunk_id'] not in to_remove
            ]
            if not self._postings[term]['postings']:
                del self._postings[term]
        # 重算 IDF
        n = len(self._chunk_meta)
        if n > 0:
            import math
            for entry in self._postings.values():
                df = len(entry['postings'])
                entry['idf'] = math.log((n - df + 0.5) / (df + 0.5) + 1)
        self._save()
        return len(to_remove)
