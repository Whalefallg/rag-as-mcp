"""
BM25Indexer roundtrip 测试 (tests/unit/test_bm25_indexer_roundtrip.py)
======================================================================
为什么需要这个文件：
  BM25 是 Sparse Retrieval 的核心，索引需要持久化到磁盘（进程重启后仍可查询）。
  roundtrip 测试验证 build->保存->加载->查询 全链路正确，
  IDF 测试确认稀有词的权重确实高于高频词（BM25 的核心语义），
  update 测试确认增量写入后新 chunk 可被检索到——这是摄取新文档时的关键行为。

验收标准（DEV_SPEC C11）：
  - build 后能 load 并对同一语料查询返回稳定 top ids
  - IDF 计算：仅在一篇文档中出现的词 IDF > 在所有文档中出现的词 IDF
  - 增量 update 后查询结果更新
  - 空语料查询返回空列表
"""
import pytest
import tempfile
import os
from src.core.types import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer


def _make_chunk(cid: str, text: str, idx: int = 0) -> Chunk:
    tokens = text.lower().split()
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    sparse = {t: c / total for t, c in freq.items()}
    return Chunk(
        id=cid, doc_id="doc1", text=text, index=idx,
        metadata={"sparse_vector": sparse, "source_path": "/test.pdf"},
    )


@pytest.fixture
def indexer(tmp_path):
    return BM25Indexer(index_dir=str(tmp_path / "bm25"))


def test_build_and_query_returns_results(indexer):
    chunks = [
        _make_chunk("c1", "machine learning model training", 0),
        _make_chunk("c2", "deep learning neural network", 1),
        _make_chunk("c3", "python programming language", 2),
    ]
    indexer.build(chunks)
    results = indexer.query({"machine": 0.5, "learning": 0.5}, top_k=3)
    assert len(results) > 0
    ids = [r["chunk_id"] for r in results]
    # machine learning 应该排在前面
    assert "c1" in ids


def test_query_returns_scores(indexer):
    chunks = [_make_chunk("c1", "hello world", 0)]
    indexer.build(chunks)
    results = indexer.query({"hello": 1.0})
    assert results[0]["score"] > 0


def test_build_persists_to_disk(tmp_path):
    idx1 = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    chunks = [_make_chunk("c1", "test persistence data", 0)]
    idx1.build(chunks)

    # 创建新实例，从磁盘加载
    idx2 = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    results = idx2.query({"test": 1.0})
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"


def test_idf_rare_term_higher_than_common(indexer):
    """只在1篇文档中出现的词 IDF 应大于在全部文档中出现的词"""
    chunks = [
        _make_chunk("c1", "common rare_term_xyz", 0),
        _make_chunk("c2", "common another_word", 1),
        _make_chunk("c3", "common yet_another", 2),
    ]
    indexer.build(chunks)
    # common 出现 3 次（df=3），rare_term_xyz 出现 1 次（df=1）
    idf_common = indexer._postings["common"]["idf"]
    idf_rare = indexer._postings["rare_term_xyz"]["idf"]
    assert idf_rare > idf_common


def test_query_empty_corpus_returns_empty(indexer):
    indexer.build([])
    results = indexer.query({"hello": 1.0})
    assert results == []


def test_query_unknown_term_returns_empty(indexer):
    chunks = [_make_chunk("c1", "hello world", 0)]
    indexer.build(chunks)
    results = indexer.query({"completely_unknown_term_zzzz": 1.0})
    assert results == []


def test_top_k_limits_results(indexer):
    chunks = [_make_chunk(f"c{i}", f"common word text {i}", i) for i in range(10)]
    indexer.build(chunks)
    results = indexer.query({"common": 1.0}, top_k=3)
    assert len(results) <= 3


def test_update_adds_new_chunk(indexer):
    chunks = [_make_chunk("c1", "hello world", 0)]
    indexer.build(chunks)

    new_chunk = _make_chunk("c2", "hello python programming", 1)
    indexer.update([new_chunk])

    results = indexer.query({"hello": 1.0}, top_k=5)
    ids = [r["chunk_id"] for r in results]
    assert "c1" in ids
    assert "c2" in ids
