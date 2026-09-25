"""
RRFusion 单元测试 (tests/unit/test_fusion_rrf.py)
=================================================
验收标准（DEV_SPEC D4）：
  - 对构造的排名输入，输出 deterministic（相同输入相同输出）
  - k 参数可配置，影响分数计算
  - 同时出现在多路结果中的 chunk 得分最高
  - 只在一路中出现的 chunk 也被保留
  - top_k 截断正确
  - source 字段标记为 "fused"
  - 空输入列表返回空结果
"""
import pytest
from src.core.types import RetrievalResult
from src.core.query_engine.fusion import RRFusion


def _make_result(cid: str, score: float = 1.0, source: str = "dense") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid, score=score, text=f"text of {cid}",
        metadata={"source_path": "/test.pdf"}, source=source,
    )


# ── 基本功能 ──────────────────────────────────────────────────────────────────

def test_fuse_returns_list():
    fusion = RRFusion()
    dense = [_make_result("c1"), _make_result("c2")]
    sparse = [_make_result("c1"), _make_result("c3")]
    result = fusion.fuse([dense, sparse])
    assert isinstance(result, list)


def test_fuse_chunk_in_both_lists_gets_highest_score():
    fusion = RRFusion(k=60)
    # c1 在 dense 排第 1，sparse 排第 1 → 得分最高
    dense = [_make_result("c1"), _make_result("c2"), _make_result("c3")]
    sparse = [_make_result("c1"), _make_result("c4"), _make_result("c5")]
    result = fusion.fuse([dense, sparse], top_k=5)
    assert result[0].chunk_id == "c1"


def test_fuse_only_in_one_list_still_included():
    fusion = RRFusion()
    dense = [_make_result("c1"), _make_result("c2")]
    sparse = [_make_result("c3")]  # c3 只在 sparse 中
    result = fusion.fuse([dense, sparse], top_k=5)
    ids = [r.chunk_id for r in result]
    assert "c3" in ids


def test_fuse_top_k_truncation():
    fusion = RRFusion()
    dense = [_make_result(f"c{i}") for i in range(10)]
    sparse = [_make_result(f"c{i}") for i in range(10)]
    result = fusion.fuse([dense, sparse], top_k=3)
    assert len(result) == 3


def test_fuse_source_marked_fused():
    fusion = RRFusion()
    result = fusion.fuse([[_make_result("c1")]], top_k=5)
    assert all(r.source == "fused" for r in result)


def test_fuse_score_is_float():
    fusion = RRFusion()
    result = fusion.fuse([[_make_result("c1")]])
    assert isinstance(result[0].score, float)
    assert result[0].score > 0


def test_fuse_deterministic():
    """相同输入，多次调用输出相同"""
    fusion = RRFusion(k=60)
    dense = [_make_result(f"c{i}") for i in range(5)]
    sparse = [_make_result(f"c{4 - i}") for i in range(5)]
    r1 = fusion.fuse([dense, sparse], top_k=5)
    r2 = fusion.fuse([dense, sparse], top_k=5)
    assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


def test_fuse_empty_input_returns_empty():
    fusion = RRFusion()
    assert fusion.fuse([]) == []


def test_fuse_empty_lists_returns_empty():
    fusion = RRFusion()
    assert fusion.fuse([[], []]) == []


def test_fuse_single_list():
    fusion = RRFusion()
    dense = [_make_result("c1"), _make_result("c2")]
    result = fusion.fuse([dense], top_k=5)
    assert len(result) == 2
    assert result[0].chunk_id == "c1"  # rank=1 得分更高


# ── k 参数验证 ────────────────────────────────────────────────────────────────

def test_fuse_rrf_formula_correctness():
    """
    验证 RRF 公式：score = 1 / (k + rank)
    k=0 时，rank=1 → score=1.0，rank=2 → score=0.5
    """
    fusion = RRFusion(k=0)
    dense = [_make_result("c1"), _make_result("c2")]
    result = fusion.fuse([dense], top_k=2)
    assert result[0].chunk_id == "c1"
    assert abs(result[0].score - 1.0) < 1e-9
    assert abs(result[1].score - 0.5) < 1e-9


def test_fuse_larger_k_smoother_scores():
    """k 越大，排名 1 和排名 2 之间的分数差越小"""
    fusion_small_k = RRFusion(k=1)
    fusion_large_k = RRFusion(k=100)
    lst = [_make_result("c1"), _make_result("c2")]
    r_small = fusion_small_k.fuse([lst])
    r_large = fusion_large_k.fuse([lst])
    diff_small = r_small[0].score - r_small[1].score
    diff_large = r_large[0].score - r_large[1].score
    assert diff_small > diff_large


# ── text / metadata 保留 ──────────────────────────────────────────────────────

def test_fuse_preserves_text():
    fusion = RRFusion()
    result = fusion.fuse([[_make_result("c1")]])
    assert result[0].text == "text of c1"


def test_fuse_preserves_metadata():
    fusion = RRFusion()
    result = fusion.fuse([[_make_result("c1")]])
    assert result[0].metadata["source_path"] == "/test.pdf"
