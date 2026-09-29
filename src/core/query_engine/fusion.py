"""Fusion / RRF (src/core/query_engine/fusion.py)"""
from typing import List, Dict

from src.core.types import RetrievalResult


class RRFusion:
    """Fuse independent ranked lists using reciprocal rank fusion."""

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF 平滑参数。值越大，排名靠前的优势越小；值越小，头部集中效应越强。
               60 是学术和工业界都验证过的经验最优值。
        """
        self._k = k

    def fuse(
        self,
        result_lists: List[List[RetrievalResult]],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        将多路召回结果用 RRF 算法融合，返回统一排序的 Top-K 列表。

        Args:
            result_lists: 多路召回结果，每路是一个 List[RetrievalResult]，
                          已按各自分数降序排列。
            top_k: 融合后保留的结果数量。
        Returns:
            按 RRF 分数降序排列的 RetrievalResult 列表。
            source 字段标记为 "fused"，score 字段替换为 RRF 融合分。
        """
        # chunk_id → 融合分数累计
        rrf_scores: Dict[str, float] = {}
        # chunk_id → 最后一次出现的 RetrievalResult（保留 text/metadata）
        id_to_result: Dict[str, RetrievalResult] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                cid = result.chunk_id
                # RRF 核心公式：每路贡献 1 / (k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._k + rank)
                id_to_result[cid] = result

        # 按融合分降序排列，截取 Top-K
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]

        return [
            RetrievalResult(
                chunk_id=cid,
                score=score,
                text=id_to_result[cid].text,
                metadata=id_to_result[cid].metadata,
                source="fused",
            )
            for cid, score in ranked
        ]

_FUSION_REGISTRY = {
    "rrf": RRFusion,
}


def create_fusion(algorithm: str):
    """按算法名创建 fusion 实现。"""
    if algorithm not in _FUSION_REGISTRY:
        supported = ", ".join(sorted(_FUSION_REGISTRY))
        raise ValueError(
            f"不支持的 fusion algorithm: '{algorithm}'。"
            f"已注册的算法: {supported}"
        )
    return _FUSION_REGISTRY[algorithm]()


def get_supported_algorithms() -> list:
    """返回当前真实注册的 fusion 算法。"""
    return list(_FUSION_REGISTRY.keys())
