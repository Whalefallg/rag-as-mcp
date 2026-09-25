"""
Fusion / RRF (src/core/query_engine/fusion.py)
===============================================
为什么需要这个模块：
  Dense 和 Sparse 各自召回了 Top-20 个 chunk，但它们的分数不可比较——
  Dense 用余弦相似度（0~1 之间），Sparse 用 BM25 分数（理论上无上界）。
  直接加权求和会因为量纲不同而失真。

  RRF（Reciprocal Rank Fusion，倒数排名融合）是解决这个问题的经典方法：
    score_rrf(chunk) = Σ  1 / (k + rank_in_list_i)
  其中 rank 是该 chunk 在某路召回结果中的排名（从 1 开始），k 是平滑参数（默认 60）。
  只看排名、不看原始分数，天然规避了量纲问题。

  设计要点：
    - 为什么 RRF 比直接加权平均更鲁棒？→ 不受分数量纲影响，对 outlier 不敏感
    - k=60 怎么来的？→ 经验值，在多个信息检索 benchmark 上表现最优
    - RRF 有什么缺点？→ 丢失了原始分数信息，两路都排第 1 的 chunk 不比一路第 1 高多少

类说明:
  - RRFusion : 实现 RRF 算法，把多路 List[RetrievalResult] 融合成一个统一排序列表。
               fuse() 方法：遍历每路结果，按排名累加倒数分，最后按融合分降序输出。
               k 参数可在初始化时配置，让调用方通过 settings 控制行为。
"""
from typing import List, Dict

from src.core.types import RetrievalResult


class RRFusion:
    """
    Reciprocal Rank Fusion：把多路召回结果融合为统一排序。

    实现说明：
      RRF 最早由 Cormack et al. (2009) 提出，现已成为混合检索系统的标配融合策略，
      LangChain / LlamaIndex / Elasticsearch 都内置了这个算法。
    """

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
