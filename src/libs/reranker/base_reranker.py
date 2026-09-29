"""Reranker 抽象层 (src/libs/reranker/base_reranker.py)"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseReranker(ABC):
    """Reranker 抽象基类"""

    def __init__(self, model: str = None, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        trace=None,
    ) -> List[Dict[str, Any]]:
        """
        对候选 chunk 列表重新排序。

        Args:
            query:      用户原始查询文本。
            candidates: 待排序的候选列表，每条包含 chunk_id/text/score/metadata。
            top_k:      返回前 N 条，None 时返回全部。
            trace:      追踪上下文（可选）。
        Returns:
            重新排序后的候选列表，最相关的排在最前面。
        """
        pass


class NoneReranker(BaseReranker):
    """空 Reranker，不做任何重排，保持 RRF 融合后的原始顺序。"""

    def __init__(self):
        super().__init__(model=None)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        trace=None,
    ) -> List[Dict[str, Any]]:
        if top_k is not None:
            return candidates[:top_k]
        return candidates
