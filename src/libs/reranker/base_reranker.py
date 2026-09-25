"""
Reranker 抽象层 (src/libs/reranker/base_reranker.py)
=====================================================
为什么需要这个文件：
  精排是 RAG 系统中成本最高的步骤（Cross-Encoder 每次都要过模型）。
  BaseReranker + NoneReranker（空对象模式）让系统可以无缝关闭精排：
  backend=none 时返回 NoneReranker，调用方代码路径完全一致，不需要 if 判断。

本文件定义精排重排的统一抽象接口，并提供一个开箱即用的空实现。

类说明:
  - BaseReranker  : Reranker 抽象基类。所有具体后端（CrossEncoder/LLM）
                    都必须继承它并实现 rerank() 方法。
                    rerank() 接收 query + 候选 chunk 列表，返回重新排序后的列表。
                    在 Retrieval Pipeline 的精排阶段被 Core 层的 Reranker 编排器调用。

  - NoneReranker  : 空对象实现（Null Object Pattern）。
                    不做任何重排，直接原样返回候选列表。
                    当 settings.rerank.backend = "none" 时工厂返回此实例，
                    使调用方无需判断"是否开启精排"，代码路径始终一致。
"""
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
