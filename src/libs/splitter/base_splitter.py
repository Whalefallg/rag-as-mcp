"""Splitter 抽象层 (src/libs/splitter/base_splitter.py)"""
from abc import ABC, abstractmethod
from typing import List


class BaseSplitter(ABC):
    """切分器抽象基类"""

    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs

    @abstractmethod
    def split_text(self, text: str, trace=None) -> List[str]:
        """
        将长文本切分为多个 chunk。

        Args:
            text: 待切分的原始文本（通常是 Markdown 格式）。
            trace: Optional trace shared with the ingestion pipeline.
        Returns:
            切分后的文本片段列表，每段长度不超过 chunk_size。
        """
        pass
