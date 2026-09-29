"""Transform 抽象层 + ChunkRefiner (src/ingestion/transform/base_transform.py)"""
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Transform 抽象基类"""

    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        对 Chunk 列表执行变换。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。
        Returns:
            处理后的 Chunk 列表，顺序与输入一致。
        """
        pass
