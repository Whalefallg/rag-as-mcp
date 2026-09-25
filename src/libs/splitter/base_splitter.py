"""
Splitter 抽象层 (src/libs/splitter/base_splitter.py)
=====================================================
为什么需要这个文件：
  chunk_size 和 chunk_overlap 是影响检索质量最关键的两个超参数。
  BaseSplitter 定义统一接口，让 Pipeline 对切分策略无感知，
  只需在 settings.yaml 改 splitter.method 即可切换 Recursive/Semantic/Fixed 策略。

本文件定义文本切分的统一抽象接口。

类说明:
  - BaseSplitter : 切分器抽象基类。所有具体切分策略（Recursive/Semantic/Fixed）
                   都必须继承它并实现 split_text() 方法。
                   chunk_size 控制每个 chunk 的最大字符数，
                   chunk_overlap 控制相邻 chunk 之间的重叠字符数（防止语义在边界处断裂）。
                   在 Ingestion Pipeline 的 Chunking 阶段被 DocumentChunker 调用。
"""
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
            trace: 追踪上下文（可选，Phase F 阶段使用）。
        Returns:
            切分后的文本片段列表，每段长度不超过 chunk_size。
        """
        pass
