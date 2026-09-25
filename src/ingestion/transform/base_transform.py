"""
Transform 抽象层 + ChunkRefiner (src/ingestion/transform/base_transform.py)
===========================================================================
为什么需要这个文件：
  Ingestion Pipeline 的 transform 阶段有三个步骤：ChunkRefiner → MetadataEnricher → ImageCaptioner。
  它们都对 List[Chunk] 做「输入 chunk 列表 → 输出处理后的 chunk 列表」的变换。
  BaseTransform 把这个共同接口抽象出来，让 Pipeline 可以用统一的方式编排所有变换器，
  将来新增一个变换器只需继承 BaseTransform，不需要修改 Pipeline 代码。

本文件定义 Transform 的抽象接口和 ChunkRefiner 具体实现。

类说明:
  - BaseTransform  : Transform 抽象基类。所有 Chunk 变换器（ChunkRefiner/MetadataEnricher/
                     ImageCaptioner）都继承它并实现 transform() 方法。
                     transform() 接收 Chunk 列表，返回处理后的 Chunk 列表。
                     设计为可链式组合：pipeline 中按顺序调用多个 transform。

  - ChunkRefiner   : Chunk 文本净化器，分两个模式运行：
                     1. 规则模式（始终执行）：正则去除页眉/页脚/多余空白/分隔线/HTML注释。
                     2. LLM 模式（可选，settings.ingestion.chunk_refiner.use_llm=true）：
                        调用 LLM 对文本进行语义重写，去除噪声同时保留核心内容。
                     降级机制：LLM 失败时回退到规则结果，在 metadata 标记降级原因，
                     不抛出异常，不阻塞 Ingestion Pipeline。
"""
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
