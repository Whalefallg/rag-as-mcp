"""
BatchProcessor 实现 (src/ingestion/embedding/batch_processor.py)
================================================================
为什么需要这个文件：
  一篇文档可能切成几百个 chunk，一次性全发给 Embedding API 容易触发限流或 OOM。
  BatchProcessor 把 chunk 列表按 batch_size 切片，逐批调用 DenseEncoder 和 SparseEncoder，
  避免单次请求过载。顺序稳定（按切片顺序处理）确保输出向量与输入 chunk 一一对应，
  上层按下标合并即可，不需要做 ID 匹配。

本文件实现批处理编排器，将 Chunk 列表分批驱动 Dense/Sparse 编码。

类说明:
  - BatchProcessor : 将 chunks 按 batch_size 分批，依次调用 DenseEncoder 和 SparseEncoder，
                     合并结果后返回完整编码列表。
                     批次顺序稳定（slice 按序切割），不做并发，保证输出顺序与输入一致。
                     每批耗时通过 TraceContext.record_stage 记录，为 Phase F trace 预留钩子。
"""
import time
from typing import List, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """批处理编排器：分批驱动 Dense + Sparse 编码"""

    def __init__(self, settings: Settings, batch_size: int = 32):
        self._dense = DenseEncoder(settings)
        self._sparse = SparseEncoder(settings)
        self._batch_size = batch_size

    def encode_all(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """
        对所有 Chunk 分批执行 Dense + Sparse 编码。

        Args:
            chunks: 待编码的完整 Chunk 列表。
            trace: 追踪上下文（可选）。
        Returns:
            完整编码后的 Chunk 列表，顺序与输入一致。
        """
        if not chunks:
            return []

        results: List[Chunk] = []
        total_batches = (len(chunks) + self._batch_size - 1) // self._batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, len(chunks))
            batch = chunks[start:end]

            t0 = time.monotonic()

            # Dense 编码
            batch = self._dense.encode(batch, trace=trace)
            # Sparse 编码
            batch = self._sparse.encode(batch, trace=trace)

            elapsed_ms = (time.monotonic() - t0) * 1000

            if trace:
                trace.record_stage(
                    "batch_encode",
                    batch_idx=batch_idx,
                    batch_size=len(batch),
                    elapsed_ms=elapsed_ms,
                )

            results.extend(batch)

        return results
