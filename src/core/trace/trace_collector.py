"""TraceCollector (src/core/trace/trace_collector.py)"""
from typing import Optional

from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger, write_trace

logger = get_logger(__name__)


class TraceCollector:
    """
    Trace 收集器：接收已完成的 TraceContext 并持久化。

    Usage：
        collector = TraceCollector()            # 写到默认 logs/traces.jsonl
        collector = TraceCollector("/tmp/t.jsonl")  # 测试时指定路径

        trace = TraceContext(trace_type="query")
        # ... 各阶段打点 ...
        trace.finish()
        collector.collect(trace)
    """

    def __init__(self, trace_file: Optional[str] = None) -> None:
        """
        Args:
            trace_file: 覆盖默认 traces.jsonl 路径（主要用于测试）。
        """
        self._trace_file = trace_file

    def collect(self, trace: TraceContext) -> None:
        """
        持久化一条已完成的 trace。

        若 trace 未调用 finish()，此处自动补调，确保总耗时被记录。

        Args:
            trace: 已完成（或未 finish）的 TraceContext 实例。
        """
        # 自动补 finish，确保 total_elapsed_ms 不为 None
        trace.finish()

        trace_dict = trace.to_dict()

        # 写入 JSON Lines 文件
        try:
            write_trace(trace_dict, trace_file=self._trace_file)
        except Exception as exc:
            logger.warning(f"Failed to persist trace {trace.trace_id}: {exc}")

        # stderr 人类可读摘要（方便实时调试）
        total_ms = trace_dict.get("total_elapsed_ms")
        stage_count = len(trace_dict.get("stages", []))
        logger.info(
            f"[trace] id={trace.trace_id[:8]}… "
            f"type={trace.trace_type} "
            f"stages={stage_count} "
            f"total={total_ms}ms"
        )


# 模块级全局 collector，供 Pipeline/HybridSearch 直接使用
global_collector = TraceCollector()
