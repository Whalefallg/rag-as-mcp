"""
TraceContext 增强实现 (src/core/trace/trace_context.py)
========================================================
为什么需要这个文件：
  每次查询或摄取调用都是一条链路，经过多个组件。
  TraceContext 在链路启动时生成唯一 trace_id，各组件把耗时/结果写入同一个 context，
  调试时能把分散的日志关联在一起。Phase F 完善了持久化和序列化能力。

  Phase F 新增内容：
    - trace_type 字段区分 "query" / "ingestion" 两类链路
    - finish() 方法标记链路结束，计算总耗时
    - to_dict() 输出完整的可 JSON 序列化字典（含 ISO 时间戳）
    - elapsed_ms() 查询指定阶段或总耗时
    - record_stage() 扩展为支持直接传入耗时（不强制用 span() 上下文管理器）

类说明:
  - SpanRecord   : 单个执行阶段的记录，含 stage/data/duration_ms/error。
  - TraceContext : 追踪上下文，贯穿单次 Pipeline 全链路。
                   trace_type="query"     → Query 链路（HybridSearch/Reranker）
                   trace_type="ingestion" → Ingestion 链路（Pipeline 七步）
"""
import uuid
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SpanRecord:
    """单个执行阶段的记录"""
    stage: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    started_at: Optional[float] = None   # monotonic 时间，仅内部使用


class TraceContext:
    """
    Pipeline 追踪上下文。

    Usage（Query 链路）：
        trace = TraceContext(trace_type="query")
        with trace.span("dense_retrieval") as span:
            span.data["top_k"] = 20
            results = dense_retriever.retrieve(...)
        trace.finish()
        collector.collect(trace)   # 持久化到 traces.jsonl

    Usage（显式记录，不用 span）：
        trace = TraceContext(trace_type="ingestion")
        t0 = time.monotonic()
        chunks = chunker.split(doc)
        trace.record_stage("split", duration_ms=(time.monotonic()-t0)*1000,
                           chunk_count=len(chunks))
        trace.finish()
    """

    def __init__(
        self,
        trace_type: str = "query",
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            trace_type: "query" 或 "ingestion"，用于 Dashboard 分类展示。
            trace_id:   外部传入时复用（测试/重试场景），否则自动生成 UUID4。
        """
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.trace_type: str = trace_type
        self._started_at_wall: datetime = datetime.now(timezone.utc)
        self._started_at_mono: float = time.monotonic()
        self._finished_at_wall: Optional[datetime] = None
        self._finished_at_mono: Optional[float] = None
        self._spans: List[SpanRecord] = []
        self._metadata: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    # 阶段记录
    # ──────────────────────────────────────────────────────────────────────

    def record_stage(
        self,
        stage: str,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        显式记录一个阶段（不自动计时）。

        Args:
            stage:       阶段名称，如 "dense_retrieval" / "split" / "upsert"。
            duration_ms: 调用方自行计时后传入（可选）。
            error:       若阶段失败，记录错误描述（可选）。
            **kwargs:    任意键值对，写入 SpanRecord.data。
        """
        self._spans.append(SpanRecord(
            stage=stage,
            data=dict(kwargs),
            duration_ms=duration_ms,
            error=error,
        ))

    @contextmanager
    def span(self, stage: str):
        """
        上下文管理器：自动计时，异常时记录 error 并重新抛出。

        Usage:
            with trace.span("rerank") as span:
                span.data["backend"] = "cross_encoder"
                results = reranker.rerank(...)
                span.data["result_count"] = len(results)
        """
        record = SpanRecord(stage=stage, started_at=time.monotonic())
        try:
            yield record
        except Exception as exc:
            record.error = str(exc)
            raise
        finally:
            record.duration_ms = (time.monotonic() - record.started_at) * 1000
            self._spans.append(record)

    # ──────────────────────────────────────────────────────────────────────
    # 元数据
    # ──────────────────────────────────────────────────────────────────────

    def set_metadata(self, key: str, value: Any) -> None:
        """设置链路级元数据（如 user_query、source_path、collection）"""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    # ──────────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────────

    def finish(self) -> None:
        """
        标记链路结束，记录结束时间。
        finish() 之后调用 to_dict() 才包含 total_elapsed_ms。
        可安全重复调用（幂等）。
        """
        if self._finished_at_mono is None:
            self._finished_at_mono = time.monotonic()
            self._finished_at_wall = datetime.now(timezone.utc)

    def elapsed_ms(self, stage: Optional[str] = None) -> Optional[float]:
        """
        查询耗时（毫秒）。

        Args:
            stage: 指定阶段名称时返回该阶段耗时；None 时返回链路总耗时。
        Returns:
            毫秒数，若尚未 finish() 且 stage=None 则返回 None。
        """
        if stage is not None:
            for s in reversed(self._spans):
                if s.stage == stage:
                    return s.duration_ms
            return None
        if self._finished_at_mono is None:
            return None
        return (self._finished_at_mono - self._started_at_mono) * 1000

    # ──────────────────────────────────────────────────────────────────────
    # 序列化
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为可直接 json.dumps() 的字典。

        输出结构：
          trace_id, trace_type, started_at (ISO), finished_at (ISO),
          total_elapsed_ms, metadata, stages[]
        """
        total_ms: Optional[float] = None
        finished_str: Optional[str] = None
        if self._finished_at_mono is not None:
            total_ms = round((self._finished_at_mono - self._started_at_mono) * 1000, 2)
            finished_str = self._finished_at_wall.isoformat()  # type: ignore[union-attr]

        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self._started_at_wall.isoformat(),
            "finished_at": finished_str,
            "total_elapsed_ms": total_ms,
            "metadata": self._metadata,
            "stages": [
                {
                    "stage": s.stage,
                    "duration_ms": round(s.duration_ms, 2) if s.duration_ms is not None else None,
                    "error": s.error,
                    **s.data,
                }
                for s in self._spans
            ],
        }
