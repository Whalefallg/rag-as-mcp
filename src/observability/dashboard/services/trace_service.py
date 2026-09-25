"""
TraceService (src/observability/dashboard/services/trace_service.py)
=====================================================================
为什么需要这个文件：
  traces.jsonl 是 Pipeline/Query 链路的持久化记录。
  Dashboard 的追踪页面需要：
    1. 按 trace_type 筛选（"ingestion" 或 "query"）
    2. 按时间倒序排列
    3. 解析阶段数据，计算耗时分布
  TraceService 封装所有读取逻辑，Dashboard 页面只调用高层接口，
  不直接操作文件或解析 JSON。

  性能考量：
    traces.jsonl 可能有上千行。TraceService 在读取时只解析必要字段，
    需要完整 stages 时再按需加载（lazy loading 思路）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_TRACE_FILE = os.environ.get("MCP_TRACE_FILE", "logs/traces.jsonl")


@dataclass
class TraceRecord:
    """单条 trace 的结构化表示"""
    trace_id: str
    trace_type: str                      # "query" 或 "ingestion"
    started_at: str                      # ISO 时间字符串
    finished_at: Optional[str]
    total_elapsed_ms: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def user_query(self) -> str:
        return self.metadata.get("user_query", "—")

    @property
    def source_path(self) -> str:
        return self.metadata.get("source_path", "—")

    @property
    def collection(self) -> str:
        return self.metadata.get("collection", "default")

    def stage_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for s in self.stages:
            if s.get("stage") == name:
                return s
        return None


class TraceService:
    """
    读取 logs/traces.jsonl，解析并过滤 TraceRecord 列表。

    Usage:
        svc = TraceService()
        ingestion_traces = svc.list(trace_type="ingestion")
        query_traces = svc.list(trace_type="query", limit=50)
    """

    def __init__(self, trace_file: Optional[str] = None) -> None:
        self._file = trace_file or _DEFAULT_TRACE_FILE

    def list(
        self,
        trace_type: Optional[str] = None,
        limit: int = 200,
        keyword: Optional[str] = None,
    ) -> List[TraceRecord]:
        """
        读取 traces.jsonl，返回按时间倒序的 TraceRecord 列表。

        Args:
            trace_type: 过滤 "query" 或 "ingestion"，None 时返回全部。
            limit:      最多返回的条数（避免大文件过慢）。
            keyword:    按 user_query 或 source_path 关键词过滤（大小写不敏感）。
        Returns:
            时间倒序的 TraceRecord 列表。
        """
        path = Path(self._file)
        if not path.exists():
            return []

        records: List[TraceRecord] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec = TraceRecord(
                trace_id=d.get("trace_id", ""),
                trace_type=d.get("trace_type", "query"),
                started_at=d.get("started_at", ""),
                finished_at=d.get("finished_at"),
                total_elapsed_ms=d.get("total_elapsed_ms"),
                metadata=d.get("metadata", {}),
                stages=d.get("stages", []),
            )

            if trace_type and rec.trace_type != trace_type:
                continue

            if keyword:
                kw = keyword.lower()
                haystack = (rec.user_query + rec.source_path).lower()
                if kw not in haystack:
                    continue

            records.append(rec)

        # 时间倒序
        records.sort(key=lambda r: r.started_at, reverse=True)
        return records[:limit]

    def get(self, trace_id: str) -> Optional[TraceRecord]:
        """按 trace_id 精确查找"""
        for rec in self.list(limit=10000):
            if rec.trace_id == trace_id:
                return rec
        return None

    def exists(self) -> bool:
        return Path(self._file).exists()

    def file_path(self) -> str:
        return str(Path(self._file).resolve())
