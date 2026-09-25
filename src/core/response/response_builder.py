"""
ResponseBuilder (src/core/response/response_builder.py)
========================================================
为什么需要这个文件：
  HybridSearch 返回的是 List[RetrievalResult]——纯数据，没有格式。
  MCP Tool 需要返回 content 数组（TextContent + 可选 ImageContent），
  格式要求严格，且需要人类可读的 Markdown 引用标注。
  ResponseBuilder 把"数据"转换成"MCP 响应格式"，职责单一，便于测试和修改。

  输出格式：
    content[0]: TextContent（Markdown，含 [1][2] 引用标注）
    content[1..n]: ImageContent（Base64 图片，可选）
    structuredContent: 机器可读的结构化引用（供高级 Client 解析）
"""
import time
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext


class ResponseBuilder:
    """
    将 RetrievalResult 列表转换为 MCP content 数组。

    MCP content 类型：
      TextContent  → {"type": "text", "text": "..."}
      ImageContent → {"type": "image", "data": "<base64>", "mimeType": "image/png"}
    """

    def __init__(self, max_text_length: int = 500) -> None:
        """
        Args:
            max_text_length: 每个结果片段在 Markdown 里截断的最大字符数。
        """
        self._max_len = max_text_length

    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        image_contents: Optional[List[Dict[str, Any]]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[Dict[str, Any]]:
        """
        构建完整的 MCP content 数组。

        Args:
            results:        排好序的检索结果列表（空时返回友好提示）。
            query:          用户原始查询（用于 Markdown 标题）。
            image_contents: 已编码的 ImageContent 列表（由 MultimodalAssembler 提供）。
        Returns:
            符合 MCP 规范的 content 数组。
        """
        started = time.monotonic()
        if not results:
            content = [{"type": "text", "text": _NO_RESULT_MSG}]
            if trace:
                trace.record_stage(
                    "response_build",
                    duration_ms=(time.monotonic() - started) * 1000,
                    result_count=0,
                    image_count=0,
                )
            return content

        # 1. 构建 Markdown 文本（含引用标注）
        markdown = self._build_markdown(results, query)
        content: List[Dict[str, Any]] = [{"type": "text", "text": markdown}]

        # 2. 追加图片内容（如有）
        if image_contents:
            content.extend(image_contents)

        if trace:
            trace.record_stage(
                "response_build",
                duration_ms=(time.monotonic() - started) * 1000,
                result_count=len(results),
                image_count=len(image_contents or []),
            )
        return content

    def build_with_structured(
        self,
        results: List[RetrievalResult],
        query: str,
        image_contents: Optional[List[Dict[str, Any]]] = None,
        trace: Optional[TraceContext] = None,
    ) -> Dict[str, Any]:
        """
        构建包含 structuredContent 的完整 tool 响应体。

        返回格式与 MCP tools/call 响应的 result 字段对应：
          {"content": [...], "structuredContent": {"citations": [...]}}
        """
        from src.core.response.citation_generator import CitationGenerator
        content = self.build(
            results,
            query,
            image_contents,
            trace=trace,
        )
        citations = CitationGenerator().generate(results)
        return {
            "content": content,
            "structuredContent": {"query": query, "citations": citations},
        }

    # ──────────────────────────────────────────────────────────────────────

    def _build_markdown(self, results: List[RetrievalResult], query: str) -> str:
        lines = [f"**检索结果**（共 {len(results)} 条相关内容）\n"]
        for i, r in enumerate(results, start=1):
            source = r.metadata.get("source_path", "未知来源")
            page = r.metadata.get("page", "")
            page_str = f"，第 {page} 页" if page else ""
            score_str = f"{r.score:.3f}" if r.score is not None else "N/A"

            # 截断长文本
            text = r.text or ""
            if len(text) > self._max_len:
                text = text[: self._max_len] + "…"

            lines.append(
                f"**[{i}]** `{source}`{page_str}（相关度: {score_str}）\n"
                f"> {text}\n"
            )

        lines.append(_CITATION_FOOTER)
        return "\n".join(lines)


_NO_RESULT_MSG = (
    "未找到相关内容。请检查知识库是否已完成数据摄取，"
    "或尝试换用不同的关键词重新查询。"
)

_CITATION_FOOTER = (
    "\n---\n*以上内容来自本地知识库，引用编号 [N] 对应上方来源。*"
)
