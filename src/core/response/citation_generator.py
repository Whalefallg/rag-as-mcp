"""
CitationGenerator (src/core/response/citation_generator.py)
============================================================
为什么需要这个文件：
  MCP 协议支持 structuredContent，供高级 Client（如 Claude Desktop）解析并渲染
  结构化引用。CitationGenerator 把 RetrievalResult 列表转换成标准化的 Citation
  列表，让 Client 能展示"回答来自哪个文档的哪一页"，增强用户对 AI 输出的信任。

  与 Markdown 引用标注的关系：
    - Markdown [1][2] 是给人看的（TextContent）
    - structuredContent.citations 是给程序解析的（机器可读）
    - 两者的 index 保持对应，Client 可以联动高亮
"""
from typing import Any, Dict, List

from src.core.types import RetrievalResult


class CitationGenerator:
    """
    从 RetrievalResult 列表生成结构化引用列表。

    每条引用格式：
      {
        "index":    1,            # 与 Markdown [N] 对应
        "chunk_id": "...",        # 向量库中的 chunk 标识
        "source":   "doc.pdf",    # 文件名（不含路径，更简洁）
        "source_path": "...",     # 完整路径（供程序使用）
        "page":     3,            # 页码（无则省略）
        "score":    0.876,        # 融合相关度分数
        "snippet":  "...",        # 前 150 字节摘要
        "tags":     [...],        # 来自 metadata.tags（若有）
      }
    """

    SNIPPET_LENGTH = 150

    def generate(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """生成结构化引用列表，顺序与 Markdown [N] 标注一致"""
        citations = []
        for i, r in enumerate(results, start=1):
            citation: Dict[str, Any] = {
                "index": i,
                "chunk_id": r.chunk_id,
                "source_path": r.metadata.get("source_path", ""),
                "source": _basename(r.metadata.get("source_path", "")),
                "score": round(r.score, 4) if r.score is not None else None,
                "snippet": _truncate(r.text or "", self.SNIPPET_LENGTH),
            }
            # 可选字段：仅在 metadata 中存在时才加入
            if "page" in r.metadata:
                citation["page"] = r.metadata["page"]
            if "tags" in r.metadata:
                citation["tags"] = r.metadata["tags"]
            if "title" in r.metadata:
                citation["title"] = r.metadata["title"]
            if "summary" in r.metadata:
                citation["summary"] = r.metadata["summary"]

            citations.append(citation)
        return citations


def _basename(path: str) -> str:
    """从完整路径提取文件名"""
    if not path:
        return ""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _truncate(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    return text[:length] + "…"
