"""
get_document_summary Tool (src/mcp_server/tools/get_document_summary.py)
=========================================================================
为什么需要这个文件：
  用户在看到检索结果后，有时需要了解某份文档的整体概况——
  这份文档是关于什么的？有多少内容？上次更新是什么时候？
  get_document_summary 通过 doc_id（source_path）从向量库 metadata 中
  聚合该文档所有 chunk 的元数据，生成一份文档级摘要。

  与 query_knowledge_hub 的区别：
    - query_knowledge_hub：基于语义相似度，返回与查询最相关的 chunk
    - get_document_summary：基于文档 ID，返回整份文档的概况
  两者互补，前者回答"哪里有答案"，后者回答"这份文档是什么"。
"""
from typing import Any, Dict, List

from src.core.settings import Settings
from src.observability.logger import get_logger

logger = get_logger(__name__)

TOOL_NAME = "get_document_summary"

TOOL_DESCRIPTION = (
    "获取指定文档的摘要与元数据信息，包括标题、摘要、标签、"
    "片段数量、页数、摄入时间等。doc_id 为文档的 source_path。"
)

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "文档唯一标识，通常为 source_path（如 data/documents/default/guide.pdf）",
        },
        "collection": {
            "type": "string",
            "description": "集合名称（默认 'default'）",
            "default": "default",
        },
    },
    "required": ["doc_id"],
}


def execute(arguments: Dict[str, Any], settings: Settings) -> List[Dict[str, Any]]:
    """
    从 ChromaDB 检索指定文档的所有 chunk，聚合生成文档摘要。

    返回格式：TextContent（Markdown 格式的文档概况）
    """
    doc_id: str = arguments.get("doc_id", "").strip()
    if not doc_id:
        return [{"type": "text", "text": "错误：doc_id 不能为空。"}]

    collection_name: str = arguments.get("collection", "default")

    logger.info(f"[get_document_summary] doc_id={doc_id!r} collection={collection_name}")

    # 尝试从 ChromaDB 获取文档信息
    try:
        summary = _get_from_chroma(doc_id, collection_name, settings)
    except Exception as e:
        logger.warning(f"_get_from_chroma raised: {e}")
        summary = {}
    if summary:
        return [{"type": "text", "text": _format_summary(summary)}]

    return [{
        "type": "text",
        "text": (
            f"未找到文档：`{doc_id}`\n\n"
            f"请确认文档已摄取（`python scripts/ingest.py`），"
            f"且 doc_id 与摄取时使用的 source_path 一致。"
        ),
    }]


def _get_from_chroma(
    doc_id: str, collection_name: str, settings: Settings
) -> Dict[str, Any]:
    """从 ChromaDB 聚合文档的所有 chunk metadata"""
    try:
        import chromadb
        persist_path = settings.vector_store.persist_path
        client = chromadb.PersistentClient(path=persist_path)

        try:
            collection = client.get_collection(collection_name)
        except Exception:
            # 尝试不含 collection 过滤直接查找
            return {}

        # 按 source_path 精确匹配
        results = collection.get(
            where={"source_path": doc_id},
            include=["metadatas", "documents"],
            limit=500,
        )
        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []

        if not metadatas:
            return {}

        return _aggregate_metadata(doc_id, metadatas, documents)

    except Exception as e:
        logger.warning(f"ChromaDB query failed for doc_id={doc_id!r}: {e}")
        return {}


def _aggregate_metadata(
    doc_id: str,
    metadatas: List[Dict],
    documents: List[str],
) -> Dict[str, Any]:
    """聚合多个 chunk 的 metadata，生成文档级概况"""
    # 从各 chunk 中提取唯一值
    titles = _unique_nonempty(m.get("title") for m in metadatas)
    summaries = _unique_nonempty(m.get("summary") for m in metadatas)
    all_tags: List[str] = []
    for m in metadatas:
        tags = m.get("tags", [])
        if isinstance(tags, list):
            all_tags.extend(tags)
        elif isinstance(tags, str) and tags:
            all_tags.extend(tags.split(","))

    pages = sorted({
        int(m["page"]) for m in metadatas
        if "page" in m and str(m["page"]).isdigit()
    })

    return {
        "doc_id": doc_id,
        "chunk_count": len(metadatas),
        "page_count": len(pages),
        "pages": pages[:5],     # 最多展示前 5 页页码
        "title": titles[0] if titles else _basename(doc_id),
        "summaries": summaries[:3],
        "tags": list(dict.fromkeys(t.strip() for t in all_tags if t.strip()))[:20],
        "first_chunk_text": (documents[0][:200] if documents else ""),
        "source_path": doc_id,
    }


def _format_summary(summary: Dict[str, Any]) -> str:
    """将聚合 metadata 格式化为 Markdown 字符串"""
    lines = [f"## 📄 {summary['title']}\n"]
    lines.append(f"**来源路径**：`{summary['source_path']}`")
    lines.append(f"**片段数量**：{summary['chunk_count']} 个 chunk")
    if summary["page_count"]:
        pages_str = "、".join(str(p) for p in summary["pages"])
        if summary["page_count"] > 5:
            pages_str += f"… 共 {summary['page_count']} 页"
        lines.append(f"**涉及页码**：{pages_str}")

    if summary["tags"]:
        lines.append(f"**标签**：{' · '.join(summary['tags'])}")

    if summary["summaries"]:
        lines.append("\n**内容摘要**：")
        for i, s in enumerate(summary["summaries"], 1):
            lines.append(f"{i}. {s}")

    if summary["first_chunk_text"]:
        lines.append(f"\n**首段预览**：\n> {summary['first_chunk_text']}…")

    return "\n".join(lines)


def _unique_nonempty(values) -> List[str]:
    seen = set()
    result = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
