"""
list_collections Tool (src/mcp_server/tools/list_collections.py)
=================================================================
为什么需要这个文件：
  用户在查询前需要知道知识库里有哪些集合（collection）可选。
  这个 tool 列出所有已摄取的集合名称及其统计信息，
  让 MCP Client 可以引导用户选择目标集合，或自动推断合适的集合。

  数据来源优先级：
    1. ChromaStore（最准确，来自实际向量数据）
    2. data/documents/ 目录（回退，仅有目录结构时使用）
    3. settings.yaml 默认集合名（最终回退）
"""
from pathlib import Path
from typing import Any, Dict, List

from src.core.settings import Settings
from src.observability.logger import get_logger

logger = get_logger(__name__)

TOOL_NAME = "list_collections"

TOOL_DESCRIPTION = (
    "列出知识库中所有可用的文档集合（collection）及其统计信息。"
    "在调用 query_knowledge_hub 前可先调用此工具确认目标集合。"
)

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


def execute(arguments: Dict[str, Any], settings: Settings) -> List[Dict[str, Any]]:
    """
    返回所有已知集合的列表。

    优先从 ChromaDB 读取真实数据；若 Chroma 不可用则扫描目录结构作为回退。
    """
    collections = _get_from_chroma(settings) or _get_from_filesystem(settings)

    if not collections:
        return [{
            "type": "text",
            "text": "知识库中暂无文档集合。请先使用 ingest.py 摄取文档。",
        }]

    lines = ["**可用的知识库集合**\n"]
    for col in collections:
        name = col.get("name", "?")
        count = col.get("chunk_count", 0)
        doc_count = col.get("document_count", "?")
        lines.append(f"- **{name}**：{count} 个片段，约 {doc_count} 份文档")

    lines.append(f"\n共 {len(collections)} 个集合。")
    lines.append("在 `query_knowledge_hub` 的 `collection` 参数中指定集合名称以限定检索范围。")

    return [{"type": "text", "text": "\n".join(lines)}]


def _get_from_chroma(settings: Settings) -> List[Dict[str, Any]]:
    """从 ChromaDB 获取集合统计信息"""
    try:
        import chromadb
        persist_path = settings.vector_store.persist_path
        client = chromadb.PersistentClient(path=persist_path)
        collections = client.list_collections()
        result = []
        for col in collections:
            try:
                count = col.count()
                result.append({
                    "name": col.name,
                    "chunk_count": count,
                    "document_count": _estimate_doc_count(col),
                })
            except Exception:
                result.append({"name": col.name, "chunk_count": 0, "document_count": "?"})
        return result
    except Exception as e:
        logger.debug(f"Chroma collection listing failed: {e}")
        return []


def _estimate_doc_count(collection) -> int:
    """通过 metadata.source_path 的唯一值数量估算文档数"""
    try:
        items = collection.get(include=["metadatas"], limit=10000)
        sources = {
            m.get("source_path", "")
            for m in (items.get("metadatas") or [])
            if m.get("source_path")
        }
        return len(sources)
    except Exception:
        return 0


def _get_from_filesystem(settings: Settings) -> List[Dict[str, Any]]:
    """回退：扫描 data/documents/ 目录"""
    try:
        docs_dir = Path("data/documents")
        if not docs_dir.exists():
            return [{"name": "default", "chunk_count": 0, "document_count": 0}]
        collections = []
        for child in sorted(docs_dir.iterdir()):
            if child.is_dir():
                pdf_count = len(list(child.glob("**/*.pdf")))
                collections.append({
                    "name": child.name,
                    "chunk_count": 0,   # 无法从目录知道精确 chunk 数
                    "document_count": pdf_count,
                })
        return collections or [{"name": "default", "chunk_count": 0, "document_count": 0}]
    except Exception as e:
        logger.debug(f"Filesystem collection scan failed: {e}")
        return []
