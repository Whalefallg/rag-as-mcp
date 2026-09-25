"""
query_knowledge_hub Tool (src/mcp_server/tools/query_knowledge_hub.py)
=======================================================================
为什么需要这个文件：
  这是整个 MCP Server 最核心的 tool——用户通过它查询知识库。
  它把 HybridSearch + Reranker + ResponseBuilder + MultimodalAssembler
  串联成一次完整的 RAG 查询，并以 MCP 规范格式返回带引用的结果。

  tool 参数设计：
    query      必填  用户的自然语言查询
    top_k      可选  返回结果数量（默认 10）
    collection 可选  限定检索的知识库集合（默认 "default"）

  延迟初始化：
    HybridSearch/Reranker 依赖 VectorStore 和 Embedding，
    首次调用时才初始化（懒加载），避免 Server 启动时因缺少数据而失败。
"""
from typing import Any, Dict, List

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import global_collector
from src.observability.logger import get_logger

logger = get_logger(__name__)

TOOL_NAME = "query_knowledge_hub"

TOOL_DESCRIPTION = (
    "在本地知识库中执行混合检索（语义向量 + 关键词 BM25），"
    "返回最相关的文档片段及其来源引用。"
    "支持多模态：若相关片段含有图片，会同时返回图片内容。"
)

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "用户的自然语言查询",
        },
        "top_k": {
            "type": "integer",
            "description": "返回的最大结果数量（默认 10，最大 50）",
            "default": 10,
            "minimum": 1,
            "maximum": 50,
        },
        "collection": {
            "type": "string",
            "description": "目标知识库集合名称（默认 'default'）",
            "default": "default",
        },
    },
    "required": ["query"],
}

# 懒加载缓存（key: settings hash → (hybrid_search, reranker, assembler)）
_cache: Dict[str, Any] = {}


def _get_components(settings: Settings):
    """懒加载并缓存核心检索组件"""
    cache_key = id(settings)
    if cache_key not in _cache:
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.core.query_engine.reranker import Reranker
        from src.core.response.response_builder import ResponseBuilder
        from src.core.response.multimodal_assembler import MultimodalAssembler

        try:
            from src.ingestion.storage.image_storage import ImageStorage
            image_storage = ImageStorage()
        except Exception:
            image_storage = None

        _cache[cache_key] = {
            "hybrid_search": HybridSearch(settings),
            "reranker": Reranker(settings),
            "builder": ResponseBuilder(),
            "assembler": MultimodalAssembler(image_storage=image_storage),
        }
    return _cache[cache_key]


def execute(arguments: Dict[str, Any], settings: Settings) -> List[Dict[str, Any]]:
    """执行知识库混合检索，并让一个 TraceContext 贯穿整条 query 链路。"""
    query: str = arguments.get("query", "").strip()
    top_k: int = min(int(arguments.get("top_k", 10)), 50)
    collection: str = arguments.get("collection", "default")

    trace = TraceContext(trace_type="query")
    trace.set_metadata("user_query", query)
    trace.set_metadata("collection", collection)
    trace.set_metadata("top_k", top_k)
    trace.set_metadata("status", "running")

    try:
        if not query:
            trace.record_stage(
                "validation",
                status="error",
                error="empty query",
            )
            trace.set_metadata("status", "error")
            return [{"type": "text", "text": "错误：查询内容不能为空。"}]

        logger.info(
            f"[query_knowledge_hub] query={query!r} "
            f"top_k={top_k} collection={collection}"
        )

        try:
            components = _get_components(settings)
            hybrid_search = components["hybrid_search"]
            reranker = components["reranker"]
            builder = components["builder"]
            assembler = components["assembler"]
            trace.record_stage("component_init", status="ok")
        except Exception as exc:
            trace.record_stage(
                "component_init",
                status="error",
                error=str(exc),
            )
            trace.set_metadata("status", "error")
            logger.error(
                f"Failed to initialize search components: {exc}",
                exc_info=True,
            )
            return [{
                "type": "text",
                "text": f"检索组件初始化失败，请检查配置。错误：{exc}",
            }]

        rerank_enabled = settings.rerank.backend != "none"
        candidate_k = (
            max(top_k, settings.rerank.top_m)
            if rerank_enabled
            else top_k
        )
        trace.set_metadata("candidate_k", candidate_k)
        trace.set_metadata("rerank_backend", settings.rerank.backend)

        try:
            results = hybrid_search.search(
                query=query,
                top_k=candidate_k,
                collection=collection,
                trace=trace,
            )
        except Exception as exc:
            trace.record_stage(
                "hybrid_search",
                status="error",
                error=str(exc),
            )
            trace.set_metadata("status", "error")
            logger.error(f"HybridSearch error: {exc}", exc_info=True)
            return [{
                "type": "text",
                "text": f"检索时发生错误：{exc}",
            }]

        if results:
            try:
                results = reranker.rerank(
                    query=query,
                    candidates=results,
                    top_k=top_k,
                    trace=trace,
                )
            except Exception as exc:
                trace.record_stage(
                    "rerank_tool_fallback",
                    status="degraded",
                    error=str(exc),
                )
                logger.warning(
                    f"Reranker failed, using fusion results: {exc}"
                )
                results = results[:top_k]

        image_contents = assembler.assemble(
            results,
            max_images=3,
            trace=trace,
        )

        content = builder.build(
            results,
            query,
            image_contents,
            trace=trace,
        )
        trace.set_metadata("result_count", len(results))
        trace.set_metadata("status", "ok")
        return content
    finally:
        global_collector.collect(trace)

