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
    """
    执行知识库混合检索并返回 MCP content 数组。

    流程：
      1. 参数校验
      2. HybridSearch.search()   → 混合召回 Top-K 候选
      3. Reranker.rerank()       → 精排（backend=none 时透传）
      4. MultimodalAssembler     → 收集关联图片
      5. ResponseBuilder         → 构建 Markdown + ImageContent
    """
    query: str = arguments.get("query", "").strip()
    if not query:
        return [{"type": "text", "text": "错误：查询内容不能为空。"}]

    top_k: int = min(int(arguments.get("top_k", 10)), 50)
    collection: str = arguments.get("collection", "default")

    logger.info(f"[query_knowledge_hub] query={query!r} top_k={top_k} collection={collection}")

    try:
        components = _get_components(settings)
        hybrid_search = components["hybrid_search"]
        reranker = components["reranker"]
        builder = components["builder"]
        assembler = components["assembler"]
    except Exception as e:
        logger.error(f"Failed to initialize search components: {e}", exc_info=True)
        return [{"type": "text", "text": f"检索组件初始化失败，请检查配置。错误：{e}"}]

    # 混合检索
    # rerank.top_m 表示精排前希望至少保留的候选池大小。
    # 用户请求的 top_k 更大时尊重用户请求，因此候选池取两者最大值。
    rerank_enabled = settings.rerank.backend != "none"
    candidate_k = (
        max(top_k, settings.rerank.top_m)
        if rerank_enabled
        else top_k
    )
    try:
        results = hybrid_search.search(
            query=query,
            top_k=candidate_k,
            collection=collection,
        )
    except Exception as e:
        logger.error(f"HybridSearch error: {e}", exc_info=True)
        return [{"type": "text", "text": f"检索时发生错误：{e}"}]

    # 精排
    if results:
        try:
            results = reranker.rerank(
                query=query,
                candidates=results,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"Reranker failed, using fusion results: {e}")
            # 精排失败时降级使用融合结果，不中断

    # 多模态组装
    image_contents = assembler.assemble(results, max_images=3)

    # 构建响应
    content = builder.build(results, query, image_contents)
    return content
