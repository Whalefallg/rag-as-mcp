"""核心数据契约 (src/core/types.py)"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ImageRef:
    """文档中一张图片的引用信息"""
    image_id: str
    file_path: str
    page: Optional[int]
    seq: int
    width: Optional[int] = None
    height: Optional[int] = None
    caption: Optional[str] = None


@dataclass
class Document:
    """Loader 的输出产物：一个解析后的文档"""
    id: str
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Splitter 的输出产物：Document 的一个文本片段"""
    id: str
    doc_id: str
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    """Embed + Upsert 阶段的最终存储单元"""
    id: str
    text: str
    metadata: Dict[str, Any]
    dense_vector: List[float] = field(default_factory=list)
    sparse_vector: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Retrieval Pipeline 的统一输出单元。

    为什么要独立定义而不用 dict：
      Dense 召回、Sparse 召回、Fusion、Reranker 每一层都会传递检索结果。
      用 dataclass 而非 dict，可以在 IDE 中自动补全字段名，
      避免 result["chunkid"] 这类拼写错误在运行时才暴露。
    """
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 可选：标记结果来源（dense / sparse / fused / reranked），便于调试
    source: Optional[str] = None


@dataclass
class ProcessedQuery:
    """
    QueryProcessor 的输出产物。

    为什么要拆成 keywords + filters：
      BM25 需要关键词列表（词粒度），VectorStore 需要 metadata 过滤条件（键值对），
      LLM 生成回答需要原始 query 文本。把三种需求合并到一个结构里，
      HybridSearch 只需调用一次 QueryProcessor 就能拿到所有后续组件需要的输入。
    """
    original: str                               # 用户原始 query
    keywords: List[str]                         # 给 BM25 用的关键词列表
    filters: Dict[str, Any] = field(default_factory=dict)  # 给 VectorStore 的前置过滤条件


@dataclass
class DocumentInfo:
    """DocumentManager 列表接口返回的文档概要"""
    source: str
    doc_id: str
    collection: str
    chunk_count: int
    image_count: int
    ingested_at: Optional[str] = None


@dataclass
class DeleteResult:
    """DocumentManager 删除操作的返回结果"""
    source: str
    chunks_deleted: int
    bm25_removed: bool
    images_deleted: int
    history_removed: bool


@dataclass
class IngestionProgress:
    """Pipeline 进度回调数据结构"""
    stage: str
    current: int
    total: int
    message: str = ""
