"""
SparseEncoder 实现 (src/ingestion/embedding/sparse_encoder.py)
==============================================================
为什么需要这个文件：
  BM25 检索分两步：Ingestion 时为每个 chunk 计算词频（TF），
  Retrieval 时再结合全语料的 IDF 算出 BM25 分数。
  SparseEncoder 负责 Ingestion 侧的 TF 计算，输出 {term: weight} 字典。
  分两步的原因：IDF 依赖全语料统计，只有所有 chunk 都处理完才能算（BM25Indexer.build 时）；
  而 TF 是单 chunk 级别的，可以在流水线中逐 chunk 实时计算。

本文件实现 Chunk 的稀疏向量编码器（BM25 统计）。

类说明:
  - SparseEncoder : 对 Chunk 列表计算 BM25 所需的词频统计，输出 term weights 结构。
                    输出格式为 {term: tf_normalized} 字典，供 BM25Indexer 消费。
                    空文本返回空字典，不抛异常。
                    当前实现：词级别 TF（词频/文档总词数），不依赖外部库。
                    BM25 的 IDF 部分由 BM25Indexer（C11）在全语料层面计算。
"""
import re
from typing import List, Dict, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext

# 分词：提取字母/数字词和中文字符
_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    """简单分词：英文按词切分（小写），中文按字切分"""
    return [t.lower() for t in _TOKENIZE_RE.findall(text)]


class SparseEncoder:
    """Chunk 稀疏向量编码器（BM25 TF 统计）"""

    def __init__(self, settings: Optional[Settings] = None):
        pass  # 当前实现不依赖 settings，为后续扩展保留接口

    def encode(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        """
        计算每个 Chunk 的 term weights，写入 Chunk.metadata["sparse_vector"]。

        Args:
            chunks: 待编码的 Chunk 列表。
            trace: 追踪上下文（可选）。
        Returns:
            附加了 sparse_vector 的 Chunk 列表。
        """
        result = []
        for chunk in chunks:
            sparse = self._compute_term_weights(chunk.text)
            result.append(Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                index=chunk.index,
                metadata={**chunk.metadata, "sparse_vector": sparse},
            ))
        return result

    def _compute_term_weights(self, text: str) -> Dict[str, float]:
        """计算文本的归一化词频字典 {term: tf}"""
        if not text or not text.strip():
            return {}

        tokens = _tokenize(text)
        if not tokens:
            return {}

        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        total = len(tokens)
        return {term: count / total for term, count in freq.items()}
