"""SparseEncoder 实现 (src/ingestion/embedding/sparse_encoder.py)"""
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
