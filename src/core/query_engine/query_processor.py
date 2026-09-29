"""QueryProcessor (src/core/query_engine/query_processor.py)"""
import re
from typing import List, Dict, Any, Optional

from src.core.types import ProcessedQuery

# Deliberately small built-in set; callers can inject a domain-specific set.
_STOP_WORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "as", "into", "from", "and",
    "or", "but", "not", "what", "how", "when", "where", "who", "which",
}
_STOP_WORDS_ZH = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
                  "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去"}
STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_ZH

_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


class QueryProcessor:
    """
    用户 query 的标准化入口。

    把自然语言 query 转为 BM25 关键词列表 + VectorStore filters，
    供 DenseRetriever 和 SparseRetriever 共同消费。
    """

    def __init__(self, stop_words: Optional[set] = None):
        """
        Args:
            stop_words: 自定义停用词集合，默认使用内置英中双语停用词表。
        """
        self._stop_words = stop_words if stop_words is not None else STOP_WORDS

    def process(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> ProcessedQuery:
        """
        处理用户 query，输出标准化的 ProcessedQuery。

        Args:
            query: 用户原始查询文本。
            filters: 外部传入的元数据过滤条件（可选），例如
                     {"collection": "my_kb", "doc_type": "pdf"}。
        Returns:
            ProcessedQuery：含 original / keywords / filters 三个字段。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        keywords = self._extract_keywords(query)
        resolved_filters = filters or {}

        return ProcessedQuery(
            original=query.strip(),
            keywords=keywords,
            filters=resolved_filters,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract unique normalized terms while preserving input order."""
        tokens = _TOKENIZE_RE.findall(query)
        seen = set()
        keywords = []
        for token in tokens:
            lower = token.lower()
            if lower not in self._stop_words and len(lower) >= 2 and lower not in seen:
                seen.add(lower)
                keywords.append(lower)
        # 保底：若全被过滤掉，把原始 query 当一个关键词
        if not keywords:
            keywords = [query.strip().lower()]
        return keywords
