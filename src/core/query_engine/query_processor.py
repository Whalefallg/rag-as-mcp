"""
QueryProcessor (src/core/query_engine/query_processor.py)
==========================================================
为什么需要这个模块：
  用户输入的自然语言 query（比如"如何在 Azure 上配置 OpenAI？"）不能直接丢给
  BM25 或 VectorStore——BM25 需要分好的关键词列表，VectorStore 需要结构化的
  filters 字典。QueryProcessor 是这条链路的"入口标准化器"：一次处理，
  产出所有后续检索组件都能直接消费的 ProcessedQuery，
  避免 DenseRetriever 和 SparseRetriever 各自重复解析同一个 query。

类说明:
  - StopWords       : 停用词集合。BM25 实现要点：为什么要过滤停用词？
                      因为"的"、"是"、"the"、"is" 这类高频词出现在几乎所有文档里，
                      IDF ≈ 0，对排序没有贡献，还会占用索引空间和查询时间。
                      这里只内置一个最小集合，生产环境可替换为完整停用词表。

  - QueryProcessor  : 核心处理器。
                      process() 执行两步：
                        1. 关键词提取：分词 → 去停用词 → 去重 → 保留有效词
                        2. filters 解析：从 query 或外部传入的 filters dict 中
                           提取 collection、doc_type 等过滤条件
                      输出 ProcessedQuery，被 HybridSearch 直接消费。
"""
import re
from typing import List, Dict, Any, Optional

from src.core.types import ProcessedQuery

# 实现说明：这是一个极简停用词表，真实系统通常用 NLTK / jieba 的停用词词典
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
        """
        关键词提取：分词 → 小写 → 去停用词 → 去重 → 按原始顺序保留。

        实现说明：这里用正则分词，生产环境可换成 jieba（中文）或
        spaCy（英文）获得更好的分词效果。
        """
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
