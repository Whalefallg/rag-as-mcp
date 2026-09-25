"""
QueryProcessor 单元测试 (tests/unit/test_query_processor.py)
============================================================
验收标准（DEV_SPEC D1）：
  - 对非空 query 输出 keywords 非空
  - filters 为 dict（可为空）
  - 停用词被过滤掉
  - 空 query 抛 ValueError
  - 关键词去重
  - 全停用词 query 有保底关键词
"""
import pytest
from src.core.query_engine.query_processor import QueryProcessor
from src.core.types import ProcessedQuery


@pytest.fixture
def proc():
    return QueryProcessor()


def test_process_returns_processed_query(proc):
    result = proc.process("how to configure Azure OpenAI")
    assert isinstance(result, ProcessedQuery)
    assert result.original == "how to configure Azure OpenAI"


def test_keywords_non_empty(proc):
    result = proc.process("machine learning model training")
    assert len(result.keywords) > 0


def test_stopwords_filtered(proc):
    result = proc.process("how to configure the system")
    # "how", "to", "the" 应被过滤掉
    for kw in result.keywords:
        assert kw not in {"how", "to", "the", "is", "a", "an"}


def test_keywords_lowercased(proc):
    result = proc.process("Azure OpenAI Configuration")
    for kw in result.keywords:
        assert kw == kw.lower()


def test_keywords_deduplicated(proc):
    result = proc.process("machine learning machine learning")
    assert len(result.keywords) == len(set(result.keywords))


def test_filters_default_empty_dict(proc):
    result = proc.process("some query")
    assert isinstance(result.filters, dict)


def test_filters_passed_through(proc):
    result = proc.process("query", filters={"collection": "my_kb", "doc_type": "pdf"})
    assert result.filters["collection"] == "my_kb"
    assert result.filters["doc_type"] == "pdf"


def test_empty_query_raises(proc):
    with pytest.raises(ValueError):
        proc.process("")


def test_whitespace_only_query_raises(proc):
    with pytest.raises(ValueError):
        proc.process("   ")


def test_all_stopwords_has_fallback(proc):
    # "is the a" 全是停用词，保底返回原始 query 作为关键词
    result = proc.process("is the a")
    assert len(result.keywords) > 0


def test_chinese_keywords(proc):
    result = proc.process("机器学习模型训练优化")
    assert len(result.keywords) > 0


def test_mixed_language_query(proc):
    result = proc.process("Azure OpenAI 配置教程")
    assert len(result.keywords) >= 1


def test_short_tokens_filtered(proc):
    # 单字英文词（长度 < 2）应被过滤
    result = proc.process("a b c machine learning")
    for kw in result.keywords:
        assert len(kw) >= 2


def test_original_preserved(proc):
    q = "  Hello World  "
    result = proc.process(q)
    assert result.original == "Hello World"
