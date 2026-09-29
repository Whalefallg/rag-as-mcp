"""冒烟测试 (tests/unit/test_smoke_imports.py)"""
import pytest


def test_import_mcp_server():
    """测试 mcp_server 包可以导入"""
    import src.mcp_server
    assert src.mcp_server is not None


def test_import_core():
    """测试 core 包可以导入"""
    import src.core
    assert src.core is not None


def test_import_ingestion():
    """测试 ingestion 包可以导入"""
    import src.ingestion
    assert src.ingestion is not None


def test_import_libs():
    """测试 libs 包可以导入"""
    import src.libs
    assert src.libs is not None


def test_import_observability():
    """测试 observability 包可以导入"""
    import src.observability
    assert src.observability is not None


def test_all_subpackages_importable():
    """测试所有子包可以导入"""
    subpackages = [
        "src.core.query_engine",
        "src.core.response",
        "src.core.trace",
        "src.ingestion.chunking",
        "src.ingestion.transform",
        "src.ingestion.embedding",
        "src.ingestion.storage",
        "src.libs.loader",
        "src.libs.llm",
        "src.libs.embedding",
        "src.libs.splitter",
        "src.libs.vector_store",
        "src.libs.reranker",
        "src.libs.evaluator",
        "src.mcp_server.tools",
        "src.observability.dashboard",
        "src.observability.evaluation",
    ]

    for package in subpackages:
        try:
            __import__(package)
        except ImportError as e:
            pytest.fail(f"Failed to import {package}: {e}")
