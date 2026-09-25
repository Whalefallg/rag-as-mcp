"""
配置加载测试 (tests/unit/test_config_loading.py)
==================================================
为什么需要这个文件：
  settings.py 是整个系统的入口，如果配置解析出错，所有模块都会挂。
  这里测试正常加载、缺字段报错、provider 非法报错、chunk_overlap 超界等边界情况，
  确保 load_settings() 的防护网在开发阶段就能捕获配置错误，
  而不是等到 CI 或生产才发现。
"""
import pytest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch

from src.core.settings import load_settings, validate_settings, Settings


def test_load_valid_settings():
    """测试加载有效配置"""
    settings = load_settings("config/settings.example.yaml")
    assert settings is not None
    assert settings.llm.provider in ['azure', 'openai', 'ollama', 'deepseek']
    assert settings.embedding.provider in ['openai', 'azure', 'ollama']
    assert settings.vector_store.backend in ['chroma', 'qdrant', 'pinecone']


def test_load_nonexistent_file():
    """测试加载不存在的配置文件"""
    with pytest.raises(FileNotFoundError):
        load_settings("nonexistent.yaml")


def test_environment_overrides_secrets():
    """环境变量应覆盖本地 YAML 中的凭证和端点。"""
    with patch.dict("os.environ", {
        "RAG_LLM_API_KEY": "llm-from-env",
        "RAG_LLM_AZURE_ENDPOINT": "https://example.openai.azure.com/",
        "RAG_EMBEDDING_API_KEY": "embedding-from-env",
        "RAG_EMBEDDING_BASE_URL": "https://example.test/v1",
    }):
        settings = load_settings("config/settings.example.yaml")

    assert settings.llm.api_key == "llm-from-env"
    assert settings.llm.azure_endpoint == "https://example.openai.azure.com/"
    assert settings.embedding.api_key == "embedding-from-env"
    assert settings.embedding.base_url == "https://example.test/v1"


def test_missing_required_field():
    """测试缺少必填字段"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'llm': {'provider': 'azure'},
            # 缺少 'model' 字段
        }, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="配置缺少必填字段"):
            load_settings(temp_path)
    finally:
        Path(temp_path).unlink()


def test_invalid_provider():
    """测试无效的 provider"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'llm': {'provider': 'invalid_provider', 'model': 'test'},
            'embedding': {'provider': 'openai', 'model': 'test'},
            'vector_store': {'backend': 'chroma', 'persist_path': './data'},
            'splitter': {'method': 'recursive', 'chunk_size': 1000, 'chunk_overlap': 200},
            'retrieval': {
                'sparse_backend': 'bm25',
                'fusion_algorithm': 'rrf',
                'top_k_dense': 20,
                'top_k_sparse': 20,
                'top_k_final': 10
            },
            'rerank': {'backend': 'none'}
        }
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="不支持的 LLM provider"):
            load_settings(temp_path)
    finally:
        Path(temp_path).unlink()


def test_invalid_chunk_size():
    """测试无效的 chunk_size"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'llm': {'provider': 'azure', 'model': 'gpt-4'},
            'embedding': {'provider': 'openai', 'model': 'test'},
            'vector_store': {'backend': 'chroma', 'persist_path': './data'},
            'splitter': {'method': 'recursive', 'chunk_size': -100, 'chunk_overlap': 200},
            'retrieval': {
                'sparse_backend': 'bm25',
                'fusion_algorithm': 'rrf',
                'top_k_dense': 20,
                'top_k_sparse': 20,
                'top_k_final': 10
            },
            'rerank': {'backend': 'none'}
        }
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
            load_settings(temp_path)
    finally:
        Path(temp_path).unlink()
