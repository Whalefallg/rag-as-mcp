"""
配置管理模块 (src/core/settings.py)
=====================================
为什么需要这个文件：
  RAG 系统有很多可变参数——用哪家 LLM、向量库存在哪里、chunk 切多大、召回多少条。
  如果这些参数散落在各模块的构造函数里，改一个参数就要改多处代码，上线前很容易漏改。
  把所有参数集中到一份 YAML，只改配置文件就能切换整个系统行为，代码本身不需要动。
  这种模式叫"外部化配置"（Externalized Configuration），是 12-Factor App 的核心原则之一。

  load_settings() 在读取时立刻做校验（validate_settings）：
  provider 名称拼错、chunk_overlap 大于 chunk_size 这类低级错误在启动时就会报错，
  而不是等到运行中途崩溃，让调试更快。

类说明:
  - LLMConfig / EmbeddingConfig / VectorStoreConfig
    / SplitterConfig / RetrievalConfig / RerankConfig :
        各模块的配置 dataclass，字段名与 settings.yaml 中的 key 一一对应。
        用 dataclass 而非 dict 的好处：IDE 自动补全字段名，拼写错误在开发时就暴露。

  - Settings      : 全局配置容器，持有所有子配置 + raw_config（原始 dict，
                    供各模块读取 YAML 中的自定义扩展字段，如 ingestion.chunk_refiner.use_llm）。

  - load_settings(): 入口函数，读取 YAML → 解析 → 校验 → 返回 Settings 实例。
  - validate_settings(): 独立的校验函数，不合法直接 raise ValueError，
                         错误信息明确指出哪个字段不对，方便快速定位。
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import yaml
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str
    model: str
    azure_endpoint: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    backend: str
    persist_path: str


@dataclass
class SplitterConfig:
    """切分器配置"""
    method: str
    chunk_size: int
    chunk_overlap: int


@dataclass
class RetrievalConfig:
    """检索配置"""
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int


@dataclass
class RerankConfig:
    """重排配置"""
    backend: str
    model: Optional[str] = None
    top_m: int = 30


@dataclass
class Settings:
    """全局配置容器，持有所有子配置和原始 YAML dict"""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    splitter: SplitterConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    raw_config: Dict[str, Any]


def load_settings(config_path: str = "config/settings.yaml") -> Settings:
    """
    加载并校验配置文件，返回 Settings 实例。

    Args:
        config_path: 配置文件路径
    Returns:
        Settings: 校验通过的配置对象
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误或缺少必填字段
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"配置文件为空: {config_path}")

    # Secrets stay outside the tracked YAML template. Environment variables
    # override local YAML values when present.
    env_overrides = {
        ("llm", "api_key"): "RAG_LLM_API_KEY",
        ("llm", "azure_endpoint"): "RAG_LLM_AZURE_ENDPOINT",
        ("embedding", "api_key"): "RAG_EMBEDDING_API_KEY",
        ("embedding", "base_url"): "RAG_EMBEDDING_BASE_URL",
        ("vision_llm", "api_key"): "RAG_VISION_API_KEY",
        ("vision_llm", "azure_endpoint"): "RAG_VISION_AZURE_ENDPOINT",
    }
    for (section, field), env_name in env_overrides.items():
        value = os.getenv(env_name)
        if value:
            config.setdefault(section, {})[field] = value

    try:
        llm_cfg = LLMConfig(
            provider=config['llm']['provider'],
            model=config['llm']['model'],
            azure_endpoint=config['llm'].get('azure_endpoint'),
            api_key=config['llm'].get('api_key')
        )

        embedding_cfg = EmbeddingConfig(
            provider=config['embedding']['provider'],
            model=config['embedding']['model'],
            api_key=config['embedding'].get('api_key'),
            base_url=config['embedding'].get('base_url')
        )

        vector_store_cfg = VectorStoreConfig(
            backend=config['vector_store']['backend'],
            persist_path=config['vector_store']['persist_path']
        )

        splitter_cfg = SplitterConfig(
            method=config['splitter']['method'],
            chunk_size=config['splitter']['chunk_size'],
            chunk_overlap=config['splitter']['chunk_overlap']
        )

        retrieval_cfg = RetrievalConfig(
            sparse_backend=config['retrieval']['sparse_backend'],
            fusion_algorithm=config['retrieval']['fusion_algorithm'],
            top_k_dense=config['retrieval']['top_k_dense'],
            top_k_sparse=config['retrieval']['top_k_sparse'],
            top_k_final=config['retrieval']['top_k_final']
        )

        rerank_cfg = RerankConfig(
            backend=config['rerank']['backend'],
            model=config['rerank'].get('model'),
            top_m=config['rerank'].get('top_m', 30)
        )

        settings = Settings(
            llm=llm_cfg,
            embedding=embedding_cfg,
            vector_store=vector_store_cfg,
            splitter=splitter_cfg,
            retrieval=retrieval_cfg,
            rerank=rerank_cfg,
            raw_config=config
        )

        validate_settings(settings)
        return settings

    except KeyError as e:
        raise ValueError(f"配置缺少必填字段: {e}")
    except Exception as e:
        raise ValueError(f"配置解析错误: {e}")


def validate_settings(settings: Settings) -> None:
    """
    校验配置有效性，不合法时 raise ValueError（含字段路径）。

    把校验逻辑独立出来有两个好处：
      1. 单元测试可以直接测试校验规则，不需要准备真实的 YAML 文件
      2. 其他代码（如测试里手动构造的 Settings）也能复用这套校验
    """
    if settings.llm.provider not in ['azure', 'openai', 'ollama', 'deepseek']:
        raise ValueError(f"不支持的 LLM provider: {settings.llm.provider}")

    if settings.embedding.provider not in ['openai', 'azure', 'ollama']:
        raise ValueError(f"不支持的 embedding provider: {settings.embedding.provider}")

    if settings.vector_store.backend not in ['chroma', 'qdrant', 'pinecone']:
        raise ValueError(f"不支持的 vector_store backend: {settings.vector_store.backend}")

    if settings.splitter.method not in ['recursive', 'semantic', 'fixed']:
        raise ValueError(f"不支持的 splitter method: {settings.splitter.method}")

    if settings.splitter.chunk_size <= 0:
        raise ValueError("splitter.chunk_size 必须大于 0")

    if settings.splitter.chunk_overlap < 0:
        raise ValueError("splitter.chunk_overlap 不能为负数")

    if settings.splitter.chunk_overlap >= settings.splitter.chunk_size:
        raise ValueError("splitter.chunk_overlap 必须小于 chunk_size")

    if settings.retrieval.top_k_dense <= 0 or settings.retrieval.top_k_sparse <= 0:
        raise ValueError("retrieval.top_k_dense 和 top_k_sparse 必须大于 0")

    if settings.rerank.backend not in ['none', 'cross_encoder', 'llm']:
        raise ValueError(f"不支持的 rerank backend: {settings.rerank.backend}")
