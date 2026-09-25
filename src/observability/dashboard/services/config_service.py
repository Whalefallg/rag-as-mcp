"""
ConfigService (src/observability/dashboard/services/config_service.py)
=======================================================================
为什么需要这个文件：
  Dashboard 各页面都需要展示"当前系统用的什么 LLM / Embedding / VectorStore"。
  ConfigService 把 Settings 解析成 Dashboard 可直接展示的结构化数据，
  避免每个页面自己解析 YAML，同时统一错误处理（配置文件缺失时返回 fallback 信息）。

  动态感知设计：
    页面不硬编码任何 provider 名称，全部从 ConfigService 获取，
    切换 provider 后 Dashboard 自动展示新配置，无需改 UI 代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import load_settings


@dataclass
class ComponentCard:
    """单个可插拔组件的展示卡片"""
    category: str
    provider: str
    model: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """完整的系统配置摘要，供 Dashboard 渲染"""
    components: List[ComponentCard]
    retrieval: Dict[str, Any]
    rerank: Dict[str, Any]
    splitter: Dict[str, Any]
    settings_path: str
    load_error: Optional[str] = None


class ConfigService:
    """
    读取 Settings 并转换为 Dashboard 展示格式。

    Usage:
        svc = ConfigService()
        config = svc.load()
        for card in config.components:
            st.metric(card.category, card.provider)
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self._path = settings_path

    def load(self) -> SystemConfig:
        """加载并解析配置，失败时返回携带 load_error 的 SystemConfig"""
        try:
            s = load_settings(self._path)
            return SystemConfig(
                components=self._extract_components(s),
                retrieval={
                    "sparse_backend": s.retrieval.sparse_backend,
                    "fusion_algorithm": s.retrieval.fusion_algorithm,
                    "top_k_dense": s.retrieval.top_k_dense,
                    "top_k_sparse": s.retrieval.top_k_sparse,
                    "top_k_final": s.retrieval.top_k_final,
                },
                rerank={
                    "backend": s.rerank.backend,
                    "model": s.rerank.model or "—",
                    "top_m": s.rerank.top_m,
                },
                splitter={
                    "method": s.splitter.method,
                    "chunk_size": s.splitter.chunk_size,
                    "chunk_overlap": s.splitter.chunk_overlap,
                },
                settings_path=str(Path(self._path).resolve()),
            )
        except Exception as exc:
            return SystemConfig(
                components=[],
                retrieval={}, rerank={}, splitter={},
                settings_path=self._path,
                load_error=str(exc),
            )

    def _extract_components(self, s) -> List[ComponentCard]:
        return [
            ComponentCard(
                category="LLM",
                provider=s.llm.provider,
                model=s.llm.model,
                details={"azure_endpoint": s.llm.azure_endpoint or "—"},
            ),
            ComponentCard(
                category="Embedding",
                provider=s.embedding.provider,
                model=s.embedding.model,
            ),
            ComponentCard(
                category="VectorStore",
                provider=s.vector_store.backend,
                model=s.vector_store.persist_path,
                details={"persist_path": s.vector_store.persist_path},
            ),
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        """从 ChromaDB 获取集合统计信息（供 Overview 页面显示）"""
        try:
            s = load_settings(self._path)
            import chromadb
            client = chromadb.PersistentClient(path=s.vector_store.persist_path)
            collections = client.list_collections()
            stats = []
            for col in collections:
                try:
                    stats.append({"name": col.name, "chunk_count": col.count()})
                except Exception:
                    stats.append({"name": col.name, "chunk_count": 0})
            return {"collections": stats, "total_collections": len(stats)}
        except Exception as exc:
            return {"collections": [], "total_collections": 0, "error": str(exc)}
