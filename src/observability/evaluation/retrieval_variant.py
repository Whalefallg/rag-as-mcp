"""Evaluation-only adapters around production retrieval primitives."""
from __future__ import annotations

from src.core.agentic.orchestrator import AgenticRAGOrchestrator
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import CoreReranker
from src.core.query_engine.sparse_retriever import SparseRetriever


class RetrievalVariant:
    def __init__(self, name, settings):
        self.name, self.settings = name, settings
        self.query_processor = QueryProcessor()
        self.dense = DenseRetriever(settings)
        self.sparse = SparseRetriever(settings)
        self.hybrid = HybridSearch(settings, self.query_processor, self.dense, self.sparse)
        self.reranker = None if settings.rerank.backend == "none" else CoreReranker(settings)
        self.agent = AgenticRAGOrchestrator(settings, self.hybrid, self.reranker if name == "agentic_rerank" else None)

    @property
    def skip_reason(self):
        return "rerank.backend=none; no real reranker configured" if self.name.endswith("_rerank") and self.reranker is None else None

    def search(self, query, top_k, collection):
        if self.name == "bm25":
            processed = self.query_processor.process(query)
            return self.sparse.retrieve(processed.keywords, top_k=top_k, collection=collection)
        if self.name == "dense": return self.dense.retrieve(query, top_k=top_k, collection=collection)
        if self.name == "hybrid": return self.hybrid.search(query, top_k=top_k, collection=collection)
        if self.name == "hybrid_rerank":
            return self.reranker.rerank(query, self.hybrid.search(query, top_k=max(top_k, self.settings.rerank.top_m), collection=collection), top_k)
        result = self.agent.search(query, top_k=top_k, collection=collection)
        return result.results, result


def create_variant(name, settings):
    if name not in {"bm25", "dense", "hybrid", "hybrid_rerank", "agentic", "agentic_rerank"}: raise ValueError(name)
    return RetrievalVariant(name, settings)
