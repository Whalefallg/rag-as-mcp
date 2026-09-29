"""Integration tests for retrieval evaluation and report aggregation."""

import json

import pytest

from src.core.settings import (
    EmbeddingConfig, LLMConfig, RerankConfig, RetrievalConfig, Settings,
    SplitterConfig, VectorStoreConfig,
)
from src.core.types import RetrievalResult
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator


def _settings() -> Settings:
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="test"),
        embedding=EmbeddingConfig(
            provider="openai", model="text-embedding-3-small", api_key="test"
        ),
        vector_store=VectorStoreConfig(
            backend="chroma", persist_path="/tmp/rag-as-mcp-eval-test"
        ),
        splitter=SplitterConfig(
            method="recursive", chunk_size=512, chunk_overlap=64
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=2,
        ),
        rerank=RerankConfig(backend="none"),
    )


class DeterministicSearch:
    """Retrieval fake for runner integration; it does not measure quality."""

    def __init__(self):
        self._results = {
            "first": [
                RetrievalResult(
                    "irrelevant", 0.9, "unrelated", {"source_path": "other.pdf"}
                ),
                RetrievalResult(
                    "expected-a", 0.8, "matching", {"source_path": "a.pdf"}
                ),
            ],
            "second": [
                RetrievalResult(
                    "expected-b", 0.9, "matching", {"source_path": "b.pdf"}
                ),
            ],
        }

    def search(self, query, **kwargs):
        return self._results.get(query, [])


@pytest.fixture
def test_set(tmp_path):
    path = tmp_path / "evaluation_cases.json"
    path.write_text(
        json.dumps({
            "test_cases": [
                {"query": "first", "expected_chunk_ids": ["expected-a"]},
                {"query": "second", "expected_chunk_ids": ["expected-b"]},
            ]
        }),
        encoding="utf-8",
    )
    return path


def test_runner_aggregates_exact_metrics(test_set):
    runner = EvalRunner(
        settings=_settings(),
        hybrid_search=DeterministicSearch(),
        evaluator=LocalRetrievalEvaluator(k=2),
    )

    report = runner.run(str(test_set))

    assert report.total_cases == 2
    assert report.failed_cases == 0
    assert report.summary["hit_rate"] == 1.0
    assert report.summary["mrr"] == pytest.approx(0.75)
    assert report.summary["precision_at_k"] == pytest.approx(0.75)


def test_runner_preserves_per_query_evidence(test_set):
    runner = EvalRunner(
        settings=_settings(),
        hybrid_search=DeterministicSearch(),
        evaluator=LocalRetrievalEvaluator(k=2),
    )

    report = runner.run(str(test_set))

    assert report.per_query[0].retrieved_chunk_ids == ["irrelevant", "expected-a"]
    assert report.per_query[1].retrieved_sources == ["b.pdf"]
