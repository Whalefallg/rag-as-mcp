"""
Pipeline 进度回调 & Ingestion trace 打点测试 (tests/unit/test_pipeline_trace.py)
================================================================================
验收标准 (DEV_SPEC F5 / F6)：
  - on_progress(stage, current, total) 在每个阶段被正确调用
  - run() 结束后 trace.stages 包含 integrity/load/split/transform/encode/upsert
  - 幂等跳过时 trace 记录 status="skipped"
  - PipelineError 时 trace 仍被 collect（finally 保证）
  - 进度回调抛异常时不中断主流程
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.core.trace.trace_context import TraceContext
from src.ingestion.pipeline import IngestionPipeline, PipelineError


def _make_settings():
    from src.core.settings import (
        Settings, LLMConfig, EmbeddingConfig,
        VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig,
    )
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="k"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="/tmp/tc"),
        splitter=SplitterConfig(method="recursive", chunk_size=512, chunk_overlap=64),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10,
        ),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def _make_pipeline(settings, on_progress=None, **mock_overrides):
    """构造一个所有外部依赖都被 mock 的 IngestionPipeline"""
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._settings = settings
    pipeline._collection = "default"
    pipeline._legacy_progress = lambda p: None
    pipeline._on_progress = on_progress

    # 构造各种 mock 组件
    pipeline._integrity = MagicMock()
    pipeline._integrity.compute_sha256.return_value = "fakehash"
    pipeline._integrity.should_skip.return_value = False

    mock_doc = MagicMock()
    mock_doc.id = "doc-id"
    mock_doc.metadata = {"images": []}
    pipeline._loader = MagicMock()
    pipeline._loader.load.return_value = mock_doc

    mock_chunks = [MagicMock() for _ in range(3)]
    pipeline._chunker = MagicMock()
    pipeline._chunker.split_document.return_value = mock_chunks

    pipeline._refiner = MagicMock()
    pipeline._refiner.transform.return_value = mock_chunks
    pipeline._enricher = MagicMock()
    pipeline._enricher.transform.return_value = mock_chunks
    pipeline._captioner = MagicMock()
    pipeline._captioner.transform.return_value = mock_chunks

    pipeline._batch = MagicMock()
    pipeline._batch.encode_all.return_value = mock_chunks

    pipeline._upserter = MagicMock()
    pipeline._bm25 = MagicMock()
    pipeline._img_store = MagicMock()

    # 应用覆盖
    for attr, val in mock_overrides.items():
        setattr(pipeline, attr, val)

    return pipeline


class TestOnProgressCallback:
    def test_all_stages_notified(self):
        settings = _make_settings()
        calls = []
        pipeline = _make_pipeline(settings, on_progress=lambda s, c, t: calls.append(s))
        trace = TraceContext(trace_type="ingestion")

        with patch("src.core.trace.trace_collector.global_collector") as mock_col:
            pipeline.run.__func__(pipeline, "/fake/doc.pdf", trace=trace, collect_trace=False)

        stage_names = calls
        assert "load" in stage_names
        assert "split" in stage_names
        assert "transform" in stage_names
        assert "encode" in stage_names
        assert "upsert" in stage_names

    def test_progress_args_are_stage_current_total(self):
        settings = _make_settings()
        records = []
        pipeline = _make_pipeline(
            settings,
            on_progress=lambda s, c, t: records.append((s, c, t))
        )
        trace = TraceContext(trace_type="ingestion")
        pipeline.run.__func__(pipeline, "/fake/doc.pdf", trace=trace, collect_trace=False)

        # 找到 load 阶段的两次回调
        load_calls = [(c, t) for s, c, t in records if s == "load"]
        assert len(load_calls) >= 1
        # 第一次 current=0, total=1（开始），第二次 current=1, total=1（完成）
        assert load_calls[0] == (0, 1)

    def test_callback_exception_does_not_abort_pipeline(self):
        settings = _make_settings()

        def _bad_callback(s, c, t):
            raise RuntimeError("callback error")

        pipeline = _make_pipeline(settings, on_progress=_bad_callback)
        trace = TraceContext(trace_type="ingestion")
        # 进度回调抛异常时流程应正常完成
        result = pipeline.run.__func__(pipeline, "/fake/doc.pdf",
                                       trace=trace, collect_trace=False)
        assert result["skipped"] is False

    def test_none_callback_ok(self):
        settings = _make_settings()
        pipeline = _make_pipeline(settings, on_progress=None)
        trace = TraceContext(trace_type="ingestion")
        result = pipeline.run.__func__(pipeline, "/fake/doc.pdf",
                                       trace=trace, collect_trace=False)
        assert result["skipped"] is False


class TestPipelineTrace:
    def test_trace_has_integrity_stage(self):
        settings = _make_settings()
        pipeline = _make_pipeline(settings)
        trace = TraceContext(trace_type="ingestion")
        pipeline.run.__func__(pipeline, "/fake/doc.pdf", trace=trace, collect_trace=False)
        stage_names = [s["stage"] for s in trace.to_dict()["stages"]]
        assert "integrity" in stage_names

    def test_trace_has_all_pipeline_stages(self):
        settings = _make_settings()
        pipeline = _make_pipeline(settings)
        trace = TraceContext(trace_type="ingestion")
        pipeline.run.__func__(pipeline, "/fake/doc.pdf", trace=trace, collect_trace=False)
        stage_names = [s["stage"] for s in trace.to_dict()["stages"]]
        for expected in ("integrity", "load", "split", "transform", "encode", "upsert"):
            assert expected in stage_names, f"missing stage: {expected}"

    def test_skipped_file_recorded_in_trace(self):
        settings = _make_settings()
        pipeline = _make_pipeline(settings)
        pipeline._integrity.should_skip.return_value = True   # 模拟已摄取
        trace = TraceContext(trace_type="ingestion")
        result = pipeline.run.__func__(pipeline, "/fake/doc.pdf",
                                       trace=trace, collect_trace=False)
        assert result["skipped"] is True
        stage = trace.to_dict()["stages"][0]
        assert stage["status"] == "skipped"

    def test_trace_collect_called_in_finally(self, tmp_path):
        """即使流程正常，global_collector.collect 应被调用"""
        settings = _make_settings()
        pipeline = _make_pipeline(settings)
        trace = TraceContext(trace_type="ingestion")

        with patch("src.ingestion.pipeline.global_collector") as mock_col:
            pipeline.run.__func__(pipeline, "/fake/doc.pdf",
                                  trace=trace, collect_trace=True)
            mock_col.collect.assert_called_once_with(trace)

    def test_trace_collect_called_even_on_error(self):
        """PipelineError 时 finally 仍收集 trace"""
        settings = _make_settings()
        pipeline = _make_pipeline(settings)
        pipeline._loader.load.side_effect = RuntimeError("disk error")

        trace = TraceContext(trace_type="ingestion")

        with patch("src.ingestion.pipeline.global_collector") as mock_col:
            with pytest.raises(PipelineError):
                pipeline.run.__func__(pipeline, "/fake/doc.pdf",
                                      trace=trace, collect_trace=True)
            mock_col.collect.assert_called_once()


class TestQueryTraceIntegration:
    """HybridSearch trace 打点的轻量验证（不需要真实 VectorStore）"""

    def test_hybrid_search_records_stages(self):
        from src.core.query_engine.hybrid_search import HybridSearch
        settings = _make_settings()

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = []
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = []
        mock_fusion = MagicMock()
        mock_fusion.fuse.return_value = []
        mock_proc = MagicMock()

        from src.core.types import ProcessedQuery
        mock_proc.process.return_value = ProcessedQuery(
            original="test",
            keywords=["test"],
            filters={},
        )

        hs = HybridSearch(
            settings,
            query_processor=mock_proc,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=mock_fusion,
        )
        trace = TraceContext(trace_type="query")
        hs.search("test query", trace=trace)

        stage_names = [s["stage"] for s in trace.to_dict()["stages"]]
        assert "dense_retrieval" in stage_names
        assert "sparse_retrieval" in stage_names

    def test_reranker_records_stage(self):
        from src.core.query_engine.reranker import CoreReranker
        from src.core.types import RetrievalResult
        settings = _make_settings()

        mock_backend = MagicMock()
        mock_backend.rerank.return_value = [0, 1]

        reranker = CoreReranker(settings, reranker=mock_backend)
        candidates = [
            RetrievalResult("c1", 0.9, "text1", {}),
            RetrievalResult("c2", 0.8, "text2", {}),
        ]
        trace = TraceContext(trace_type="query")
        results = reranker.rerank("query", candidates, trace=trace)

        stage_names = [s["stage"] for s in trace.to_dict()["stages"]]
        assert "rerank" in stage_names
        rerank_stage = next(s for s in trace.to_dict()["stages"] if s["stage"] == "rerank")
        assert rerank_stage.get("fallback") is False

    def test_reranker_fallback_records_fallback_true(self):
        from src.core.query_engine.reranker import CoreReranker
        from src.core.types import RetrievalResult
        settings = _make_settings()

        mock_backend = MagicMock()
        mock_backend.rerank.side_effect = RuntimeError("model unavailable")

        reranker = CoreReranker(settings, reranker=mock_backend)
        candidates = [RetrievalResult("c1", 0.9, "text1", {})]
        trace = TraceContext(trace_type="query")
        results = reranker.rerank("query", candidates, trace=trace)

        rerank_stage = next(s for s in trace.to_dict()["stages"] if s["stage"] == "rerank")
        assert rerank_stage.get("fallback") is True
        assert len(results) == 1   # 降级仍返回结果
        assert results[0].metadata.get("fallback") is True
