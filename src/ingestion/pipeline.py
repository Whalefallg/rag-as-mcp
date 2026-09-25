"""
Ingestion Pipeline 编排 (src/ingestion/pipeline.py)
====================================================
为什么需要这个文件：
  把「加载→切分→清洗→编码→写入」这条链路串在一起的编排器。
  没有 Pipeline 的话，每次调用方都要手动依次调 Loader/Chunker/Encoder/Upserter，
  步骤顺序容易搞错，错误处理也会散落各处。
  IngestionPipeline 封装成一次 ingest(file_path) 调用，
  内置幂等跳过（SHA256 检查）、每步失败打标记、进度回调。

  Phase F 新增：
    - trace 注入（trace_type="ingestion"），各阶段 record_stage
    - on_progress 回调签名：(stage_name, current, total) → None
    - trace.finish() + TraceCollector.collect() 在 run() 末尾自动持久化

  设计问题：Pipeline 模式有什么好处？
    → 关注点分离、步骤可替换、错误处理集中、测试时可 mock 单步。

类说明:
  - PipelineError      : 阶段性错误，携带 stage 名和原始异常。
  - IngestionPipeline  : 串行执行七步：
                           integrity → load → split → transform →
                           encode → upsert → store_images
"""
import time
from pathlib import Path
from typing import Callable, List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import global_collector
from src.core.types import IngestionProgress
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage


class PipelineError(Exception):
    """Pipeline 执行阶段错误，包含阶段名称和原始异常"""
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"[{stage}] {cause}")
        self.stage = stage
        self.cause = cause


ProgressCallback = Callable[[IngestionProgress], None]


class IngestionPipeline:
    """MVP Ingestion Pipeline — 串行执行全链路文档摄取"""

    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        llm=None,
        vision_llm=None,
        progress_callback: Optional[ProgressCallback] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self._settings = settings
        self._collection = collection
        self._legacy_progress = progress_callback or (lambda p: None)
        self._on_progress = on_progress   # F6: 新式回调 (stage, current, total)

        self._integrity = SQLiteIntegrityChecker()
        self._loader = PdfLoader()
        self._chunker = DocumentChunker(settings)
        self._refiner = ChunkRefiner(settings, llm=llm)
        self._enricher = MetadataEnricher(settings, llm=llm)
        self._captioner = ImageCaptioner(settings, vision_llm=vision_llm)
        self._batch = BatchProcessor(settings)
        self._upserter = VectorUpserter(settings)
        self._bm25 = BM25Indexer()
        self._img_store = ImageStorage()

    # ──────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        file_path: str,
        force: bool = False,
        trace: Optional[TraceContext] = None,
        collect_trace: bool = True,
    ) -> dict:
        """
        对单个文件执行完整 Ingestion Pipeline。

        Args:
            file_path:     待摄取的文件路径。
            force:         True 时跳过完整性检查（强制重新摄取）。
            trace:         追踪上下文（None 时自动创建 ingestion 类型）。
            collect_trace: True 时在 run() 结束后自动持久化 trace。
        Returns:
            摘要字典：{skipped, chunk_count, image_count, trace_id}
        Raises:
            PipelineError: 任意阶段失败时。
        """
        if trace is None:
            trace = TraceContext(trace_type="ingestion")

        abs_path = str(Path(file_path).resolve())
        file_hash = None
        trace.set_metadata("source_path", abs_path)
        trace.set_metadata("collection", self._collection)

        try:
            result = self._run_internal(abs_path, force, trace)
        finally:
            if collect_trace:
                try:
                    global_collector.collect(trace)
                except Exception:
                    pass   # trace 持久化失败不应影响主流程

        return result

    def run_batch(
        self,
        file_paths: List[str],
        force: bool = False,
    ) -> List[dict]:
        """对多个文件依次执行 Pipeline，返回每个文件的结果"""
        results = []
        for path in file_paths:
            try:
                result = self.run(path, force=force)
                result["file"] = path
                results.append(result)
            except PipelineError as e:
                results.append({"file": path, "error": str(e), "stage": e.stage})
        return results

    # ──────────────────────────────────────────────────────────────────────
    # 内部实现
    # ──────────────────────────────────────────────────────────────────────

    def _run_internal(self, abs_path: str, force: bool, trace: TraceContext) -> dict:
        file_hash = None

        # ── 1. 完整性检查 ──────────────────────────────────────────────────
        _t = time.monotonic()
        try:
            file_hash = self._integrity.compute_sha256(abs_path)
            if not force and self._integrity.should_skip(file_hash):
                trace.record_stage("integrity",
                                   duration_ms=(time.monotonic() - _t) * 1000,
                                   status="skipped", method="sha256")
                return {"skipped": True, "reason": "already_ingested",
                        "trace_id": trace.trace_id}
            trace.record_stage("integrity",
                               duration_ms=(time.monotonic() - _t) * 1000,
                               status="ok", method="sha256")
        except Exception as e:
            raise PipelineError("integrity", e)

        try:
            # ── 2. 加载文档 ────────────────────────────────────────────────
            self._notify("load", 0, 1)
            _t = time.monotonic()
            try:
                document = self._loader.load(abs_path)
            except Exception as e:
                raise PipelineError("load", e)
            trace.record_stage("load",
                               duration_ms=(time.monotonic() - _t) * 1000,
                               method="markitdown",
                               image_count=len(document.metadata.get("images", [])))
            self._notify("load", 1, 1)

            # ── 3. 切分 ────────────────────────────────────────────────────
            self._notify("split", 0, 1)
            _t = time.monotonic()
            try:
                chunks = self._chunker.split_document(document)
            except Exception as e:
                raise PipelineError("split", e)
            n = len(chunks)
            trace.record_stage("split",
                               duration_ms=(time.monotonic() - _t) * 1000,
                               method="recursive",
                               chunk_count=n)
            self._notify("split", n, n)

            # ── 4. Transform 链 ────────────────────────────────────────────
            self._notify("transform", 0, n)
            _t = time.monotonic()
            try:
                chunks = self._refiner.transform(chunks, trace)
                chunks = self._enricher.transform(chunks, trace)
                chunks = self._captioner.transform(chunks, trace)
            except Exception as e:
                raise PipelineError("transform", e)
            trace.record_stage("transform",
                               duration_ms=(time.monotonic() - _t) * 1000,
                               chunk_count=len(chunks))
            self._notify("transform", n, n)

            # ── 5. 编码 ────────────────────────────────────────────────────
            self._notify("encode", 0, n)
            _t = time.monotonic()
            try:
                chunks = self._batch.encode_all(chunks, trace)
            except Exception as e:
                raise PipelineError("encode", e)
            trace.record_stage("encode",
                               duration_ms=(time.monotonic() - _t) * 1000,
                               chunk_count=len(chunks))
            self._notify("encode", n, n)

            # ── 6. 存储向量 + BM25 ─────────────────────────────────────────
            self._notify("upsert", 0, n)
            _t = time.monotonic()
            try:
                self._upserter.upsert(chunks, collection=self._collection, trace=trace)
                self._bm25.build(chunks)
            except Exception as e:
                raise PipelineError("upsert", e)
            trace.record_stage("upsert",
                               duration_ms=(time.monotonic() - _t) * 1000,
                               method="chroma",
                               chunk_count=n)
            self._notify("upsert", n, n)

            # ── 7. 存储图片 ────────────────────────────────────────────────
            image_count = len(document.metadata.get("images", []))
            if image_count:
                self._notify("store_images", 0, image_count)
                _t = time.monotonic()
                try:
                    self._store_images(document)
                except Exception as e:
                    raise PipelineError("store_images", e)
                trace.record_stage("store_images",
                                   duration_ms=(time.monotonic() - _t) * 1000,
                                   image_count=image_count)
                self._notify("store_images", image_count, image_count)

            # ── 成功 ───────────────────────────────────────────────────────
            self._integrity.mark_success(file_hash, abs_path)
            return {
                "skipped": False,
                "chunk_count": n,
                "image_count": image_count,
                "trace_id": trace.trace_id,
            }

        except PipelineError:
            if file_hash:
                try:
                    self._integrity.mark_failed(file_hash, "pipeline_error")
                except Exception:
                    pass
            raise

    def _notify(self, stage: str, current: int, total: int) -> None:
        """触发进度回调（同时支持新旧两种签名）"""
        # 新式回调 F6：(stage_name, current, total)
        if self._on_progress is not None:
            try:
                self._on_progress(stage, current, total)
            except Exception:
                pass
        # 旧式回调（向下兼容）
        try:
            self._legacy_progress(
                IngestionProgress(stage=stage, current=current,
                                  total=total, message="")
            )
        except Exception:
            pass

    def _store_images(self, document) -> None:
        doc_hash = document.id
        for img in document.metadata.get("images", []):
            img_path = img.get("path", "")
            if not img_path:
                continue
            p = Path(img_path)
            if not p.exists():
                continue
            self._img_store.save(
                image_id=img["image_id"],
                image_bytes=p.read_bytes(),
                collection=self._collection,
                doc_hash=doc_hash,
                page_num=img.get("page"),
            )
