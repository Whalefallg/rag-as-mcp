"""
ImageCaptioner 降级测试 (tests/unit/test_image_captioner_fallback.py)
=====================================================================
为什么需要这个文件：
  ImageCaptioner 让纯文本检索链路也能看到图片内容（通过 LLM 生成的文字描述）。
  但 Vision LLM 可能未配置、可能抛异常，此时系统不应崩溃——
  降级测试确认：Vision LLM 失败时 chunk 仍然通过，只标记 has_unprocessed_images=True，
  让调用方知道图片内容没有被处理，而不是静默丢弃或抛出致命错误。

验收标准（DEV_SPEC C7）：
  - 启用模式：有 image_refs 时调用 Vision LLM，caption 写入 metadata
  - 降级模式：disabled / 无 vision_llm / 异常时保留 image_refs 并标记 has_unprocessed_images
  - 无 image_refs 的 chunk 直接通过，不触发 Vision LLM 调用
"""
import pytest
from src.core.types import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner


def _make_chunk_with_image(img_path: str = "/fake/img.png") -> Chunk:
    return Chunk(
        id="c1", doc_id="doc1",
        text="text with [IMAGE: img_001]",
        index=0,
        metadata={
            "image_refs": ["img_001"],
            "images": [{"image_id": "img_001", "path": img_path, "page": 0, "seq": 0}],
        },
    )


def _make_chunk_no_image() -> Chunk:
    return Chunk(id="c2", doc_id="doc1", text="plain text", index=1, metadata={})


def _make_settings_enabled():
    from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf",
                                  top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={"ingestion": {"image_captioner": {"enabled": True}}},
    )


def _mock_vision_llm(caption: str = "A diagram showing..."):
    from src.libs.llm.base_llm import ChatResponse
    class MockVisionLLM:
        def chat_with_image(self, text, image_path=None, image_bytes=None, trace=None):
            return ChatResponse(content=caption, model="gpt-4o")
    return MockVisionLLM()


# ── 降级模式：disabled ────────────────────────────────────────────────────────

def test_disabled_does_not_generate_caption():
    captioner = ImageCaptioner()  # enabled=False 默认
    chunk = _make_chunk_with_image()
    result = captioner.transform([chunk])
    assert result[0].metadata.get("has_unprocessed_images") is True
    # 没有 caption
    for img in result[0].metadata.get("images", []):
        assert "caption" not in img or img.get("caption") is None


def test_disabled_no_image_refs_passes_through():
    captioner = ImageCaptioner()
    chunk = _make_chunk_no_image()
    result = captioner.transform([chunk])
    assert "has_unprocessed_images" not in result[0].metadata


# ── 降级模式：无 vision_llm ───────────────────────────────────────────────────

def test_no_vision_llm_marks_unprocessed():
    settings = _make_settings_enabled()
    captioner = ImageCaptioner(settings=settings, vision_llm=None)
    chunk = _make_chunk_with_image()
    result = captioner.transform([chunk])
    assert result[0].metadata["has_unprocessed_images"] is True


# ── 降级模式：Vision LLM 抛出异常 ────────────────────────────────────────────

def test_vision_llm_exception_marks_unprocessed(tmp_path):
    settings = _make_settings_enabled()

    class FailingVisionLLM:
        def chat_with_image(self, **kwargs):
            raise RuntimeError("vision API down")

    # 创建真实图片文件（避免因文件不存在而被跳过）
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captioner = ImageCaptioner(settings=settings, vision_llm=FailingVisionLLM())
    chunk = _make_chunk_with_image(img_path=str(img_file))
    result = captioner.transform([chunk])
    assert result[0].metadata.get("has_unprocessed_images") is True


# ── 启用模式：正常生成 caption ────────────────────────────────────────────────

def test_enabled_with_mock_llm_generates_caption(tmp_path):
    settings = _make_settings_enabled()

    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captioner = ImageCaptioner(settings=settings, vision_llm=_mock_vision_llm("A test diagram"))
    chunk = _make_chunk_with_image(img_path=str(img_file))
    result = captioner.transform([chunk])

    images = result[0].metadata.get("images", [])
    assert len(images) == 1
    assert images[0].get("caption") == "A test diagram"


# ── 无 image_refs 的 chunk 直接通过 ──────────────────────────────────────────

def test_chunk_without_image_refs_passes_unchanged():
    settings = _make_settings_enabled()
    captioner = ImageCaptioner(settings=settings, vision_llm=_mock_vision_llm())
    chunk = _make_chunk_no_image()
    result = captioner.transform([chunk])
    assert result[0].text == "plain text"
    assert "has_unprocessed_images" not in result[0].metadata
