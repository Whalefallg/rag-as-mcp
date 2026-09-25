"""
Vision LLM 单元测试 (tests/unit/test_vision_llm.py)
====================================================
为什么需要这个文件：
  多模态 RAG 的图片理解依赖 Vision LLM，一旦 API 不可用或图片返回为空，
  就会导致 Chunk 的 caption 缺失，检索时图片内容完全不可见。
  测试的重点：未提供图片时抛出 ValueError而不是静默失败，
  base64 工具方法正确（是 Vision API 消费图片的基础），
  以及工厂路由正确连接 AzureVisionLLM。

验收标准（DEV_SPEC B8/B9）：
  - BaseVisionLLM 的 base64 工具方法正确
  - 未提供图片时抛出 ValueError
  - AzureVisionLLM 的注册装饰器生效（注册进 Vision 工厂）
  - Vision LLM 工厂路由正确
  - chat() 在 Vision LLM 上抛出 NotImplementedError（需用 chat_with_image）
"""
import base64
import pytest
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import ChatResponse
from src.libs.llm.llm_factory import get_supported_vision_providers, create_vision_llm
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig


# ── Fake Vision LLM（测试专用）────────────────────────────────────────────────

class FakeVisionLLM(BaseVisionLLM):
    """不调用真实 API 的 Vision LLM 实现"""

    def chat_with_image(self, text, image_path=None, image_bytes=None, trace=None):
        if image_path is None and image_bytes is None:
            raise ValueError("必须提供 image_path 或 image_bytes 之一")
        return ChatResponse(content=f"description of image: {text}", model=self.model)


# ── base64 工具方法测试 ───────────────────────────────────────────────────────

def _make_minimal_png() -> bytes:
    """生成一个最小的合法 1×1 白色 PNG 字节序列"""
    import struct, zlib
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat_data = zlib.compress(b"\x00\xff\xff\xff")
    idat = chunk(b"IDAT", idat_data)
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def test_bytes_to_base64_roundtrip():
    raw = b"hello bytes"
    b64 = BaseVisionLLM._bytes_to_base64(raw)
    assert base64.b64decode(b64) == raw


def test_load_image_as_base64(tmp_path):
    png = _make_minimal_png()
    img_file = tmp_path / "test.png"
    img_file.write_bytes(png)
    b64 = BaseVisionLLM._load_image_as_base64(str(img_file))
    assert base64.b64decode(b64) == png


def test_load_image_missing_file_raises():
    with pytest.raises(ValueError, match="图片文件不存在"):
        BaseVisionLLM._load_image_as_base64("/nonexistent/path/image.png")


# ── FakeVisionLLM 功能测试 ────────────────────────────────────────────────────

def test_fake_vision_llm_chat_with_image_bytes():
    llm = FakeVisionLLM(model="gpt-4o")
    resp = llm.chat_with_image(text="describe this", image_bytes=b"fake_image_data")
    assert isinstance(resp, ChatResponse)
    assert "description of image" in resp.content


def test_fake_vision_llm_no_image_raises():
    llm = FakeVisionLLM(model="gpt-4o")
    with pytest.raises(ValueError, match="必须提供"):
        llm.chat_with_image(text="describe this")


def test_base_vision_llm_chat_raises_not_implemented():
    """Vision LLM 上调用 chat() 应当抛出 NotImplementedError"""
    llm = FakeVisionLLM(model="test")
    with pytest.raises(NotImplementedError):
        from src.libs.llm.base_llm import ChatMessage
        llm.chat([ChatMessage(role="user", content="hello")])


# ── Vision LLM 工厂测试 ───────────────────────────────────────────────────────

def test_azure_vision_llm_registered():
    """AzureVisionLLM 通过装饰器注册后，工厂能识别 'azure' provider"""
    # 导入触发注册
    import src.libs.llm.azure_vision_llm  # noqa: F401
    assert "azure" in get_supported_vision_providers()


def _make_settings_with_vision():
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o", azure_endpoint="https://fake.openai.azure.com"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf", top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


def test_create_vision_llm_azure():
    import src.libs.llm.azure_vision_llm  # noqa: F401
    from src.libs.llm.azure_vision_llm import AzureVisionLLM
    settings = _make_settings_with_vision()
    vlm = create_vision_llm(settings)
    assert isinstance(vlm, AzureVisionLLM)
