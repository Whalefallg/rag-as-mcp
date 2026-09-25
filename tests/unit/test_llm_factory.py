"""
LLM Factory 单元测试 (tests/unit/test_llm_factory.py)
=====================================================
为什么需要这个文件：
  LLMFactory 是系统切换 LLM 供应商的唯一入口。
  测试工厂路由逻辑，确保 settings.yaml 中写 provider: azure 就真的返回 AzureLLM，
  而不是误返回其他实现或静默失败。
  FakeLLM 的测试顺便验证了 BaseLLM 的消息校验规则，
  这样业务代码在调用 chat() 前能放心依赖校验会替它挡住格式错误的消息。

验收标准（DEV_SPEC B1）：
  - 工厂能根据 provider 名称路由到正确的类
  - 未知 provider 抛出 ValueError 且错误信息可读
  - BaseLLM 的消息校验逻辑正确
  - FakeLLM 的 chat() 正常工作
"""
import pytest
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import register_llm, create_llm, get_supported_providers
from src.core.settings import Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig, SplitterConfig, RetrievalConfig, RerankConfig


# ── Fake 实现（测试专用）─────────────────────────────────────────────────────

@register_llm("fake_llm")
class FakeLLM(BaseLLM):
    def chat(self, messages):
        self._validate_messages(messages)
        return ChatResponse(content="fake response", model=self.model)


def _make_settings(provider="fake_llm", model="test-model"):
    return Settings(
        llm=LLMConfig(provider=provider, model=model),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        splitter=SplitterConfig(method="recursive", chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf", top_k_dense=20, top_k_sparse=20, top_k_final=10),
        rerank=RerankConfig(backend="none"),
        raw_config={},
    )


# ── 工厂路由测试 ──────────────────────────────────────────────────────────────

def test_factory_creates_correct_instance():
    llm = create_llm(_make_settings(provider="fake_llm"))
    assert isinstance(llm, FakeLLM)
    assert llm.model == "test-model"


def test_factory_unknown_provider_raises_valueerror():
    with pytest.raises(ValueError, match="不支持的 LLM provider"):
        create_llm(_make_settings(provider="totally_unknown"))


def test_factory_error_message_lists_registered_providers():
    try:
        create_llm(_make_settings(provider="bad_provider"))
    except ValueError as e:
        assert "fake_llm" in str(e)  # 错误信息应包含已注册的 provider


def test_get_supported_providers_includes_fake():
    assert "fake_llm" in get_supported_providers()


# ── BaseLLM 消息校验测试 ──────────────────────────────────────────────────────

def test_chat_empty_messages_raises():
    llm = FakeLLM(model="test")
    with pytest.raises(ValueError, match="消息列表不能为空"):
        llm.chat([])


def test_chat_invalid_role_raises():
    llm = FakeLLM(model="test")
    with pytest.raises(ValueError, match="无效的消息角色"):
        llm.chat([ChatMessage(role="god", content="hello")])


def test_chat_empty_content_raises():
    llm = FakeLLM(model="test")
    with pytest.raises(ValueError, match="消息内容不能为空"):
        llm.chat([ChatMessage(role="user", content="")])


def test_chat_valid_messages_returns_response():
    llm = FakeLLM(model="my-model")
    resp = llm.chat([ChatMessage(role="user", content="hello")])
    assert isinstance(resp, ChatResponse)
    assert resp.content == "fake response"
    assert resp.model == "my-model"


def test_chat_all_valid_roles():
    llm = FakeLLM(model="test")
    for role in ("system", "user", "assistant"):
        resp = llm.chat([ChatMessage(role=role, content="hi")])
        assert resp.content == "fake response"
