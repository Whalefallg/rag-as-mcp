"""
所有工厂集成测试 (tests/integration/test_all_factories.py)
===========================================================
为什么需要这个文件：
  各个工厂（LLM/Embedding/Splitter/VectorStore）都是靠注册表动态创建的。
  单独测试某个工厂只能验证它自己，集成测试验证"所有工厂能同时工作"——
  比如注册表没有相互污染、Settings 能正确路由到每个工厂。
  这一个测试跑通，基本能保证 Ingestion Pipeline 的依赖注入链路是通的。
"""
import pytest
from src.libs.llm.llm_factory import create_llm, register_llm
from src.libs.embedding.embedding_factory import create_embedding, register_embedding
from src.libs.splitter.splitter_factory import create_splitter, register_splitter
from src.libs.vector_store.vector_store_factory import create_vector_store, register_vector_store
from src.libs.reranker.reranker_factory import create_reranker
from src.libs.evaluator.evaluator_factory import create_evaluator, register_evaluator
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.core.settings import load_settings

@register_llm("test_llm")
class FactoryLLM(BaseLLM):
    def chat(self, messages):
        return ChatResponse(content="test", model=self.model)

@register_embedding("test_emb")
class FactoryEmbedding(BaseEmbedding):
    def embed(self, texts, trace=None):
        return [[1.0] * 3 for _ in texts]

@register_splitter("test_split")
class FactorySplitter(BaseSplitter):
    def split_text(self, text, trace=None):
        return [text[:100], text[100:]]

@register_vector_store("test_store")
class FactoryVectorStore(BaseVectorStore):
    def upsert(self, records, trace=None):
        pass
    def query(self, vector, top_k, filters=None, trace=None):
        return []
    def get_by_ids(self, ids):
        return []
    def delete_by_metadata(self, filter):
        return 0

@register_evaluator("test_eval")
class FactoryEvaluator(BaseEvaluator):
    def evaluate(self, query, retrieved_chunks, generated_answer=None, ground_truth=None):
        return {"hit_rate": 0.9}

def test_all_factories_work_together():
    """测试所有工厂可以协同工作"""
    settings = load_settings("config/settings.example.yaml")
    
    # 测试 Reranker（使用默认的 none）
    reranker = create_reranker(settings)
    assert reranker is not None
    
    # 测试 Evaluator
    evaluator = create_evaluator("test_eval")
    assert evaluator is not None
    metrics = evaluator.evaluate("test query", [])
    assert "hit_rate" in metrics
