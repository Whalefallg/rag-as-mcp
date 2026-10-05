from types import SimpleNamespace
from src.core.types import RetrievalResult
from src.observability.evaluation import retrieval_variant as rv

class FakeQP:
    def process(self,q): return SimpleNamespace(keywords=q.split())
class FakeSparse:
    def retrieve(self,keywords,top_k,collection): return [RetrievalResult(chunk_id="1",score=1,text="x")]

def test_bm25_adapter_uses_query_processor_and_collection():
    adapter=object.__new__(rv.RetrievalVariant); adapter.name="bm25"; adapter.query_processor=FakeQP(); adapter.sparse=FakeSparse()
    assert adapter.search("redis oom",10,"opsbench_v1")[0].chunk_id=="1"

def test_rerank_variant_is_explicitly_skipped_when_disabled():
    adapter=object.__new__(rv.RetrievalVariant); adapter.name="hybrid_rerank"; adapter.reranker=None
    assert "no real reranker" in adapter.skip_reason
