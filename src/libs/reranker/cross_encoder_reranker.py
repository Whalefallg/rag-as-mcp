"""
Cross-Encoder Reranker 实现 (src/libs/reranker/cross_encoder_reranker.py)
=========================================================================
为什么需要这个文件：
  Cross-Encoder 把 (query, chunk) 联合编码，能捕捉 query 和文本的交互关系，
  精排质量远超 Bi-Encoder（Embedding 的余弦相似度）。
  代价是速度慢 10x+，所以只对 Top-20~30 候选做精排，不用于大规模召回。
"""
from typing import List, Dict, Any, Optional

from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import register_reranker

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 模型的精排重排"""

    def __init__(self, model: str = DEFAULT_MODEL, scorer=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._scorer = scorer

    def _get_scorer(self):
        if self._scorer is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise RuntimeError(
                    "请先安装 sentence-transformers 依赖：pip install sentence-transformers"
                )
            self._scorer = CrossEncoder(self.model)
        return self._scorer

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        trace=None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates

        try:
            scorer = self._get_scorer()
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores: List[float] = scorer.predict(pairs).tolist()

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = score

            ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            if top_k is not None:
                ranked = ranked[:top_k]
            return ranked

        except Exception:
            result = candidates
            if top_k is not None:
                result = candidates[:top_k]
            return result
