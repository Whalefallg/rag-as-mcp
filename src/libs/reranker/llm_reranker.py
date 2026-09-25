"""
LLM Reranker 实现 (src/libs/reranker/llm_reranker.py)
======================================================
为什么需要这个文件：
  LLM 对候选文本的「理解力」最强，能考虑语境和推理关系，
  精排质量最高但 API 成本也最高。适合对召回质量要求极高、候选数量少的场景。
  失败时静默回退到原始顺序，不阻塞整条 Retrieval 链路。
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import register_reranker

DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"
DEFAULT_PROMPT_TEMPLATE = """You are a relevance ranking assistant.
Given the query and candidate chunks, output a JSON array of chunk IDs ranked by relevance (most relevant first).

Query: {query}

Candidates:
{candidates}

Output only a JSON array of IDs, e.g. ["id1", "id2", "id3"]"""


@register_reranker("llm")
class LLMReranker(BaseReranker):
    """基于 LLM 的精排重排"""

    def __init__(self, model: str = None, llm=None, prompt_path: str = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._llm = llm
        self._prompt_template = self._load_prompt(prompt_path or DEFAULT_PROMPT_PATH)

    def _load_prompt(self, prompt_path: str) -> str:
        path = Path(prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return DEFAULT_PROMPT_TEMPLATE

    def _get_llm(self):
        if self._llm is None:
            raise RuntimeError(
                "LLMReranker 需要注入 LLM 实例，请通过工厂创建时传入 llm 参数"
            )
        return self._llm

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        trace=None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates

        candidates_text = "\n".join(
            f"[{c['id']}] {c.get('text', '')[:200]}" for c in candidates
        )
        prompt = self._prompt_template.format(
            query=query,
            candidates=candidates_text,
        )

        try:
            from src.libs.llm.base_llm import ChatMessage
            llm = self._get_llm()
            response = llm.chat([ChatMessage(role="user", content=prompt)])
            raw = response.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            ranked_ids: List[str] = json.loads(raw)
        except Exception:
            result = candidates
            if top_k is not None:
                result = candidates[:top_k]
            return result

        id_to_candidate = {c["id"]: c for c in candidates}
        reranked = [id_to_candidate[cid] for cid in ranked_ids if cid in id_to_candidate]
        remaining = [c for c in candidates if c["id"] not in set(ranked_ids)]
        result = reranked + remaining
        if top_k is not None:
            result = result[:top_k]
        return result
