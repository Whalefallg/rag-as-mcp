import json
import re
from typing import Optional

from src.core.agentic.types import EvidenceAssessment, QueryAnalysis
from src.libs.llm.base_llm import BaseLLM, ChatMessage


class QueryRewriter:
    def __init__(self, use_llm: bool = False, llm: Optional[BaseLLM] = None):
        self._use_llm = use_llm
        self._llm = llm
        self.llm_calls = 0

    def rewrite(self, query: str, analysis: QueryAnalysis, assessment: EvidenceAssessment) -> str:
        if self._use_llm and self._llm is not None:
            try:
                self.llm_calls += 1
                response = self._llm.chat([ChatMessage("user", "Return JSON only with rewritten_query. Preserve exact identifiers. Remove conversational noise and make the information need explicit. Query: " + query)])
                rewritten = str(json.loads(response.content)["rewritten_query"]).strip()
                if rewritten:
                    return self._preserve_identifiers(query, rewritten)
            except Exception:
                pass
        cleaned = re.sub(r"^(?:请问|麻烦|能否|could you|please|I want to know)\s*", "", query.strip(), flags=re.I)
        cleaned = re.sub(r"\b(?:pls|thx|thanks)\b", "", cleaned, flags=re.I)
        cleaned = " ".join(cleaned.split())
        suffix = " 相关原理、配置、失败模式与处理方法" if analysis.intent.value in {"multi_hop", "comparative"} else " 相关定义、配置与示例"
        rewritten = cleaned + suffix
        return self._preserve_identifiers(query, rewritten)

    @staticmethod
    def _preserve_identifiers(original: str, rewritten: str) -> str:
        identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_.]*(?:\(\))?|\b(?:ERR|HTTP)[-_ ]?\d{3,5}\b", original)
        missing = [item for item in identifiers if item.lower() not in rewritten.lower()]
        return (rewritten + " " + " ".join(missing)).strip()
