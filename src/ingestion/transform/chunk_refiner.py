"""ChunkRefiner 实现 (src/ingestion/transform/chunk_refiner.py)"""
import re
from pathlib import Path
from typing import List, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform

DEFAULT_PROMPT_PATH = "config/prompts/chunk_refinement.txt"
DEFAULT_PROMPT = """You are a text cleaning assistant.
Clean the following text chunk by removing noise (headers, footers, page numbers, excessive whitespace, HTML artifacts) while preserving all meaningful content.
Return only the cleaned text, nothing else.

Text:
{text}"""


class ChunkRefiner(BaseTransform):
    """Chunk 文本净化器：规则去噪 + 可选 LLM 增强"""

    # 规则去噪的正则模式
    _PATTERNS = [
        (re.compile(r"<!--.*?-->", re.DOTALL), ""),               # HTML 注释
        (re.compile(r"<[^>]+>"), ""),                              # HTML 标签
        (re.compile(r"^[-=_*]{3,}\s*$", re.MULTILINE), ""),       # 分隔线
        (re.compile(r"^(Page|页码?)\s*\d+\s*$", re.MULTILINE | re.IGNORECASE), ""),  # 页码行
        (re.compile(r"\n{3,}"), "\n\n"),                           # 超过两个连续换行
        (re.compile(r"[ \t]{2,}"), " "),                           # 连续空格/tab
    ]

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm=None,
        prompt_path: Optional[str] = None,
    ):
        self._use_llm = self._read_use_llm_flag(settings)
        self._llm = llm
        self._prompt_template = self._load_prompt(prompt_path or DEFAULT_PROMPT_PATH)

    def _read_use_llm_flag(self, settings: Optional[Settings]) -> bool:
        if settings is None:
            return False
        raw = settings.raw_config or {}
        return raw.get("ingestion", {}).get("chunk_refiner", {}).get("use_llm", False)

    def _load_prompt(self, prompt_path: str) -> str:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return DEFAULT_PROMPT

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        result = []
        for chunk in chunks:
            try:
                refined = self._rule_based_refine(chunk.text)
                refined_by = "rule"

                if self._use_llm:
                    llm_result = self._llm_refine(refined, trace)
                    if llm_result is not None:
                        refined = llm_result
                        refined_by = "llm"

                new_chunk = Chunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=refined.strip(),
                    index=chunk.index,
                    metadata={**chunk.metadata, "refined_by": refined_by},
                )
                result.append(new_chunk)
            except Exception as e:
                # 单个 chunk 失败不阻塞整体，保留原文
                chunk.metadata["refine_error"] = str(e)
                result.append(chunk)

        return result

    def _rule_based_refine(self, text: str) -> str:
        """应用规则模式对文本去噪"""
        result = text
        for pattern, replacement in self._PATTERNS:
            result = pattern.sub(replacement, result)
        return result.strip()

    def _llm_refine(self, text: str, trace: Optional[TraceContext]) -> Optional[str]:
        """调用 LLM 重写文本，失败返回 None（调用方回退到规则结果）"""
        if self._llm is None:
            return None
        try:
            from src.libs.llm.base_llm import ChatMessage
            prompt = self._prompt_template.format(text=text)
            response = self._llm.chat([ChatMessage(role="user", content=prompt)])
            refined = response.content.strip()
            # 空响应视为失败
            return refined if refined else None
        except Exception:
            return None
