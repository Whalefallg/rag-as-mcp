"""
MetadataEnricher 实现 (src/ingestion/transform/metadata_enricher.py)
=====================================================================
为什么需要这个文件：
  检索结果展示时需要显示「这个结果来自哪里、讲的是什么」——
  裸露的原始 chunk 文本对用户不友好。MetadataEnricher 为每个 chunk 提取
  title（标题）、summary（摘要）、tags（标签），存入 chunk.metadata，
  后续 MCP Tool 的响应可以直接用这些字段渲染结果卡片。
  LLM 模式提取质量最高，规则模式作为兜底，确保字段永远非空。

本文件实现 Chunk 元数据增强器。

类说明:
  - MetadataEnricher : 继承 BaseTransform，为每个 Chunk 生成 title/summary/tags。
                       规则模式（兜底）：取首行作 title，取前 200 字作 summary，
                                        从高频词中提取 tags。
                       LLM 模式（核心）：调用 LLM 生成高质量语义元数据，
                                        输出 JSON 格式：{"title": ..., "summary": ..., "tags": [...]}。
                       降级行为：LLM 失败时回退到规则结果，metadata 标记 enriched_by。
"""
import json
import re
from pathlib import Path
from typing import List, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform

DEFAULT_PROMPT_PATH = "config/prompts/metadata_enrichment.txt"
DEFAULT_PROMPT = """Analyze the following text chunk and return a JSON object with:
- "title": a concise title (max 10 words)
- "summary": a 1-2 sentence summary
- "tags": a list of 3-5 relevant keywords

Return only valid JSON, no markdown wrapping.

Text:
{text}"""


class MetadataEnricher(BaseTransform):
    """Chunk 元数据增强器：规则提取 + 可选 LLM 增强"""

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
        return raw.get("ingestion", {}).get("metadata_enricher", {}).get("use_llm", False)

    def _load_prompt(self, prompt_path: str) -> str:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return DEFAULT_PROMPT

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        result = []
        for chunk in chunks:
            try:
                enriched_meta = self._rule_based_enrich(chunk.text)
                enriched_by = "rule"

                if self._use_llm:
                    llm_meta = self._llm_enrich(chunk.text, trace)
                    if llm_meta is not None:
                        enriched_meta = llm_meta
                        enriched_by = "llm"

                new_chunk = Chunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    index=chunk.index,
                    metadata={**chunk.metadata, **enriched_meta, "enriched_by": enriched_by},
                )
                result.append(new_chunk)
            except Exception as e:
                chunk.metadata["enrich_error"] = str(e)
                result.append(chunk)

        return result

    def _rule_based_enrich(self, text: str) -> dict:
        """规则模式：取首行作 title，截前 200 字作 summary，提取高频词作 tags"""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        title = lines[0][:80] if lines else "Untitled"

        summary = text.strip()[:200]
        if len(text.strip()) > 200:
            summary = summary.rsplit(" ", 1)[0] + "..."

        # 简单提取非停用词中长度 >= 4 的词作为 tags
        words = re.findall(r"\b[a-zA-Z\u4e00-\u9fff]{4,}\b", text)
        freq: dict = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        tags = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:5]]

        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(self, text: str, trace: Optional[TraceContext]) -> Optional[dict]:
        """调用 LLM 生成元数据，失败返回 None"""
        if self._llm is None:
            return None
        try:
            from src.libs.llm.base_llm import ChatMessage
            prompt = self._prompt_template.format(text=text[:2000])  # 限制 token
            response = self._llm.chat([ChatMessage(role="user", content=prompt)])
            raw = response.content.strip()
            # 清理可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            meta = json.loads(raw)
            # 校验必须字段
            if "title" in meta and "summary" in meta and "tags" in meta:
                return meta
            return None
        except Exception:
            return None
