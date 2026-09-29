"""ImageCaptioner 实现 (src/ingestion/transform/image_captioner.py)"""
from pathlib import Path
from typing import List, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform

DEFAULT_PROMPT_PATH = "config/prompts/image_captioning.txt"
DEFAULT_PROMPT = """Describe the content of this image concisely (2-3 sentences).
Focus on: diagrams/charts data, text visible in the image, and key visual elements.
Context from surrounding text: {context}"""


class ImageCaptioner(BaseTransform):
    """图片 caption 生成器：调用 Vision LLM，失败时优雅降级"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vision_llm=None,
        prompt_path: Optional[str] = None,
    ):
        self._enabled = self._read_enabled_flag(settings)
        self._vision_llm = vision_llm
        self._prompt_template = self._load_prompt(prompt_path or DEFAULT_PROMPT_PATH)

    def _read_enabled_flag(self, settings: Optional[Settings]) -> bool:
        if settings is None:
            return False
        raw = settings.raw_config or {}
        return raw.get("ingestion", {}).get("image_captioner", {}).get("enabled", False)

    def _load_prompt(self, prompt_path: str) -> str:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return DEFAULT_PROMPT

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        result = []
        for chunk in chunks:
            image_refs = chunk.metadata.get("image_refs", [])
            images = chunk.metadata.get("images", [])

            if not self._enabled or not image_refs or self._vision_llm is None:
                # 降级：标记未处理图片
                if image_refs:
                    chunk.metadata["has_unprocessed_images"] = True
                result.append(chunk)
                continue

            # 为每张图片生成 caption
            updated_images = list(images)
            all_success = True
            for i, img_info in enumerate(updated_images):
                img_path = img_info.get("path", "")
                if not img_path or not Path(img_path).exists():
                    all_success = False
                    continue
                try:
                    prompt = self._prompt_template.format(
                        context=chunk.text[:500]
                    )
                    response = self._vision_llm.chat_with_image(
                        text=prompt,
                        image_path=img_path,
                    )
                    updated_images[i] = {**img_info, "caption": response.content.strip()}
                except Exception:
                    all_success = False
                    continue

            new_meta = {**chunk.metadata, "images": updated_images}
            if not all_success:
                new_meta["has_unprocessed_images"] = True

            result.append(Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                index=chunk.index,
                metadata=new_meta,
            ))

        return result
