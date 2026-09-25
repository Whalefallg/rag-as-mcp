"""
ImageCaptioner 实现 (src/ingestion/transform/image_captioner.py)
================================================================
为什么需要这个文件：
  纯文本的向量检索看不到图片——如果文档里有流程图、数据表格截图，
  用户查询相关内容时这些信息完全检索不到。
  ImageCaptioner 用 Vision LLM 为每张图片生成文字描述（caption），
  把描述缝合进 chunk 正文，让纯文本检索链路也能间接检索到图片内容。
  降级机制：Vision LLM 未配置或失败时标记 has_unprocessed_images=True，
  整条 Pipeline 不中断，文本内容仍然正常摄取。

本文件实现图片 caption 生成器。

类说明:
  - ImageCaptioner : 继承 BaseTransform，为 Chunk 中引用的图片生成文字描述（caption）。
                     启用模式：当 settings.ingestion.image_captioner.enabled=true 且
                               chunk.metadata["image_refs"] 非空时，调用 Vision LLM
                               为每张图片生成 caption，写入 metadata["images"][i]["caption"]。
                     降级模式：Vision LLM 未配置、调用失败或图片读取失败时，
                               保留 image_refs 原样，在 metadata 标记
                               has_unprocessed_images=True，不抛异常，不阻塞 Pipeline。
"""
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
