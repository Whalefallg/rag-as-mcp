"""
PDF Loader 实现 (src/libs/loader/pdf_loader.py)
================================================
为什么需要这个文件：
  PDF 是知识库最常见的文档格式，但解析比纯文本复杂得多——
  需要处理多栏布局、图片提取、表格识别。
  PdfLoader 用 PyMuPDF 逐页提取文本和图片，生成 Document 对象供 Pipeline 消费。
  图片以字节存储在 Document.metadata["images"] 中，供 ImageCaptioner 使用。

本文件实现基于 PyMuPDF (fitz) 的 PDF 文档加载。

类说明:
  - PdfLoader : 继承 BaseLoader，解析 PDF 文件并输出统一的 Document 对象。
                文本处理：按页提取文本，拼接为 Markdown 风格的规范化文本。
                图片处理：提取每页嵌入图片，保存为 PNG 到 data/images/{doc_hash}/，
                          在文本中对应位置插入 [IMAGE: {image_id}] 占位符，
                          并在 metadata["images"] 中记录 ImageRef 信息。
                降级行为：图片提取失败不阻塞文本解析，仅记录 warning 日志。
                文档 ID：文件路径的 SHA256 前 16 字符，确保同一文件 ID 稳定。
"""
import hashlib
import os
from pathlib import Path
from typing import List, Optional

from src.libs.loader.base_loader import BaseLoader
from src.core.types import Document, ImageRef


def _compute_doc_id(path: str) -> str:
    """根据文件路径计算确定性文档 ID（路径 SHA256 前 16 字符）"""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


class PdfLoader(BaseLoader):
    """PDF 文档加载器（依赖 PyMuPDF）"""

    def __init__(self, image_output_dir: str = "data/images"):
        """
        Args:
            image_output_dir: 图片保存的根目录，图片按 doc_hash 分子目录存放。
        """
        self._image_output_dir = image_output_dir

    def load(self, path: str) -> Document:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF 文件不存在: {path}")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("请先安装 PyMuPDF：pip install pymupdf")

        doc_id = _compute_doc_id(path)
        doc = fitz.open(path)

        text_parts: List[str] = []
        image_refs: List[dict] = []
        text_offset = 0

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")

            # 提取当前页的嵌入图片
            page_images = self._extract_page_images(
                doc, page, page_num, doc_id, path
            )

            for seq, img_ref in enumerate(page_images):
                placeholder = f"[IMAGE: {img_ref['image_id']}]"
                # 将图片占位符追加到当前页文本末尾
                page_text = page_text + "\n" + placeholder + "\n"

                img_ref["text_offset"] = text_offset + len(page_text) - len(placeholder) - 1
                img_ref["text_length"] = len(placeholder)
                image_refs.append(img_ref)

            text_parts.append(page_text)
            text_offset += len(page_text)

        full_text = "\n\n".join(text_parts)

        metadata = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
            "page_count": len(doc),
            "title": Path(path).stem,
            "images": image_refs,
        }

        doc.close()
        return Document(id=doc_id, source=path, text=full_text, metadata=metadata)

    def _extract_page_images(
        self,
        doc,
        page,
        page_num: int,
        doc_hash: str,
        source_path: str,
    ) -> List[dict]:
        """提取页面图片，保存到磁盘，返回 ImageRef dict 列表。失败时返回空列表。"""
        results = []
        try:
            img_dir = Path(self._image_output_dir) / doc_hash
            img_dir.mkdir(parents=True, exist_ok=True)

            image_list = page.get_images(full=True)
            for seq, img_info in enumerate(image_list):
                xref = img_info[0]
                image_id = f"{doc_hash}_{page_num}_{seq}"
                img_path = img_dir / f"{image_id}.png"

                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_path.write_bytes(img_bytes)

                    results.append({
                        "image_id": image_id,
                        "path": str(img_path),
                        "page": page_num,
                        "seq": seq,
                        "width": base_image.get("width"),
                        "height": base_image.get("height"),
                        "text_offset": 0,   # 由调用方回填
                        "text_length": 0,   # 由调用方回填
                        "position": {},
                    })
                except Exception:
                    # 单张图片失败，跳过，不影响其他图片
                    continue
        except Exception:
            # 整页图片提取失败，降级返回空列表
            pass
        return results
