"""DocumentChunker 适配器 (src/ingestion/chunking/document_chunker.py)"""
import hashlib
import re
from typing import List, Dict, Any

from src.core.types import Document, Chunk
from src.core.settings import Settings
from src.libs.splitter.splitter_factory import create_splitter

# 匹配文本中的图片占位符，如 [IMAGE: abc123_0_1]
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class DocumentChunker:
    """Document → List[Chunk] 的业务适配器"""

    def __init__(self, settings: Settings):
        self._splitter = create_splitter(settings)

    def split_document(self, document: Document) -> List[Chunk]:
        """
        将 Document 切分为 Chunk 列表。

        Args:
            document: 已加载的文档对象。
        Returns:
            切分后的 Chunk 列表，顺序与原文一致。
        """
        raw_chunks = self._splitter.split_text(document.text)
        # 空文档返回空列表
        if not raw_chunks:
            return []

        # 构建文档级 images 索引，便于 O(1) 查找
        doc_images: Dict[str, dict] = {}
        for img in document.metadata.get("images", []):
            doc_images[img["image_id"]] = img

        chunks = []
        for index, text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            metadata = self._build_metadata(document, index, text, doc_images)
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=document.id,
                text=text,
                index=index,
                metadata=metadata,
            ))
        return chunks

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """
        生成确定性 Chunk ID。
        格式：{doc_id}_{index:04d}_{content_hash[:8]}
        同一 Document 重复切分产生相同的 ID 序列。
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{doc_id}_{index:04d}_{content_hash[:8]}"

    def _build_metadata(
        self,
        document: Document,
        index: int,
        text: str,
        doc_images: Dict[str, dict],
    ) -> Dict[str, Any]:
        """
        构建 Chunk 的 metadata：继承文档级字段，添加 chunk 级字段，
        并按需分发该 Chunk 实际引用的图片。
        """
        # 从 Document 继承所有 metadata（不含 images，images 按需分发）
        metadata = {
            k: v for k, v in document.metadata.items()
            if k != "images"
        }
        metadata["chunk_index"] = index
        metadata["content_hash"] = hashlib.sha256(text.encode()).hexdigest()

        # 扫描占位符，分发该 chunk 实际引用的图片子集
        referenced_ids = _IMAGE_PLACEHOLDER_RE.findall(text)
        if referenced_ids:
            metadata["image_refs"] = referenced_ids
            metadata["images"] = [
                doc_images[img_id]
                for img_id in referenced_ids
                if img_id in doc_images
            ]

        return metadata
