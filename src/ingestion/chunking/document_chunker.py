"""
DocumentChunker 适配器 (src/ingestion/chunking/document_chunker.py)
====================================================================
为什么需要这个文件：
  libs.splitter 是纯工具——只把字符串切成更小的字符串，不知道 Document/Chunk。
  DocumentChunker 是 Ingestion 业务层和 libs 工具层之间的适配器：
  把 Document 喂给 splitter，拿到 List[str] 之后补全 Chunk 的所有业务字段。
  Chunk ID 用内容哈希生成，保证同一 Document 多次切分产生相同 ID（幂等基础）。
  图片引用分发确保 [IMAGE: xxx] 占位符所在的 chunk 记录对应 image_ref，
  后续 ImageCaptioner 才知道要为哪个 chunk 补充图片描述。

本文件实现 Document → List[Chunk] 的业务适配器层。

类说明:
  - DocumentChunker : 连接 libs.splitter（纯文本切分工具）和 Ingestion Pipeline 的适配器。
                      libs.splitter 只做 str→List[str]，不涉及业务对象。
                      DocumentChunker 负责 6 个增值职责：
                        1. Chunk ID 生成（格式：{doc_id}_{index:04d}_{content_hash[:8]}）
                        2. 元数据继承（从 Document 复制 source_path/collection 等）
                        3. 添加 chunk_index 字段
                        4. 建立 source_ref 指向父 Document.id
                        5. 图片引用按需分发（扫描 [IMAGE: id] 占位符，仅分发该 chunk 引用的图片）
                        6. 类型转换（List[str] → List[Chunk]）
"""
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
