"""
MultimodalAssembler (src/core/response/multimodal_assembler.py)
================================================================
为什么需要这个文件：
  当检索结果中包含图片引用（chunk.metadata["image_refs"]）时，
  MCP 响应需要同时返回文本和图片。
  MultimodalAssembler 负责：
    1. 从 RetrievalResult.metadata["image_refs"] 收集图片 ID
    2. 通过 ImageStorage 查找本地文件路径
    3. 读取图片字节并 Base64 编码
    4. 构建 MCP ImageContent 对象（{"type":"image","data":"<b64>","mimeType":"..."}）

  Client 兼容性：
    - Claude Desktop：完整支持图片渲染
    - GitHub Copilot：当前可能仅展示文本，图片作为附加内容
    - 所有 Client：TextContent 始终在 content[0]，图片在后续位置，保证最低兼容性

  降级策略：
    - 图片文件不存在 → 跳过，不阻塞文本响应
    - Base64 编码失败 → 记录日志，跳过该图片
    - ImageStorage 未配置 → 直接返回空列表
"""
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

logger = get_logger(__name__)

# 支持的图片 MIME 类型
_SUPPORTED_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_DEFAULT_MIME = "image/png"


class MultimodalAssembler:
    """
    从检索结果中提取图片并构建 MCP ImageContent 列表。

    Usage:
        assembler = MultimodalAssembler(image_storage=storage)
        image_contents = assembler.assemble(results, max_images=3)
    """

    def __init__(self, image_storage=None) -> None:
        """
        Args:
            image_storage: ImageStorage 实例（可为 None，此时跳过图片处理）
        """
        self._storage = image_storage

    def assemble(
        self,
        results: List[RetrievalResult],
        max_images: int = 3,
        trace: Optional[TraceContext] = None,
    ) -> List[Dict[str, Any]]:
        """
        从检索结果中收集图片，返回 MCP ImageContent 列表。

        Args:
            results:    排好序的检索结果列表。
            max_images: 最多返回的图片数量（避免响应过大）。
        Returns:
            MCP ImageContent 列表，每项格式：
            {"type": "image", "data": "<base64>", "mimeType": "image/png"}
        """
        started = time.monotonic()
        if self._storage is None:
            if trace:
                trace.record_stage(
                    "multimodal_assembly",
                    duration_ms=(time.monotonic() - started) * 1000,
                    image_count=0,
                    storage_available=False,
                )
            return []

        image_contents: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for result in results:
            if len(image_contents) >= max_images:
                break

            collection = result.metadata.get("collection")
            image_refs = _parse_image_refs(
                result.metadata.get("image_refs", [])
            )
            for image_id in image_refs:
                dedup_key = (collection, image_id)
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                content = self._load_image(
                    image_id, collection=collection
                )
                if content is not None:
                    image_contents.append(content)
                    if len(image_contents) >= max_images:
                        break

        if trace:
            trace.record_stage(
                "multimodal_assembly",
                duration_ms=(time.monotonic() - started) * 1000,
                image_count=len(image_contents),
                storage_available=True,
            )
        return image_contents

    def _load_image(
        self,
        image_id: str,
        collection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载单张图片并编码为 Base64 ImageContent。"""
        try:
            if collection is None:
                path_str = self._storage.get_path(image_id)
            else:
                path_str = self._storage.get_path(
                    image_id, collection=collection
                )
            if path_str is None:
                logger.debug(f"Image not found in storage: {image_id}")
                return None

            file_path = Path(path_str)
            if not file_path.exists():
                logger.warning(f"Image file missing: {file_path}")
                return None

            image_bytes = file_path.read_bytes()
            mime_type = _detect_mime(file_path)
            b64_data = base64.standard_b64encode(image_bytes).decode("ascii")

            return {
                "type": "image",
                "data": b64_data,
                "mimeType": mime_type,
            }
        except Exception as e:
            logger.warning(f"Failed to load image {image_id}: {e}")
            return None


def _parse_image_refs(raw) -> List[str]:
    """兼容内存 list 与 Chroma 中 JSON-string 两种 image_refs 表示。"""
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [raw]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return []


def _detect_mime(path: Path) -> str:
    """根据文件扩展名推断 MIME 类型，未知时默认 image/png"""
    mime, _ = mimetypes.guess_type(str(path))
    if mime in _SUPPORTED_MIME:
        return mime
    return _DEFAULT_MIME
