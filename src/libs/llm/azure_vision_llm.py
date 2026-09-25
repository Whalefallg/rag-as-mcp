"""
Azure Vision LLM 实现 (src/libs/llm/azure_vision_llm.py)
==========================================================
为什么需要这个文件：
  多模态 RAG 需要让 LLM「看」图片。AzureVisionLLM 把图片转为 base64
  并按 OpenAI Vision API 格式发送，让 ImageCaptioner 能用同一套接口处理文本和图片。
  自动压缩超大图片，避免触发 API 的尺寸限制。

本文件实现基于 Azure OpenAI 的多模态（图文）LLM 调用。

类说明:
  - AzureVisionLLM : 继承 BaseVisionLLM，调用 Azure OpenAI 的 GPT-4o / GPT-4-Vision 模型。
                     支持两种图片输入方式：
                       - 本地文件路径：自动读取文件并转为 base64
                       - 原始字节数据：直接转为 base64
                     图片过大时自动压缩（长边不超过 max_image_size），
                     避免超出 API 的图片尺寸限制或消耗过多 token。
                     通过 LLMFactory.create_vision_llm() 创建，
                     settings.yaml 中设置 vision_llm.provider: azure 时工厂使用此实现。
                     主要被 ImageCaptioner 调用，传入文档图片，
                     返回图片的结构化文字描述（流程图逻辑、数据图表数值、截图内容等）。
"""
import os
from typing import Optional

from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import ChatResponse
from src.libs.llm.llm_factory import register_vision_llm

DEFAULT_MAX_IMAGE_SIZE = 2048


@register_vision_llm("azure")
class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM 实现（GPT-4o / GPT-4-Vision）"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = None,
        azure_endpoint: str = None,
        api_version: str = "2024-02-01",
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._api_version = api_version
        self._max_image_size = max_image_size

    def chat_with_image(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        trace=None,
    ) -> ChatResponse:
        if image_path is None and image_bytes is None:
            raise ValueError("必须提供 image_path 或 image_bytes 之一")

        try:
            from openai import AzureOpenAI
        except ImportError:
            raise RuntimeError("请先安装 openai 依赖：pip install openai")

        # 获取图片 base64
        if image_bytes is not None:
            raw = image_bytes
        else:
            raw = open(image_path, "rb").read()

        raw = self._maybe_resize(raw)
        b64 = self._bytes_to_base64(raw)
        mime_type = self._detect_mime(b64)

        try:
            client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }],
            )
            return ChatResponse(
                content=response.choices[0].message.content,
                model=response.model,
            )
        except Exception as e:
            raise RuntimeError(f"Azure Vision API 调用失败 [{self.model}]: {e}") from e

    def _maybe_resize(self, image_bytes: bytes) -> bytes:
        """图片长边超过 max_image_size 时按比例压缩，否则原样返回。"""
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            if max(w, h) <= self._max_image_size:
                return image_bytes
            scale = self._max_image_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format=img.format or "PNG")
            return buf.getvalue()
        except ImportError:
            return image_bytes  # Pillow 未安装，跳过压缩

    @staticmethod
    def _detect_mime(b64: str) -> str:
        """根据 base64 数据头判断图片 MIME 类型。"""
        import base64 as b64mod
        header = b64mod.b64decode(b64[:16])
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        return "image/png"
