"""
Vision LLM 抽象层 (src/libs/llm/base_vision_llm.py)
=====================================================
为什么需要这个文件：
  Vision LLM 比普通 LLM 多一个「图片输入」参数，接口签名不同。
  BaseVisionLLM 继承 BaseLLM 并新增 chat_with_image()，
  让 ImageCaptioner 可以用统一接口调用不同的 Vision 后端，
  同时 base64 工具方法复用，不需要每个实现重复写。

本文件定义多模态（图文）LLM 调用的统一抽象接口，是 BaseLLM 的扩展。

类说明:
  - BaseVisionLLM : Vision LLM 抽象基类，专门处理"文字 + 图片"的多模态输入。
                    普通 BaseLLM 只能接受文本，BaseVisionLLM 额外支持传入图片
                    （本地文件路径 或 base64 编码的字节串）。
                    在 Ingestion Pipeline 的 ImageCaptioner 阶段被调用：
                    传入图片 + 文档上下文文字，让 Vision LLM 生成图片的文字描述（Caption），
                    描述会被缝合进 Chunk 正文，让纯文本检索链路也能检索到图片内容。
"""
import base64
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse


class BaseVisionLLM(BaseLLM):
    """Vision LLM 抽象基类，支持文字 + 图片的多模态输入"""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        trace=None,
    ) -> ChatResponse:
        """
        发送图文混合请求。

        Args:
            text: 文字提示词（例如图片的上下文描述或分析指令）。
            image_path: 图片本地文件路径（与 image_bytes 二选一）。
            image_bytes: 图片的原始字节数据（与 image_path 二选一）。
            trace: 追踪上下文（可选）。
        Returns:
            ChatResponse: LLM 对图片内容的描述或分析结果。
        Raises:
            ValueError: 未提供图片时抛出。
            RuntimeError: API 调用失败。
        """
        pass

    # chat() 继承自 BaseLLM，Vision LLM 同样支持纯文本对话
    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        raise NotImplementedError(
            "Vision LLM 请使用 chat_with_image() 进行图文对话，"
            "或使用普通 LLM 进行纯文本对话。"
        )

    @staticmethod
    def _load_image_as_base64(image_path: str) -> str:
        """将本地图片文件读取并编码为 base64 字符串，供子类调用。"""
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"图片文件不存在: {image_path}")
        raw = path.read_bytes()
        return base64.b64encode(raw).decode("utf-8")

    @staticmethod
    def _bytes_to_base64(image_bytes: bytes) -> str:
        """将图片字节数据编码为 base64 字符串，供子类调用。"""
        return base64.b64encode(image_bytes).decode("utf-8")
