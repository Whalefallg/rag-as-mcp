"""LLM 抽象层 (src/libs/llm/base_llm.py)"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """一条聊天消息"""
    role: str    # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    """LLM 的响应结果"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": 10, "completion_tokens": 50}


class BaseLLM(ABC):
    """LLM 抽象基类"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        """
        发送聊天请求。

        Args:
            messages: 消息列表，至少包含一条 user 消息。
        Returns:
            ChatResponse: 统一格式的响应结果。
        Raises:
            ValueError: 消息格式错误。
            RuntimeError: API 调用失败。
        """
        pass

    def _validate_messages(self, messages: List[ChatMessage]) -> None:
        """校验消息列表格式，子类在 chat() 开头调用。"""
        if not messages:
            raise ValueError("消息列表不能为空")
        for msg in messages:
            if msg.role not in ("system", "user", "assistant"):
                raise ValueError(f"无效的消息角色: {msg.role}")
            if not msg.content:
                raise ValueError("消息内容不能为空")
