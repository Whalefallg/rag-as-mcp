"""
LLM 抽象层 (src/libs/llm/base_llm.py)
=======================================
为什么需要这个文件：
  上层业务代码（MetadataEnricher/ChunkRefiner/LLMReranker）只需要一个
  「接收消息列表、返回文本」的接口。BaseLLM 定义这个接口，
  切换 Azure/OpenAI/Ollama/DeepSeek 只需改 settings.yaml，业务代码零修改。
  这种「依赖倒置」是设计中解释可扩展架构的经典示例。

本文件定义 LLM 调用的统一数据结构和抽象接口。

类说明:
  - ChatMessage  : 一条聊天消息的数据结构（role + content），是调用 LLM 的基本单元。
                   role 只能是 "system" / "user" / "assistant" 三种。

  - ChatResponse : LLM 返回结果的数据结构（content + model + token 用量），
                   所有 Provider 的返回都统一包装成这个格式，上层不需要处理各家差异。

  - BaseLLM      : LLM 抽象基类。所有具体 Provider（Azure/OpenAI/Ollama/DeepSeek）
                   都必须继承它并实现 chat() 方法。
                   上层业务代码（MetadataEnricher/Reranker 等）只依赖这个接口，
                   切换 Provider 只需改 settings.yaml，业务代码零修改。
"""
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
