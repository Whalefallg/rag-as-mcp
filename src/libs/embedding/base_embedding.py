"""Embedding 抽象层 (src/libs/embedding/base_embedding.py)"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Embedding 抽象基类"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def embed(self, texts: List[str], trace=None) -> List[List[float]]:
        """
        批量将文本转为向量。

        Args:
            texts: 待向量化的文本列表，例如 ["苹果是什么", "香蕉的营养"]。
            trace: Optional trace shared with the ingestion pipeline.
        Returns:
            与输入等长的向量列表，每个向量是一个浮点数列表。
        Raises:
            ValueError: 输入格式错误。
            RuntimeError: API 调用失败。
        """
        pass

    def _validate_texts(self, texts: List[str]) -> None:
        """校验文本列表，子类在 embed() 开头调用。"""
        if not texts:
            raise ValueError("文本列表不能为空")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"文本[{i}]类型错误，期望 str，实际 {type(text).__name__}")
